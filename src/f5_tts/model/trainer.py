from __future__ import annotations

import gc
import math
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
import comet_ml

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists
import traceback

from pynvml import *


def print_gpu_utilization(device_id):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def remap_time_embed_weights(state_dict):
    """Remap single time_embed to dual t_time_embed and r_time_embed"""
    keys_to_remap = []
    
    # Find all time_embed keys
    for key in state_dict.keys():
        if "time_embed." in key and "t_time_embed." not in key and "r_time_embed." not in key:
            keys_to_remap.append(key)
    
    # Create mappings for both embedders
    for key in keys_to_remap:
        # Replace time_embed with t_time_embed and r_time_embed
        t_key = key.replace("time_embed.", "t_time_embed.")
        r_key = key.replace("time_embed.", "r_time_embed.")
        
        # Clone the weights to both embedders
        state_dict[t_key] = state_dict[key].clone()
        state_dict[r_key] = state_dict[key].clone()
    
    # Remove original keys
    for key in keys_to_remap:
        del state_dict[key]
    
    return state_dict

# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "comet",  # "comet" | "tensorboard" | None
        comet_project="test_f5-tts",  # Renamed from wandb_project
        comet_workspace=None,  # Comet workspace name
        comet_experiment_name="test_run",  # Renamed from wandb_run_name
        comet_api_key=None,  # Optional: can also be set via environment variable
        comet_resume_id: str = None,  # Renamed from wandb_resume_id
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        model_cfg_dict: dict = dict(),  # training config
        gradient_checkpointing: bool = True,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.log_samples = log_samples

        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        self.experiment = None
        if self.accelerator.is_main_process:
            # Initialize Comet ML experiment
            if exists(comet_resume_id):
                # Resume existing experiment
                self.experiment = comet_ml.ExistingExperiment(
                    previous_experiment=comet_resume_id,
                )
            else:
                # Create new experiment
                self.experiment = comet_ml.Experiment(
                    project_name=comet_project,
                    workspace=comet_workspace,
                    experiment_name=comet_experiment_name,
                )
            
            # Log hyperparameters
            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.experiment.log_parameters(model_cfg_dict)

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{comet_experiment_name}")

        self.model = model

        if gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                if self.is_main:
                    print("Gradient checkpointing enabled")
            else:
                if self.is_main:
                    print("Model does not support gradient checkpointing")

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):  # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(
                f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu"
            )

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        # checkpoint["ema_model_state_dict"] = remap_time_embed_weights(
        #         checkpoint["ema_model_state_dict"]
        # )
        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            # checkpoint["model_state_dict"] = remap_time_embed_weights(
            #     checkpoint["model_state_dict"]
            # )

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            # self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # if self.scheduler:
            #     self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
            print('F5-TTS INFO: Successfully Loading from pretrained model')
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            update = 0

        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )
            try:
                for batch in current_dataloader:
                    with self.accelerator.accumulate(self.model):
                        text_inputs = batch["text"]
                        mel_spec = batch["mel"].permute(0, 2, 1)
                        mel_lengths = batch["mel_lengths"]

                        # TODO. add duration predictor training
                        if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                            dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                            self.experiment.log_metric("duration loss", dur_loss.item(), step=global_update)

                        loss, _, _ = self.model(
                            mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                        )
                        self.accelerator.backward(loss)

                        if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        if global_update % 10 == 0 and self.is_main:
                            self.ema_model.update()

                        global_update += 1
                        progress_bar.update(1)
                        progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                    if self.accelerator.is_local_main_process:
                        self.experiment.log_metric("loss", loss.item(), step=global_update)
                        self.experiment.log_metric("lr", self.scheduler.get_last_lr()[0], step=global_update)

                        if self.logger == "tensorboard":
                            self.writer.add_scalar("loss", loss.item(), global_update)
                            self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                    if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                        self.save_checkpoint(global_update, last=True)

                    if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                        self.save_checkpoint(global_update)

                        if self.log_samples and self.accelerator.is_local_main_process:
                            ref_audio_len = mel_lengths[0]
                            infer_text = [
                                text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                            ]
                            with torch.inference_mode():
                                generated, _ = self.accelerator.unwrap_model(self.model).mfsample(
                                    cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                    text=infer_text,
                                    duration=ref_audio_len * 2,
                                    steps=nfe_step,
                                    cfg_strength=cfg_strength,
                                    sway_sampling_coef=sway_sampling_coef,
                                )
                                generated = generated.to(torch.float32)
                                gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                                ref_mel_spec = batch["mel"][0].unsqueeze(0)
                                if self.vocoder_name == "vocos":
                                    gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                    ref_audio = vocoder.decode(ref_mel_spec).cpu()
                                elif self.vocoder_name == "bigvgan":
                                    gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                    ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                            gen_audio_path = f"{log_samples_path}/update_{global_update}_gen.wav"
                            ref_audio_path = f"{log_samples_path}/update_{global_update}_ref.wav"
                            torchaudio.save(gen_audio_path, gen_audio, target_sample_rate)
                            torchaudio.save(ref_audio_path, ref_audio, target_sample_rate)
                            
                            # Log audio files to Comet ML
                            if self.logger == "comet" and self.experiment:
                                self.experiment.log_audio(
                                    gen_audio_path, 
                                    name=f"generated_audio_update_{global_update}",
                                    step=global_update
                                )
                                self.experiment.log_audio(
                                    ref_audio_path, 
                                    name=f"reference_audio_update_{global_update}",
                                    step=global_update
                                )

                            self.model.train()

                    if global_update % 100 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    print(f"F5-TTS trained ok with shape {mel_lengths.shape} and {mel_spec.shape}")
                    print(f"device {self.accelerator.device.index} is using {print_gpu_utilization(self.accelerator.device.index)}")
                    del text_inputs, mel_spec, mel_lengths
                    
            except Exception as e:
                print(f"F5-TTS ERROR: {e} with shape of data {mel_lengths.shape} and {mel_spec.shape}")
                print(f"**** Few leng of failed {mel_lengths[:10]}")
                self.save_checkpoint(global_update)
                traceback.print_exc()
                raise e

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
