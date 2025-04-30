# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

import dino_utils
import vision_transformer as vits
from vision_transformer import DINOHead

from infodrop_resnext import resnext50_32x4d, load_dino_mugs
from huggingface_hub import hf_hub_download


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small'] + torchvision_archs, help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels of input square patches - default 16 (for 16x16 patches). If <16, we recommend disabling mixed precision training (--use_fp16 false) to avoid instabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=dino_utils.bool_flag, help="""Whether or not to weight normalize the last layer of the DINO head. Not normalizing leads to better performance but can make the training unstable. In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.9995, type=float, help="""Base EMA parameter for teacher update. The value is increased to 1 during training with cosine schedule. We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=dino_utils.bool_flag, help="Whether to use batch normalizations in projection head (Default: False)")

    # Teacher temperature parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float, help="""Initial value for the teacher temperature: 0.04 works well in most cases. Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup) of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int, help='Number of warmup epochs for the teacher temperature')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=dino_utils.bool_flag, default=False, help="""Whether or not to use half precision for training. We recommend disabling for small patch sizes and bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.0, help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=1.0, help="""Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size', default=512, type=int, help='total batch size on all GPUs')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=0, type=int, help="""Number of epochs during which we keep the output layer fixed.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of linear warmup (highest LR used during training).""")
    parser.add_argument("--warmup_epochs", default=0, type=int, help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=0.0005, help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.), help="""Scale range of the cropped image before resizing, relatively to the origin image. Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small local views to generate. Set this parameter to 0 to disable multi-crop training.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4), help="""Scale range of the cropped image before resizing, relatively to the origin image.""")

    # Misc
    parser.add_argument("--data_path", default="", type=str, help="""path to dataset""")
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--seed', default=1, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--save_prefix", default="", type=str, help="""prefix for saving checkpoint and log files""")
    
    # Lightning specific
    parser.add_argument('--accelerator', default='auto', type=str, help='Accelerator to use for training')
    parser.add_argument('--devices', default='auto', type=str, help='Devices to use for training')
    parser.add_argument('--precision', default='32-true', type=str, help='Precision for training')
    
    # Wandb specific
    parser.add_argument('--wandb_project', default="dino-training", type=str, help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None, type=str, help='Weights & Biases entity (team) name')
    
    return parser


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.5)], p=0.9),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(0.1),
            dino_utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class DINOLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        
        # Build student and teacher networks
        args.arch = args.arch.replace("deit", "vit")
        if args.arch in vits.__dict__.keys():
            student = vits.__dict__[args.arch](patch_size=args.patch_size, drop_path_rate=0.1)
            teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
            self.embed_dim = student.embed_dim
        elif args.arch in torchvision_models.__dict__.keys():
            student = resnext50_32x4d(pretrained=False)
            model_name = "dino_sfp_resnext50"
            checkpoint = hf_hub_download(repo_id="eminorhan/"+model_name, filename=model_name+".pth")
            load_dino_mugs(student, checkpoint, "student")

            teacher = resnext50_32x4d(pretrained=False)
            checkpoint = hf_hub_download(repo_id="eminorhan/"+model_name, filename=model_name+".pth")
            load_dino_mugs(teacher, checkpoint, "teacher")

            self.embed_dim = student.fc.weight.shape[1]
            print(f'Embedding dimension of student & teacher nets: {self.embed_dim}')
        else:
            raise ValueError(f"Unknown architecture: {args.arch}")

        # Multi-crop wrapper handles forward with inputs of different resolutions
        self.student = dino_utils.MultiCropWrapper(
            student, 
            DINOHead(
                self.embed_dim, 
                args.out_dim, 
                use_bn=args.use_bn_in_head, 
                norm_last_layer=args.norm_last_layer
            )
        )
        
        self.teacher = dino_utils.MultiCropWrapper(
            teacher, 
            DINOHead(
                self.embed_dim, 
                args.out_dim, 
                args.use_bn_in_head
            )
        )
        
        # Synchronize batch norms (if any)
        if dino_utils.has_batchnorms(self.student):
            self.student = nn.SyncBatchNorm.convert_sync_batchnorm(self.student)
            self.teacher = nn.SyncBatchNorm.convert_sync_batchnorm(self.teacher)
        
        # Teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict())
        
        # There is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        # Set up loss function
        self.dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        )
        
        # Track the current epoch for loss computation
        self.current_epoch_from_dino_loss = 0
        
        # Initialize momentum and weight decay schedules
        self.momentum_values = None
        self.weight_decay_values = None
        self.steps_per_epoch = None
        
    def on_train_start(self):
        # Update epoch counter for loss
        self.current_epoch_from_dino_loss = self.current_epoch
        
        # Get dataloader length now that it's definitely initialized
        self.steps_per_epoch = len(self.trainer.train_dataloader)
        
        # Create momentum schedule
        self.momentum_values = dino_utils.cosine_scheduler(
            self.args.momentum_teacher, 
            1,
            self.args.epochs,
            self.steps_per_epoch
        )
        
        # Create weight decay schedule
        self.weight_decay_values = dino_utils.cosine_scheduler(
            self.args.weight_decay, 
            self.args.weight_decay_end, 
            self.args.epochs,
            self.steps_per_epoch
        )
        
        # Log model architecture to wandb
        if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "watch"):
            self.logger.experiment.watch(
                self.student,
                log="gradients",
                log_freq=100
            )
    
    def configure_optimizers(self):
        # Get parameter groups (with weight decay differentiations)
        params_groups = dino_utils.get_params_groups(self.student)
        
        # Setup optimizer based on args
        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(params_groups)
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
        elif self.args.optimizer == "lars":
            optimizer = dino_utils.LARS(params_groups)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
            
        return optimizer
    
    def training_step(self, batch, batch_idx):
        images, _ = batch
        
        # Update weight decay according to schedule
        if self.momentum_values is not None:  # Only after on_train_start has been called
            global_step = self.trainer.global_step
            for i, param_group in enumerate(self.optimizers().param_groups):
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = self.weight_decay_values[global_step % self.steps_per_epoch]
            
            # Also update learning rate manually (simpler than using scheduler)
            lr_schedule = dino_utils.cosine_scheduler(
                self.args.lr,
                self.args.min_lr,
                self.args.epochs,
                self.steps_per_epoch,
                warmup_epochs=self.args.warmup_epochs
            )
            for param_group in self.optimizers().param_groups:
                param_group["lr"] = lr_schedule[global_step % self.steps_per_epoch]
        
        # Teacher and student forward passes
        teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = self.student(images)
        loss = self.dino_loss(student_output, teacher_output, self.current_epoch)
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        if self.momentum_values is not None:
            current_lr = self.optimizers().param_groups[0]["lr"]
            current_wd = self.optimizers().param_groups[0]["weight_decay"]
            momentum_value = self.momentum_values[global_step % self.steps_per_epoch]
            
            # Log detailed metrics
            metrics = {
                "learning_rate": current_lr,
                "weight_decay": current_wd,
                "momentum_value": momentum_value,
                "epoch": self.current_epoch,
                "epoch_fraction": (batch_idx + 1) / self.steps_per_epoch,
                "global_step": global_step,
            }
            
            self.log_dict(metrics, prog_bar=False, sync_dist=True)
            
            # Every 500 steps, log a sample image to wandb
            if batch_idx % 500 == 0 and batch_idx > 0:
                # Log a sample of the training images
                try:
                    # Convert the first global crop image to a wandb Image
                    sample_img = images[0][0].detach().cpu()
                    norm_sample = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min())
                    self.logger.experiment.log({"sample_image": wandb.Image(norm_sample)})
                except Exception as e:
                    print(f"Failed to log image: {e}")
        
        # Cancel gradients for the last layer if needed
        if self.current_epoch < self.args.freeze_last_layer:
            for param in self.student.head.last_layer.parameters():
                param.grad = None
                
        return loss
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # Override Lightning's gradient clipping to use our custom clip function
        if self.args.clip_grad > 0:
            dino_utils.clip_gradients(self.student, self.args.clip_grad)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA update for the teacher
        if self.momentum_values is not None:  # Only after on_train_start has been called
            global_step = self.trainer.global_step
            m = self.momentum_values[global_step % self.steps_per_epoch]  # momentum parameter
            with torch.no_grad():
                for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def on_save_checkpoint(self, checkpoint):
        """
        Add custom state to the checkpoint to include teacher model and DINO loss.
        """
        checkpoint['teacher_state_dict'] = self.teacher.state_dict()
        checkpoint['dino_loss_state_dict'] = self.dino_loss.state_dict()
        
    def on_load_checkpoint(self, checkpoint):
        """
        Load custom state from the checkpoint including teacher model and DINO loss.
        """
        # Load teacher state
        if 'teacher_state_dict' in checkpoint:
            self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
            
        # Load DINO loss state
        if 'dino_loss_state_dict' in checkpoint:
            self.dino_loss.load_state_dict(checkpoint['dino_loss_state_dict'])

    def on_train_end(self):
        # Log final model parameters
        if hasattr(self.logger, "experiment"):
            # Calculate total parameters and trainable parameters
            total_params = sum(p.numel() for p in self.student.parameters())
            trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
            
            # Log final summary metrics
            self.logger.experiment.summary["total_parameters"] = total_params
            self.logger.experiment.summary["trainable_parameters"] = trainable_params
            self.logger.experiment.summary["final_train_loss"] = self.trainer.callback_metrics.get("train_loss", 0.0)
            self.logger.experiment.summary["total_epochs"] = self.current_epoch
            self.logger.experiment.summary["total_steps"] = self.global_step


class DINODataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
    def setup(self, stage=None):
        # Transform for data augmentation
        transform = DataAugmentationDINO(
            self.args.global_crops_scale, 
            self.args.local_crops_scale, 
            self.args.local_crops_number
        )
        
        # Load dataset
        self.dataset = ImageFolder(self.data_path, transform=transform)
        print(f'Data loaded: dataset contains {len(self.dataset)} images')
        
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )


def main():
    parser = argparse.ArgumentParser('DINO Lightning', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Setup CUDA
    cudnn.benchmark = True
    
    # Print arguments
    print("git:\n  {}\n".format(dino_utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # Initialize data module
    data_module = DINODataModule(args)
    
    # Initialize model
    model = DINOLightningModule(args)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=args.save_prefix + "{epoch:04d}",
        every_n_epochs=1,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup Weights & Biases logger
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.save_prefix if args.save_prefix else None,
        save_dir=args.output_dir,
        log_model=True,
        tags=["dino", args.arch],
        config=vars(args),
        entity=args.wandb_entity,
    )
    
    # Add Weights & Biases callback to monitor system metrics
    wandb_callback = None
    try:
        from pytorch_lightning.callbacks import DeviceStatsMonitor
        device_stats = DeviceStatsMonitor()
        callbacks = [checkpoint_callback, lr_monitor, device_stats]
    except ImportError:
        print("DeviceStatsMonitor not available in this version of PyTorch Lightning.")
        callbacks = [checkpoint_callback, lr_monitor]
    
    # Check for existing checkpoint
    checkpoint_path = "/workspace/last.ckpt"
    print(f"Resuming from checkpoint: {checkpoint_path}")
    # last_checkpoint_path = os.path.join(args.output_dir, "last.ckpt")
    # if os.path.exists(last_checkpoint_path):
    #     checkpoint_path = last_checkpoint_path
    # else:
    #     # Check if there's a checkpoint from the old format to convert
    #     old_checkpoint_path = os.path.join(args.output_dir, args.save_prefix + "_checkpoint.pth")
    #     if os.path.exists(old_checkpoint_path):
    #         print(f"Found old format checkpoint: {old_checkpoint_path}")
    #         print("Loading weights from old checkpoint format...")
    #         # Load old checkpoint data and manually transfer to the model
    #         try:
    #             checkpoint = torch.load(old_checkpoint_path, map_location="cpu")
    #             model.student.load_state_dict(checkpoint['student'])
    #             model.teacher.load_state_dict(checkpoint['teacher'])
    #             model.dino_loss.load_state_dict(checkpoint['dino_loss'])
    #             print(f"Successfully loaded weights from {old_checkpoint_path}")
    #         except Exception as e:
    #             print(f"Failed to load old checkpoint: {e}")
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        # Use ckpt_path for PyTorch Lightning 2.4.0+
        ckpt_path=checkpoint_path,
    )
    
    # Train model
    start_time = time.time()
    trainer.fit(model, data_module)
    
    # Print training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main() 