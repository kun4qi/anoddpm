import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
import collections
import copy
import sys
import time
from random import seed

from utils import load_json
from utils import check_manual_seed
from utils import Logger
from utils import ModelSaver
from utils import Time
from dataio import CKBrainMetDataModule

from simplex import Simplex_CLASS
from GaussianDiffusion import GaussianDiffusionModel, get_beta_schedule, generate_simplex_noise, random_noise
from UNet import UNetModel, update_ema_params

    

class anoddpm(LightningModule):
    def __init__(self, config, needs_save):
        super().__init__()
        self.config = config
        self.needs_save = needs_save
        self.output_dir = config.save.output_root_dir
        self.batch_size = self.config.dataset.batch_size
        self.automatic_optimization = False  # For advanced/expert users who want to do esoteric optimization schedules or techniques, use manual optimization.
        self.train_lr = 0.0001
        if self.config.training.noise == "gauss":
            self.noise_fn = lambda x, t: torch.randn_like(x)
        else:
            self.simplex = Simplex_CLASS()
            if self.config.training.noise == "simplex_randParam":
                self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, True, in_channels=self.config.model.input_dim)
            elif self.config.training.noise == "random":
                self.noise_fn = lambda x, t: random_noise(self.simplex, x, t)
            elif self.config.training.noise == "simplex":
                self.noise_fn = lambda x, t: generate_simplex_noise(self.simplex, x, t, False, in_channels=self.config.model.input_dim)

        # networks
        self.model = UNetModel(
            self.config.dataset.image_size, self.config.training.base_channels, self.config.training.channel_mults, 
            dropout=self.config.training.dropout, n_heads=self.config.training.num_heads, n_head_channels=self.config.training.num_head_channels, in_channels=self.config.model.input_dim)
        
        self.betas = get_beta_schedule(self.config.training.timesteps, self.config.training.beta_schedule)

        self.diffusion = GaussianDiffusionModel(
            self.config.dataset.image_size, self.betas, loss_weight=self.config.training.loss_weight,
            loss_type=self.config.training.loss_type, noise=self.config.training.noise, img_channels=self.config.model.input_dim
            )

    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            if self.needs_save:
                #save recon image
                if self.current_epoch == 1 or (self.current_epoch - 1) % self.config.save.save_image_epoch_interval == 0:
                    image = batch['image']

                    output = self.diffusion.forward_backward(
                        self.model, image,
                        see_whole_sequence="half",
                        t_distance=self.config.training.t_distance, denoise_fn=self.config.training.noise
                        )
                    
                    original_image, noise, x_t, x_recon = output[0], output[1], output[2], output[-1]

                    original_image = original_image[:self.config.save.n_save_images, ...]
                    noise = noise[:self.config.save.n_save_images, ...]
                    x_t = x_t[:self.config.save.n_save_images, ...]
                    x_recon = x_recon[:self.config.save.n_save_images, ...]
                    
                    if self.config.save.n_save_images > self.batch_size:
                        raise ValueError(f'can not save images properly')
                    
                    self.logger.train_log_images(torch.cat([original_image, noise, x_t, x_recon]), self.current_epoch-1, self.config.save.n_save_images)

                    
        if self.needs_save:
            self.log('epoch', self.current_epoch, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            if self.global_step % 2 ==0:
                self.log('iteration', self.global_step/2, on_step=True, on_epoch=False, logger=True)

        m_optim = self.optimizers()
        m_optim.zero_grad()

        #difusion training
        image = batch['image']
        
        loss, _ = self.diffusion.p_loss(self.model, image, self.config)

        self.manual_backward(loss)
        m_optim.step()

        if self.needs_save:
            self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            
        return {'loss': loss}
        

    def validation_step(self, batch, batch_idx):

        if self.config.training.val_mode == "train":
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()


        #save recon image
        if batch_idx == 0:
            if self.needs_save:
                if self.current_epoch % self.config.save.save_image_epoch_interval == 0:
                    image = batch['image']

                    output = self.diffusion.forward_backward(
                        self.model, image,
                        see_whole_sequence="half",
                        t_distance=self.config.training.t_distance, denoise_fn=self.config.training.noise
                        )
                    
                    original_image, noise, x_t, x_recon = output[0], output[1], output[2], output[-1]

                    original_image = original_image[:self.config.save.n_save_images, ...]
                    noise = noise[:self.config.save.n_save_images, ...]
                    x_t = x_t[:self.config.save.n_save_images, ...]
                    x_recon = x_recon[:self.config.save.n_save_images, ...]

                    if self.config.save.n_save_images > self.batch_size:
                        raise ValueError(f'can not save images properly')
                    self.logger.val_log_images(torch.cat([original_image, noise, x_t, x_recon]), self.current_epoch, self.config.save.n_save_images)

        #difusion training
        image = batch['image']

        loss, _ = self.diffusion.p_loss(self.model, image, self.config)

        if self.needs_save:
            metrics = {
            'epoch': self.current_epoch,
            'Val_loss': loss.item(),
            }
            self.logger.log_val_metrics(metrics)
                
        return {'Val_loss': loss}


    def configure_optimizers(self):
        m_optim = optim.AdamW(self.model.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay, betas=(0.9, 0.999))
        return [m_optim]


def main(config, needs_save):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.visible_devices
    seed = check_manual_seed(config.training.seed)
    print('Using manual seed: {}'.format(seed))

    #set logger
    monitoring_metrics = ['epoch', 'iteration', 'loss', 'Val_loss']
  
    logger = Logger(save_dir=config.save.output_root_dir,
                    config=config,
                    seed=config.training.seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics
                    )
    save_dir_path = logger.log_dir
    os.makedirs(save_dir_path, exist_ok=True)
    
    #save config
    logger.log_hyperparams(config, needs_save)

    #set callbacks
    checkpoint_callback = ModelSaver(
        limit_num=config.save.n_saved,
        save_interval=config.save.save_epoch_interval,
        monitor=None,
        dirpath=logger.log_dir,
        filename='ckpt-{epoch:04d}',
        save_top_k=-1,
        save_last=False
    )

    #time per epoch
    timer = Time(config)

    dm = CKBrainMetDataModule(config)

    trainer = Trainer(
        default_root_dir=config.save.output_root_dir,
        accelerator="gpu",
        devices=config.training.visible_devices,
        strategy=DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=config.training.sync_batchnorm,
        max_epochs=config.training.n_epochs,
        callbacks=[checkpoint_callback, timer],
        logger=logger,
        deterministic=False,
        log_every_n_steps=1,
        num_sanity_val_steps = 0
        )
    
    dm.prepare_data()
    dm.setup(stage="fit")
    print('Training starts: [max_epochs] {}, [max_iterations] {}'.format(
        config.training.n_epochs, config.training.n_epochs * len(dm.train_dataloader()))
    )

    if not config.model.saved:
      model = anoddpm(config, needs_save, *dm.size())
      trainer.fit(model, dm)

    else:
      print(f'model load from {config.save.load_model_dir + config.save.model_savename}')
      model = anoddpm.load_from_checkpoint(config.save.load_model_dir + config.save.model_savename, config=config, needs_save=needs_save,)
      trainer.fit(model, dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-s', '--save', help='save logs and models', action='store_true')
    args = parser.parse_args()

    config = load_json(args.config)

    main(config, args.save)
