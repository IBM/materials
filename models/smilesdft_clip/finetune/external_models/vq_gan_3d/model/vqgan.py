"""Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import numpy as np
import pickle as pkl
import random
import gc
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from external_models.vq_gan_3d.utils import shift_dim, adopt_weight, comp_getattr
from external_models.vq_gan_3d.model.lpips import LPIPS
from external_models.vq_gan_3d.model.codebook import Codebook


def silu(x):
    return x*torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class MeanPooling(nn.Module):
    def __init__(self, kernel_size=16):
        super(MeanPooling, self).__init__()
        # Define a 3D average pooling layer
        self.pool = nn.AvgPool3d(kernel_size=kernel_size)

    def forward(self, x):
        # Apply average pooling
        x = self.pool(x)
        # Flatten the tensor to a single dimension per batch element
        x = x.view(x.size(0), -1)
        return x


class VQGAN(nn.Module):
    def __init__(self):
        super().__init__()

        self._set_seed(0)
        self.embedding_dim = 256
        self.n_codes = 16384

        self.encoder = Encoder(16, [4,4,4], 1, 'group', 'replicate', 32)
        self.decoder = Decoder(16, [4,4,4], 1, 'group', 32)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(self.enc_out_ch, 256, 1, padding_type='replicate')
        self.post_vq_conv = SamePadConv3d(256, self.enc_out_ch, 1)

        self.codebook = Codebook(16384, 256, no_random_restart=False, restart_thres=False)

        self.pooling = MeanPooling(kernel_size=16)

        self.gan_feat_weight = 4
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(1, 64, 3, norm_layer=nn.BatchNorm2d)

        self.disc_loss = hinge_d_loss
        self.perceptual_model = LPIPS()
        self.image_gan_weight = 1
        self.perceptual_weight = 4
        self.l1_weight = 4

    def encode(self, x, include_embeddings=False, quantize=True):
        with torch.enable_grad():
            h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def feature_extraction(self, x):
        """Extract embeddings given a grid."""
        # encondings = self.encode(x, include_embeddings=False, quantize=True)
        # h = F.embedding(encondings, self.codebook.embeddings)
        # with torch.enable_grad():
        #     h = self.pooling(h)
        # return h
        h = self.encode(x, include_embeddings=False, quantize=False)
        with torch.enable_grad():
            h = self.pooling(h.permute(0, 2, 3, 4, 1))
        return h

    def forward(self, global_step, x, optimizer_idx=None, log_image=False, gpu_id=0):
        B, C, T, H, W = x.shape

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z, gpu_id)

        #vq_output['embeddings'] = torch.exp(vq_output['embeddings']) 
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        recon_loss = (F.l1_loss(x_recon, x) * self.l1_weight)

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, T, [B]).to(gpu_id)
        frame_idx_selected = frame_idx.reshape(-1,
                                               1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            return frames, frames_recon, x, x_recon

        if optimizer_idx == 0:
            # Autoencoder - train the "generator"

            # Perceptual loss
            perceptual_loss = 0
            if self.perceptual_weight > 0:
                perceptual_loss = self.perceptual_model(
                    frames, frames_recon).mean() * self.perceptual_weight
                # perceptual_loss = .123

            # Discriminator loss (turned on after a certain epoch)
            logits_image_fake, pred_image_fake = self.image_discriminator(
                frames_recon)
            g_image_loss = -torch.mean(logits_image_fake)
            g_loss = self.image_gan_weight*g_image_loss 
            disc_factor = adopt_weight(
                global_step, threshold=self.cfg.model.discriminator_iter_start)
            aeloss = disc_factor * g_loss

            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            image_gan_feat_loss = 0
            feat_weights = 4.0 / (3 + 1)
            if self.image_gan_weight > 0:
                logits_image_real, pred_image_real = self.image_discriminator(
                    frames)
                for i in range(len(pred_image_fake)-1):
                    image_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_image_fake[i], pred_image_real[i].detach(
                        )) * (self.image_gan_weight > 0)

            gan_feat_loss = disc_factor * self.gan_feat_weight * \
                (image_gan_feat_loss)

            return recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss, (g_image_loss, image_gan_feat_loss, vq_output['commitment_loss'], vq_output['perplexity'])

        if optimizer_idx == 1:
            # Train discriminator
            logits_image_real, _ = self.image_discriminator(frames.detach())

            logits_image_fake, _ = self.image_discriminator(
                frames_recon.detach())

            d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
            disc_factor = adopt_weight(
                global_step, threshold=self.cfg.model.discriminator_iter_start)
            discloss = disc_factor * \
                (self.image_gan_weight*d_image_loss)

            return discloss

        perceptual_loss = self.perceptual_model(
            frames, frames_recon) * self.perceptual_weight
        return recon_loss, x_recon, vq_output, perceptual_loss

    def load_checkpoint(self, ckpt_path):
        # load checkpoint file
        ckpt_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # load hyparameters
        self.config = ckpt_dict['hparams']['_content']
        self.embedding_dim = self.config['model']['embedding_dim']
        self.n_codes = self.config['model']['n_codes']

        print(self.config['model'])
        # instantiate modules
        self.encoder = Encoder(
            self.config['model']['n_hiddens'],
            self.config['model']['downsample'],
            self.config['dataset']['image_channels'], 
            self.config['model']['norm_type'], 
            self.config['model']['padding_type'],
            self.config['model']['num_groups'],
        )
        self.decoder = Decoder(
            self.config['model']['n_hiddens'],
            self.config['model']['downsample'], 
            self.config['dataset']['image_channels'], 
            self.config['model']['norm_type'], 
            self.config['model']['num_groups']
        )
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(self.enc_out_ch, self.embedding_dim, 1, padding_type=self.config['model']['padding_type'])
        self.post_vq_conv = SamePadConv3d(self.embedding_dim, self.enc_out_ch, 1)
        self.codebook = Codebook(
            self.n_codes, 
            self.embedding_dim,
            no_random_restart=self.config['model']['no_random_restart'], 
            restart_thres=False
        )
        self.gan_feat_weight = self.config['model']['gan_feat_weight']
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            self.config['dataset']['image_channels'], 
            self.config['model']['disc_channels'],
            self.config['model']['disc_layers'], 
            norm_layer=nn.BatchNorm2d
        )
        self.disc_loss = hinge_d_loss
        self.perceptual_model = LPIPS()
        self.image_gan_weight = self.config['model']['gan_feat_weight']
        self.perceptual_weight = self.config['model']['perceptual_weight'] 
        self.l1_weight = self.config['model']['l1_weight']

        # restore model weights
        self.load_state_dict(ckpt_dict["MODEL_STATE"], strict=True)

        # load RNG states each time the model and states are loaded from checkpoint
        if 'rng' in self.config:
            rng = self.config['rng']
            for key, value in rng.items():
                if key =='torch_state':
                    torch.set_rng_state(value.cpu())
                elif key =='cuda_state':
                    torch.cuda.set_rng_state(value.cpu())
                elif key =='numpy_state':
                    np.random.set_state(value)
                elif key =='python_state':
                    random.setstate(value)
                else:
                    print('unrecognized state')

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch['data']
        x = x.to(self.device)
        frames, frames_rec, _, _ = self(x, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        #log['mean_org'] = batch['mean_org']
        #log['std_org'] = batch['std_org']
        return log
    
    def _set_seed(self, value):
        print('Random Seed:', value)
        random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed(value)
        torch.cuda.manual_seed_all(value)
        np.random.seed(value)
        cudnn.deterministic = True
        cudnn.benchmark = True
        cudnn.enabled = True


def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, n_hiddens = 16, downsample = [2,2,2] , image_channel=64, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(
                in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens = 16, upsample= [4,4,4], image_channel=1, norm_type='group', num_groups=1):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        print('n_times_upsample :', n_times_upsample)
        max_us = n_times_upsample.max()
        print('max_us :', max_us)


        in_channels = n_hiddens*2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(
                in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(
            out_channels, image_channel, kernel_size=3)
        

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


def load_VQGAN(folder="../data/checkpoints/pretrained", ckpt_filename="VQGAN_43.pt"):
    model = VQGAN()
    model.load_checkpoint(os.path.join(folder, ckpt_filename))
    model.eval()
    return model