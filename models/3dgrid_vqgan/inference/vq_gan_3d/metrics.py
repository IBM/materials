import torch
import torch.nn.functional as F

import os
import sys
import math
import lpips as lpips_metric
import piq
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def normalize_tensor(outmap):
    flattened_outmap = outmap.view(outmap.shape[0], -1, 1, 1) # Use 1's to preserve the number of dimensions for broadcasting later, as explained
    outmap_min, _ = torch.min(flattened_outmap, dim=1, keepdim=True)
    outmap_max, _ = torch.max(flattened_outmap, dim=1, keepdim=True)
    outmap = (outmap - outmap_min) / (outmap_max - outmap_min)
    return outmap


class ImageMetrics:

    def __init__(self, grid_true, grid_pred, device='cpu'):
        self.grid_true = grid_true.to(device)  # [N, H, W]
        self.grid_pred = grid_pred.to(device) # [N, H, W]
        self.num_sequence = grid_true.shape[0]
        self.loss_fn_vgg = lpips_metric.LPIPS(net='vgg', verbose=False).to(device) # closer to "traditional" perceptual loss, when used for optimization
        self.device = device

    def ssim(self):
        """Structured Similarity Index Metric"""
        a = normalize_tensor(self.grid_true.unsqueeze(0))
        b = normalize_tensor(self.grid_pred.unsqueeze(0))
        ssim = piq.ssim(a, b, data_range=1., reduction='none').squeeze().item()
        return ssim

    def mssim(self):
        """Mean Structured Similarity Index Metric"""
        mssim = 0
        for idx in range(self.num_sequence):
            max_value = max([self.grid_true[idx].max(), self.grid_pred[idx].max()])
            min_value = min([self.grid_true[idx].min(), self.grid_pred[idx].min()])
            data_range = abs(max_value - min_value)

            a = self.grid_true[idx].detach().cpu().numpy()
            b = self.grid_pred[idx].detach().cpu().numpy()

            mssim += ssim(a, b, data_range=data_range.item())
        return mssim / self.num_sequence
    
    def multiscale_ssim(self):
        """Multi-Scale SSIM"""
        a = normalize_tensor(self.grid_true.unsqueeze(0))
        b = normalize_tensor(self.grid_pred.unsqueeze(0))
        ms_ssim_index = piq.multi_scale_ssim(a, b, data_range=1., kernel_size=7).item()
        return ms_ssim_index

    def psnr(self):
        """Peak Signal-to-Noise Ratio"""
        psnr = 0
        for idx in range(self.num_sequence):
            max_value = max([self.grid_true[idx].max(), self.grid_pred[idx].max()])
            min_value = min([self.grid_true[idx].min(), self.grid_pred[idx].min()])
            data_range = abs(max_value - min_value)

            a = self.grid_true[idx].detach().cpu().numpy()
            b = self.grid_pred[idx].detach().cpu().numpy()

            psnr += peak_signal_noise_ratio(a, b, data_range=data_range.item())
        return psnr / self.num_sequence

    def _calculate_fid(self, img1, img2):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()

        # calculate mean and covariance statistics
        mu1, sigma1 = img1.mean(axis=0), cov(img1, rowvar=False)
        mu2, sigma2 = img2.mean(axis=0), cov(img2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def fid(self):
        """Frechet Inception Distance"""
        fid = 0
        for idx in range(self.num_sequence):
            fid += self._calculate_fid(self.grid_true[idx], self.grid_pred[idx])
        return fid / self.num_sequence

    def _calculate_lpips(self, img1, img2):
        img1 = (2 * img1 / img1.max() - 1)  # normalize between -1 to 1
        img2 = (2 * img2 / img2.max() - 1)  # normalize between -1 to 1
        perceptual_loss = self.loss_fn_vgg(img1, img2).squeeze()
        return perceptual_loss.item()

    def lpips(self):
        """Learned Perceptual Image Patch Similarity"""
        perceptual_loss = 0
        for idx in range(self.num_sequence):
            perceptual_loss += self._calculate_lpips(self.grid_true[idx], self.grid_pred[idx])
        return perceptual_loss / self.num_sequence

    def reconstruction(self):
        return F.l1_loss(self.grid_true, self.grid_pred).item()

    def IS(self):
        """Inception Score"""
        is_score = 0
        for idx in range(self.num_sequence):
            is_score += piq.IS(distance='l1')(self.grid_true[idx], self.grid_pred[idx])
        return (is_score / self.num_sequence).item()

    def kid(self):
        """Kernel Inception Distance"""
        kid_score = 0
        for idx in range(self.num_sequence):
            kid_score += piq.KID()(self.grid_true[idx], self.grid_pred[idx])
        return (kid_score / self.num_sequence).item()

    def get_metrics(self):
        blockPrint()
        metrics = dict(
            SSIM=self.ssim(),
            MSSIM=self.mssim(),
            MS_SSIM=self.multiscale_ssim(),
            PSNR=self.psnr(),
            IS=self.IS(),
            FID=self.fid(),
            KID=self.kid(),
            LPIPS=self.lpips(),
            Reconstruction=self.reconstruction(),
        )
        enablePrint()
        return metrics