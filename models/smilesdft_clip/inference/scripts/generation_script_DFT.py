# @title Load libraries and variables

import argparse
import os
import random
import math
import sys
sys.path.append('./')

from PIL import Image
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

import kornia.augmentation as K
import numpy as np
import imageio
import cv2
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

from external_models.vq_gan_3d.utils import shift_dim
from external_models.vq_gan_3d import load_VQGAN
from contrastive_model.load import load_clip, load_siglip

import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt


def plot_voxel_grid(tensor, filename='voxel_plot.png', thresholds=[0.5, 0.25, 0.125, 0.0125], title='Voxel Grid Plot'):
    """
    Plots a 3D voxel grid from a tensor and saves the plot to a file.

    Args:
        tensor (torch.Tensor): The input tensor with shape [1, 1, D, H, W].
        filename (str): The name of the file to save the plot.
        thresholds (list of float): Thresholds for voxel visibility to use different colors.
        title (str): Title of the plot.
    """
    # Convert tensor to NumPy array and remove singleton dimensions
    data_np = tensor.detach().squeeze().cpu().numpy()

    # Create voxel grid
    x, y, z = np.indices(np.array(data_np.shape) + 1) / (max(data_np.shape) + 1)

    colors = [0]*5
    alpha = 0.3

    colors[0] = [1, 0, 0, alpha]
    colors[1] = [.75, 0, .25, alpha]
    colors[2] = [.50, 0, .50, alpha]
    colors[3] = [.25, 0, .75, alpha]
    colors[4] = [0, 0, 1, alpha]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the voxels for each threshold
    for i, threshold in enumerate(tqdm(thresholds)):
        ax.voxels(x, y, z, np.clip(data_np - threshold, 0, 1),
                  linewidth=0.5, facecolors=colors[i % len(colors)], alpha=alpha)

    ax.set(xlabel='', ylabel='', zlabel='')
    ax.set_aspect('equal')
    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)

    # Add a title to the plot
    ax.set_title(title, fontsize=15)

    # Remove grid lines
    ax.grid(False)

    plt.tight_layout()

    # Save the plot
    plt.savefig(filename)

    # Show the plot
    plt.show()


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w, s = input.shape
    dh, dw, ds = size

    input = input.view([n * c, 1, h, w, s])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :, None])

    if ds < s:
        kernel_s = lanczos(ramp(ds / s, 2), 2).to(input.device, input.dtype)
        pad_s = (kernel_s.shape[0] - 1) // 2
        input = F.pad(input, (pad_s, pad_s, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_s[None, None, None, None, :])

    input = input.view([n, c, h, w, s])
    return F.interpolate(input, size, mode='trilinear', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.augs = nn.Sequential(
            # K.RandomHorizontalFlip(p=0.5),
            # K.RandomVerticalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            # K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5),
            K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5),
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),

)
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool3d((self.cut_size, self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool3d((self.cut_size, self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX, sideZ = input.shape[2:5]
        max_size = min(sideX, sideY, sideZ)
        min_size = min(sideX, sideY, sideZ, self.cut_size)
        cutouts = []

        for _ in range(self.cutn):

            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            # offsetx = torch.randint(0, sideX - size + 1, ())
            # offsety = torch.randint(0, sideY - size + 1, ())
            # offsetz = torch.randint(0, sideZ - size + 1, ())
            # cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size, offsetz:offsetz + size]
            # cutouts.append(resample(cutout, (self.cut_size, self.cut_size, self.cut_size)))

            # cutout = transforms.Resize(size=(self.cut_size, self.cut_size, self.cut_size))(input)

            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)

        seq = 128
        cutouts = torch.cat(cutouts, dim=0)
        batch = [self.augs(cutouts[:, :, :, :, idx]) for idx in range(seq)]
        batch = torch.cat(batch, dim=-1).view(-1, 1, 128, 128, 128)

        del cutouts
        torch.cuda.empty_cache()

        # batch = self.augs(torch.cat(cutouts, dim=0))

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch # (32, 1, 128, 128, 128)


# def resize_image(image, out_size):
#     ratio = image.size[0] / image.size[1]
#     area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
#     size = round((area * ratio)**0.5), round((area / ratio)**0.5)
#     return image.resize(size, Image.LANCZOS)


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


def synth(z, model):
    z_q = vector_quantize(z.movedim(1, 3), model.codebook.embeddings).movedim(3, 1)
    # return clamp_with_grad(model.decoder(model.post_vq_conv(shift_dim(z_q, -1, 1))).add(1).div(2), 0, 1)  # -> (1, 1, 128, 128, 128)
    # return clamp_with_grad(model.decoder(model.post_vq_conv(shift_dim(z_q, -1, 1))), 0, 1)  # -> (1, 1, 128, 128, 128)
    return model.decoder(model.post_vq_conv(shift_dim(z_q, -1, 1)))  # -> (1, 1, 128, 128, 128)


# @torch.no_grad()
# def checkin(i, losses, z, args, model):
#     losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
#     tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
#     out = synth(z, args, model)
#     TF.to_pil_image(out[0].cpu()).save('progress.png')
#     display.display(display.Image('progress.png'))


import subprocess
# subprocess.run(["nvidia-smi"]) 
def ascend_txt(i, args, model, perceptor, normalize, make_cutouts, z, z_orig, pMs):
    out = synth(z, model)
    cts = make_cutouts(out)
    # cts = normalize(cts)

    image_features = perceptor.image_encoder(cts)
    iii = perceptor.image_projection(image_features.float())

    # iii = perceptor.encode_image(normalize(out)).float()
    # out.shape -> (1, 3, 224, 160)
    # normalize(make_cutouts(out)).shape -> (32, 3, 224, 224)
    # iii.shape -> (32, 512)

    result = []

    if args.init_weight:  # almost false
        # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
        result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1/torch.tensor(i*2 + 1))*args.init_weight) / 2)
    
    for prompt in pMs:
        result.append(prompt(iii))

    if not os.path.exists('./steps'):
        os.mkdir('./steps')

    plt.imshow(out.squeeze()[-1].detach().cpu().numpy(), cmap='inferno')
    plt.axis('off')
    plt.savefig('./steps/' + str(i) + '.png', dpi=70, bbox_inches='tight', pad_inches=0)

    # img = np.array(out.mul(255).clamp(0, 255)[0].squeeze()[-1].unsqueeze(0).cpu().detach().numpy().astype(np.uint8))[:,:,:]
    # img = np.transpose(img, (1, 2, 0))
    # cv2.imwrite('./steps/' + str(i) + '.png', np.array(img))

    return result, out


def train(i, args, model, perceptor, normalize, make_cutouts, z, z_orig, pMs, opt, z_min, z_max):
    opt.zero_grad()
    lossAll, out = ascend_txt(i, args, model, perceptor, normalize, make_cutouts, z, z_orig, pMs)
    # if i % args.display_freq == 0:
        # checkin(i, lossAll, z, args, model)

    loss = sum(lossAll)
    loss.backward()
    opt.step()
    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))

    return loss.item(), out


def main():
    #@title Parameters
    texts = "C#CC(C)C1CC1CC"
    width =  128
    height = 128
    sequence = 128
    model = "vqgan_imagenet_f16_16384"
    images_interval = 50
    init_image = ""
    target_images = ""
    seed = 99
    max_iterations = 600
    stepsize = 0.5

    if seed == -1:
        seed = None
    if init_image == "None":
        init_image = None
    if target_images == "None" or not target_images:
        target_images = []
    else:
        target_images = target_images.split("|")
        target_images = [image.strip() for image in target_images]

    texts = [phrase.strip() for phrase in texts.split("|")]
    if texts == ['']:
        texts = []


    args = argparse.Namespace(
        prompts=texts,
        image_prompts=target_images,
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size=[width, height, sequence],
        init_image=init_image,
        init_weight=0.,
        clip_model='ViT-B/32',
        vqgan_config=f'{model}.yaml',
        vqgan_checkpoint=f'{model}.ckpt',
        step_size=stepsize,
        cutn=4,
        cut_pow=1.,
        display_freq=images_interval,
        seed=seed,
    )


    #@title Actually do the run...
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if texts:
        print('Using texts:', texts)
    if target_images:
        print('Using image prompts:', target_images)
    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed
    torch.manual_seed(seed)
    print('Using seed:', seed)

    # load models
    model = load_VQGAN(folder='./data/checkpoints/vqgan', ckpt_filename='VQGAN_43.pt')
    perceptor = load_clip(folder='./data/checkpoints/clip', ckpt_filename='CLIP_8_20250127-024339.pt')
    # perceptor = load_siglip(folder='./data/checkpoints/siglip', ckpt_filename='SigLIP_8_20250127-024423.pt')

    # disable model's gradients
    model.eval().requires_grad_(False).to(device)
    perceptor.eval().requires_grad_(False).to(device)

    # cut_size = perceptor.visual.input_resolution  # -> 224
    cut_size = 128
    # f = 2**(model.decoder.num_resolutions - 1)  # model.decoder.num_resolutions -> 5
    f = 2**(3 - 1)  # model.decoder.num_resolutions -> 5
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    toksX, toksY, toksZ = args.size[0] // f, args.size[1] // f, args.size[2] // f  # -> 32, 32, 32
    sideX, sideY, sideZ = toksX * f, toksY * f, toksZ * f  # not used

    e_dim = model.embedding_dim
    n_toks = model.n_codes
    z_min = model.codebook.embeddings.min(dim=0).values[None, None, None, None, :]  # -> (1, 256, 1, 1)
    z_max = model.codebook.embeddings.max(dim=0).values[None, None, None, None, :]  # -> (1, 256, 1, 1)

    if args.init_image:
        img = Image.open(args.init_image)
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY, sideZ), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX * toksZ], device=device), n_toks).float()
        z = one_hot @ model.codebook.embeddings
        z = z.view([-1, toksY, toksX, toksZ, e_dim])
        z = torch.rand_like(z)*2
        # 3dgrid-vqgan z.shape -> (1, 32, 32, 32, 256)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args.step_size)

    # normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
    #                                 std=[0.26862954, 0.26130258, 0.27577711])
    normalize = transforms.Normalize(mean=[0.48145466],
                                    std=[0.26862954])


    pMs = []

    for prompt in args.prompts:
        txt, weight, stop = parse_prompt(prompt)
        # embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        embed = perceptor.text_encoder(txt).to(device)
        embed = perceptor.text_projection(embed).float().to(device)
        pMs.append(Prompt(embed, weight, stop).to(device))


    i = 0
    try:
        with tqdm(total=max_iterations) as pbar:
            while True:
                loss, out = train(i, args, model, perceptor, normalize, make_cutouts, z, z_orig, pMs, opt, z_min, z_max)
                if i == max_iterations:
                    np.save('out.npy', out.detach().cpu().numpy())

                    init_frame = 1 #This is the frame where the video will start
                    last_frame = max_iterations #You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.

                    min_fps = 10
                    max_fps = 60

                    total_frames = last_frame-init_frame

                    length = 2 #Desired time of the video in seconds

                    frames = []
                    tqdm.write('Generating video...')
                    for i in range(init_frame,last_frame+1): #
                        frames.append(Image.open("./steps/"+ str(i) +'.png'))

                    fps = np.clip(total_frames/length,min_fps,max_fps)

                    from subprocess import Popen, PIPE
                    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', 'video.mp4'], stdin=PIPE)
                    for im in tqdm(frames):
                        im.save(p.stdin, 'PNG')
                    p.stdin.close()
                    p.wait()

                    plot_voxel_grid(out.half(), filename=f'generated_voxel_plot_C#CC(C)C1CC1CC_{seed}.png', title='Prompt generated 3D energy grid - C#CCC1CC1(C)CO')
                    break
                i += 1
                pbar.update()
                pbar.set_description('[OPTIMIZING]')
                pbar.set_postfix(loss=loss)
                pbar.refresh()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()