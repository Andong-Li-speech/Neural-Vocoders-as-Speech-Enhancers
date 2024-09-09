"""
Copyright (c) 2023, JaeBinCHA7
All rights reserved.

This source code is created based on the implementation of ideas presented in the paper:
Kim, Doyeon, et al. "HD-DEMUCS: General Speech Restoration with Heterogeneous Decoders." arXiv preprint arXiv:2306.01411 (2023).
Available at: https://arxiv.org/abs/2306.01411

This source code is licensed under the MIT license found in the
LICENSE file at https://github.com/JaeBinCHA7?tab=repositories (if applicable).
"""

import math
import functools
import torch as th
import torch
from torch import nn
from torch.nn import functional as F
from dataset import inverse_mel
from torchaudio import transforms

def sinc(t):
    """sinc.
    :param t: the input tensor
    """
    return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)


def kernel_upsample2(zeros=56):
    """kernel_upsample2.
    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape # [32, 1, 32085]
    kernel = kernel_upsample2(zeros).to(x) # [1, 1, 112]
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time) # [32, 1, 32085]
    y = th.stack([x, out], dim=-1) # [32, 1, 32085, 2]
    return y.view(*other, -1)


def kernel_downsample2(zeros=56):
    """kernel_downsample2.
    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)



def capture_init(init):
    """capture_init.
    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class HDDemucas(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.
    """

    @capture_init
    def __init__(self, h):

        super().__init__()
        self.h = h
        self.chin = h.chin  # 1
        chin = self.chin
        self.chout = h.chout  # 1
        chout = self.chout
        self.hidden = h.hidden  # 48
        hidden = self.hidden
        self.depth = h.depth  # 5
        self.kernel_size = h.kernel_size  # 8
        self.stride = h.stride  # 4 
        self.causal = h.causal  # False
        self.resample = h.resample  # 4
        self.growth = h.growth  # 2
        self.max_hidden = h.max_hidden  # 10_000
        self.normalize = h.normalize  # False
        self.glu = h.glu  # True
        self.rescale = h.rescale  # 0.1
        self.floor = h.floor  # 1e-3
        self.sampling_rate = h.sampling_rate  # 
        self.n_fft = h.n_fft
        self.hop_size = h.hop_size
        self.win_size = h.win_size
        self.init_phase = h.init_phase
        self.num_mels = h.num_mels

        if self.init_phase.lower() == 'griffin_lim':
            self.transform = transforms.GriffinLim(n_fft=self.n_fft, n_iter=32, win_length=self.win_size, hop_length=self.hop_size, power=1)

        if self.resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.encoder = nn.ModuleList()
        self.decoder_map = nn.ModuleList()
        self.decoder_mask = nn.ModuleList()
        activation = nn.GLU(1) if self.glu else nn.ReLU()
        ch_scale = 2 if self.glu else 1
        dilation_factor = [1, 3, 5, 7, 9]
        for index in range(self.depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, self.kernel_size, self.stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            # Suppression block
            decode_mask = []
            decode_mask += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, self.kernel_size, self.stride),
            ]
            if index > 0:
                # decode_mask.append(nn.ReLU()) # Original DEMUCS
                decode_mask.append(nn.Sigmoid())  # HD-DEMUCS
            self.decoder_mask.insert(0, nn.Sequential(*decode_mask))

            # refinement block
            decode_map = []
            decode_map += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                # nn.ConvTranspose1d(hidden, chout, kernel_size, stride),  # Original DEMUCS
                nn.ConvTranspose1d(hidden, chout, self.kernel_size, self.stride, dilation=dilation_factor[index],
                                   padding=7 * index)  # HD-DEMUCS
            ]
            if index > 0:
                decode_map.append(nn.ReLU())
            self.decoder_map.insert(0, nn.Sequential(*decode_map))

            chout = hidden
            chin = hidden
            hidden = min(int(self.growth * hidden), self.max_hidden)

        self.lstm = BLSTM(chin, bi=not self.causal)
        if self.rescale:
            rescale_module(self, reference=self.rescale)

        # Fusion block
        self.fb_conv1 = nn.Sequential()
        self.fb_conv1.append(nn.Conv1d(2, 2, 3, 1, padding=1))
        self.fb_conv1.append(nn.LeakyReLU())

        self.fb_conv2 = nn.Sequential()
        self.fb_conv2.append(nn.Conv1d(2, 2, 3, 1, padding=1))
        self.fb_conv2.append(nn.LeakyReLU())

        self.fb_conv3 = nn.Sequential()
        self.fb_conv3.append(nn.Conv1d(2, 2, 3, 1, padding=1))
        self.fb_conv3.append(nn.Sigmoid())

        self.weight = nn.Parameter(th.tensor(0.5, requires_grad=True))

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mel, inv_mel=None):
        """
        input: (B, Fm, T)
        return: 
            logamp: (B, F, T)
            pha: (B, F, T)
            rea: (B, F, T)
            imag: (B, F, T)
            audio: (B, 1, L)
        """
        if inv_mel is None:
            inv_amp = (
                inverse_mel(
                    mel,
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.h.fmin,
                    self.h.fmax
                )
                .abs().clamp_min(1e-5)
            )
        else:
            inv_amp = inv_mel
        # first transform to waveform with IFTFT
        if self.init_phase.lower() in ['zero', 'rand']:
            if self.init_phase.lower() == 'zero':
                inv_phase = torch.zeros([*inv_amp.shape], device=inv_amp.device)
            elif self.init_phase.lower() == 'rand':
                inv_phase = 2 * math.pi * torch.rand_like(inv_amp) - math.pi  # [-pi, pi)
            inv_comp = torch.complex(inv_amp * torch.cos(inv_phase), inv_amp * torch.sin(inv_phase))
            inv_wav = torch.istft(inv_comp,
                                n_fft=self.n_fft,
                                hop_length=self.hop_size,
                                win_length=self.win_size,
                                window=torch.hann_window(self.win_size).to(mel.device),
                                )  # (B, L)
        elif self.init_phase.lower() == 'griffin_lim':
            inv_wav = self.transform(inv_amp)

        mix = inv_wav.unsqueeze(1)  # (B, 1, L)

        if self.normalize:
            mean = mix.mean(dim=(1, 2), keepdim=True)
            std = mix.std(dim=(1, 2), keepdim=True)
            mix = (mix - mean) / (1e-5 + std)
        else:
            mean, std = 0., 1. 
        length = mix.shape[-1]
        
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        x_us = x
        skips_mask = []
        skips_map = []
        for encode in self.encoder:
            x = encode(x)
            skips_mask.append(x)

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)

        x_mask = x
        for decode in self.decoder_mask:
            skip = skips_mask.pop(-1)
            x_mask = x_mask + skip
            x_mask = decode(x_mask)
            skips_map.append(x_mask)

        x_map = x
        for decode in self.decoder_map:
            x_map = decode(x_map)
            skip = skips_map.pop(0)
            x_map = x_map + skip

        d_s = x_mask * x_us
        d_r = x_map

        x_fb = th.concat((d_s, d_r), dim=1)

        x_fb = self.fb_conv1(x_fb)
        x_fb = self.fb_conv2(x_fb)
        x_fb = self.fb_conv3(x_fb)

        out = d_s * (1 - self.weight) * x_fb[:, :1, ...] + d_r * self.weight * x_fb[:, 1:, ...]

        if self.resample == 2:
            out = downsample2(out)

        elif self.resample == 4:
            out = downsample2(out)
            out = downsample2(out)

        out_wav = out[..., :length] * std + mean  # (B, L)

        return out_wav.squeeze(1)