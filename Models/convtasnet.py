import torch
import math
import torch.nn as nn
from dataset import inverse_mel
from torchaudio import transforms


class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
           this module has learnable per-element affine parameters 
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    '''
       This module can be seen as the gradient of Conv1d with respect to its input. 
       It is also known as a fractionally-strided convolution 
       or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(nn.Module):
    '''
       Consider only residual links
    '''

    def __init__(self, in_channels=256, out_channels=512,
                 kernel_size=3, dilation=1, norm='gln', causal=False, skip_con=False):
        super(Conv1D_Block, self).__init__()
        # conv 1 x 1
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        # not causal don't need to padding, causal need to pad+1 = kernel_size
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise convolution
        self.dwconv = Conv1D(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        self.Sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.skip_con = skip_con
        if skip_con:
            self.skip_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)

    def forward(self, x):
        # x: N x C x L
        # N x O_C x L
        c = self.conv1x1(x)
        # N x O_C x L
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        # causal: N x O_C x (L+pad)
        # noncausal: N x O_C x L
        c = self.dwconv(c)
        # N x O_C x L
        if self.causal:
            c = c[:, :, :-self.pad]
        if self.skip_con:
            return x+self.Sc_conv(c), self.skip_conv(c)
        c = self.Sc_conv(c)
        return x+c


class ConvTasNet(nn.Module):
    '''
       ConvTasNet module
       N	Number of ﬁlters in autoencoder
       L	Length of the ﬁlters (in samples)
       B	Number of channels in bottleneck and the residual paths’ 1 × 1-conv blocks
       Sc	Number of channels in skip-connection paths’ 1 × 1-conv blocks
       H	Number of channels in convolutional blocks
       P	Kernel size in convolutional blocks
       X	Number of convolutional blocks in each repeat
       R	Number of repeats
    '''

    def __init__(self,
                 h,
                 ):
        super(ConvTasNet, self).__init__()
        self.h = h
        self.N = h.N  # 512
        self.L = h.L  # 16
        self.B = h.B  # 128
        self.H = h.H  # 512
        self.P = h.P  # 3
        self.X = h.X  # 8
        self.R = h.R  # 3
        self.norm = h.norm  # "gLN"
        self.num_spks = h.num_spks  # 1
        self.activate = h.activate  # "relu"
        self.causal = h.causal  # False
        self.skip_con = h.skip_con  # False 
        self.init_phase = h.init_phase
        self.hop_size = h.hop_size
        self.win_size = h.win_size
        self.n_fft = h.n_fft

        if self.init_phase.lower() == 'griffin_lim':
            self.transform = transforms.GriffinLim(n_fft=self.n_fft, n_iter=32, win_length=self.win_size, hop_length=self.hop_size, power=1)

        # n x 1 x T => n x N x T
        self.encoder = Conv1D(1, self.N, self.L, stride=self.L // 2, padding=0)
        # n x N x T  Layer Normalization of Separation
        self.LayerN_S = select_norm('gln', self.N)
        # n x B x T  Conv 1 x 1 of  Separation
        self.BottleN_S = Conv1D(self.N, self.B, 1)
        # Separation block
        # n x B x T => n x B x T
        self.separation = self._Sequential_repeat(
            self.R, self.X, in_channels=self.B, out_channels=self.H, kernel_size=self.P, norm=self.norm, causal=self.causal, skip_con=self.skip_con)
        # n x B x T => n x 2*N x T
        self.gen_masks = Conv1D(self.B, self.num_spks*self.N, 1)
        # n x N x T => n x 1 x L
        self.decoder = ConvTrans1D(self.N, 1, self.L, stride=self.L//2)
        # activation function
        active_f = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=0)
        }
        self.activation_type = self.activate
        self.activation = active_f[self.activate]
        self.num_spks = self.num_spks
        self.skip_con = self.skip_con

    def _Sequential_block(self, num_blocks, **block_kwargs):
        '''
           Sequential 1-D Conv Block
           input:
                 num_block: how many blocks in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        Conv1D_Block_lists = [Conv1D_Block(
            **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

        return Conv1D_Block_lists

    def _Sequential_repeat(self, num_repeats, num_blocks, **block_kwargs):
        '''
           Sequential repeats
           input:
                 num_repeats: Number of repeats
                 num_blocks: Number of block in every repeats
                 **block_kwargs: parameters of Conv1D_Block
        '''
        repeats_lists = []
        for i in range(num_repeats):
            repeats_lists += self._Sequential_block(
                num_blocks, **block_kwargs)
        return nn.ModuleList(repeats_lists)

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
                    self.h.num_mels,
                    self.h.sampling_rate,
                    self.h.hop_size,
                    self.h.win_size,
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
                                n_fft=self.h.n_fft,
                                hop_length=self.h.hop_size,
                                win_length=self.h.win_size,
                                window=torch.hann_window(self.h.win_size).to(mel.device),
                                )
        elif self.init_phase.lower() == 'griffin_lim':
            inv_wav = self.transform(inv_amp)
        x = inv_wav.unsqueeze(1)  # (B, 1, L)

        # if x.dim() >= 3:
        #     raise RuntimeError(
        #         "{} accept 1/2D tensor as input, but got {:d}".format(
        #             self.__name__, x.dim()))
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # x: n x 1 x L => n x N x T
        w = self.encoder(x)
        # n x N x L => n x B x L
        e = self.LayerN_S(w)
        e = self.BottleN_S(e)
        # n x B x L => n x B x L
        skip_connection = torch.zeros((e.shape[0], e.shape[1], e.shape[2]), device=e.device)
        for i in range(len(self.separation)):
            out = self.separation[i](e)
            if self.skip_con:
                e, skip = out
                skip_connection += skip
            else:
                e = out
        e = skip_connection if self.skip_con else e
        # n x B x L => n x num_spk*N x L
        m = self.gen_masks(e)
        d = w * m
        out_wav = self.decoder(d)

        return out_wav.squeeze(1)


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_convtasnet():
    x = torch.randn(320)
    nnet = ConvTasNet(skip_con=True)
    s = nnet(x)
    print(str(check_parameters(nnet))+' Mb')
    print(s[1].shape)


if __name__ == "__main__":
    test_convtasnet()