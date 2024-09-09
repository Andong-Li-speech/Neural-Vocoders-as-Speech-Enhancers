import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ASPResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ASPResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

class PSPResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(PSPResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class APNet(torch.nn.Module):
    def __init__(self, h):
        super(APNet, self).__init__()
        self.h = h
        self.ASP_num_kernels = len(h.ASP_resblock_kernel_sizes)
        self.PSP_num_kernels = len(h.PSP_resblock_kernel_sizes)

        self.ASP_input_conv = weight_norm(Conv1d(h.num_mels, h.ASP_channel, h.ASP_input_conv_kernel_size, 1, 
                                                 padding=get_padding(h.ASP_input_conv_kernel_size, 1)))
        self.PSP_input_conv = weight_norm(Conv1d(h.num_mels, h.PSP_channel, h.PSP_input_conv_kernel_size, 1, 
                                                 padding=get_padding(h.PSP_input_conv_kernel_size, 1)))

        self.ASP_ResNet = nn.ModuleList()
        for j, (k, d) in enumerate(zip(h.ASP_resblock_kernel_sizes, h.ASP_resblock_dilation_sizes)):
            self.ASP_ResNet.append(ASPResBlock(h, h.ASP_channel, k, d))

        self.PSP_ResNet = nn.ModuleList()
        for j, (k, d) in enumerate(zip(h.PSP_resblock_kernel_sizes, h.PSP_resblock_dilation_sizes)):
            self.PSP_ResNet.append(PSPResBlock(h, h.PSP_channel, k, d))

        self.ASP_output_conv = weight_norm(Conv1d(h.ASP_channel, h.n_fft//2+1, h.ASP_output_conv_kernel_size, 1, 
                                                  padding=get_padding(h.ASP_output_conv_kernel_size, 1)))
        self.PSP_output_R_conv = weight_norm(Conv1d(h.PSP_channel, h.n_fft//2+1, h.PSP_output_R_conv_kernel_size, 1, 
                                                    padding=get_padding(h.PSP_output_R_conv_kernel_size, 1)))
        self.PSP_output_I_conv = weight_norm(Conv1d(h.PSP_channel, h.n_fft//2+1, h.PSP_output_I_conv_kernel_size, 1, 
                                                    padding=get_padding(h.PSP_output_I_conv_kernel_size, 1)))

        self.ASP_output_conv.apply(init_weights)
        self.PSP_output_R_conv.apply(init_weights)
        self.PSP_output_I_conv.apply(init_weights)

    def forward(self, mel):

        logamp = self.ASP_input_conv(mel)
        logamps = None
        for j in range(self.ASP_num_kernels):
            if logamps is None:
                logamps = self.ASP_ResNet[j](logamp)
            else:
                logamps += self.ASP_ResNet[j](logamp)
        logamp = logamps / self.ASP_num_kernels
        logamp = F.leaky_relu(logamp)
        logamp = self.ASP_output_conv(logamp)

        pha = self.PSP_input_conv(mel)
        phas = None
        for j in range(self.PSP_num_kernels):
            if phas is None:
                phas = self.PSP_ResNet[j](pha)
            else:
                phas += self.PSP_ResNet[j](pha)
        pha = phas / self.PSP_num_kernels
        pha = F.leaky_relu(pha)   
        R = self.PSP_output_R_conv(pha)
        I = self.PSP_output_I_conv(pha)

        pha = torch.atan2(I,R)

        rea = torch.exp(logamp)*torch.cos(pha)
        imag = torch.exp(logamp)*torch.sin(pha)

        spec = torch.complex(rea, imag)

        audio = torch.istft(spec, self.h.n_fft, hop_length=self.h.hop_size, win_length=self.h.win_size, window=torch.hann_window(self.h.win_size).to(mel.device), center=True)

        return logamp, pha, rea, imag, audio


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2

def phase_loss(phase_r, phase_g, n_fft, frames):

    MSELoss = torch.nn.MSELoss()

    GD_matrix = torch.triu(torch.ones(n_fft//2+1,n_fft//2+1),diagonal=1)-torch.triu(torch.ones(n_fft//2+1,n_fft//2+1),diagonal=2)-torch.eye(n_fft//2+1)
    GD_matrix = GD_matrix.to(phase_g.device)

    GD_r = torch.matmul(phase_r.permute(0,2,1), GD_matrix)
    GD_g = torch.matmul(phase_g.permute(0,2,1), GD_matrix)

    PTD_matrix = torch.triu(torch.ones(frames,frames),diagonal=1)-torch.triu(torch.ones(frames,frames),diagonal=2)-torch.eye(frames)
    PTD_matrix = PTD_matrix.to(phase_g.device)

    PTD_r = torch.matmul(phase_r, PTD_matrix)
    PTD_g = torch.matmul(phase_g, PTD_matrix)

    IP_loss = torch.mean(-torch.cos(phase_r-phase_g))
    GD_loss = torch.mean(-torch.cos(GD_r-GD_g))
    PTD_loss = torch.mean(-torch.cos(PTD_r-PTD_g))


    return IP_loss, GD_loss, PTD_loss


def amplitude_loss(log_amplitude_r, log_amplitude_g):

    MSELoss = torch.nn.MSELoss()

    amplitude_loss = MSELoss(log_amplitude_r, log_amplitude_g)

    return amplitude_loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def STFT_consistency_loss(rea_r, rea_g, imag_r, imag_g):

    C_loss=torch.mean(torch.mean((rea_r-rea_g)**2+(imag_r-imag_g)**2,(1,2)))
    
    return C_loss