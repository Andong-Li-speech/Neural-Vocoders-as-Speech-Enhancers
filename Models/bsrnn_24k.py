import numpy as np
import torch
import torch.nn as nn


class ResRNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout: float = 0.,
                 causal: bool = True,
                 residual: bool = True,
                 ):
        super(ResRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.causal = causal
        self.residual = residual
        self.eps = torch.finfo(torch.float32).eps
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=not causal)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*(int(not causal) + 1), input_size)

    def forward(self, input):
        """
        input: (B, T, nband, C)
        return: (B, T, nband, C)
        """
        batch_size, t1, t2, E = input.shape
        x = self.norm(input)
        x = x.view(batch_size * t1, t2, E)
        rnn_output, _ = self.rnn(self.dropout(x))
        rnn_output = self.proj(rnn_output).view(*input.shape)
        if self.residual:
            return input + rnn_output
        else:
            return rnn_output


class BSNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 dropout: float=0.,
                 causal: bool = True,
                 ):
        super(BSNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.causal = causal

        self.time_rnn = ResRNN(input_size=in_channels, hidden_size=hidden_channels, dropout=dropout, causal=causal)

        self.band_rnn = ResRNN(input_size=in_channels, hidden_size=hidden_channels, dropout=dropout, causal=False)

        self.out_norm = nn.LayerNorm(in_channels)

    def forward(self, input):
        """
        input: (B, nband, T, C)
        return: (B, nband, T, C)
        """
        # time-modeling
        x = input
        output = self.time_rnn(x)  # (B, nband, T, C)
        # band-modeling
        input = output.transpose(1, 2).contiguous()  # (B, T, nband, C)
        output = self.band_rnn(input)
        # reshape
        output = output.transpose(1, 2).contiguous()

        return self.out_norm(output)


class BSRNN_24k(nn.Module):
    def __init__(self, h):
        super(BSRNN_24k, self).__init__()
        self.h = h
        self.sr = h.sampling_rate
        self.win_size = h.win_size
        self.win_shift = h.hop_size
        self.n_fft = h.n_fft
        self.fft_reso = (self.sr / self.n_fft)
        self.feature_dim = h.feature_dim
        self.num_repeat = h.num_repeat
        self.dropout = h.dropout
        self.causal = h.causal
        self.eps = torch.finfo(torch.float32).eps

        # 0-1k (100 hz hop), 1k-4k (250 hz hop), 4k-8k (500 hz hop), 8k-16k (1k hz hop), 16k-20k (2k hop), >20k remains 1 band
        bw_100 = int(np.floor(100 / self.fft_reso))  # 4 bands
        bw_250 = int(np.floor(250 / self.fft_reso))  # 11 bands
        bw_500 = int(np.floor(500 / self.fft_reso))  # 23 bands
        bw_1k = int(np.floor(1000 / self.fft_reso))  # 46 bands

        self.band_width = [bw_100] * 10  # 10 * 4 = 40
        self.band_width += [bw_250] * 12  # 12 * 11 = 132
        self.band_width += [bw_500] * 8  # 8 * 23 = 184
        self.band_width += [bw_1k] * 3   # 3 * 46 = 138
        self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))  # remains
        self.nband = len(self.band_width)
        print(f'Totally splitting {len(self.band_width)} bands.')

        self.encoder = nn.ModuleList([])
        for i in range(self.nband):
            self.encoder.append(
                nn.Sequential(
                    nn.LayerNorm(self.band_width[i]),
                    nn.Linear(self.band_width[i], self.feature_dim)
                )
            )
        self.separator = nn.ModuleList([])
        for i in range(self.num_repeat):
            self.separator.append(BSNet(in_channels=self.feature_dim, hidden_channels=self.feature_dim, dropout=self.dropout, causal=self.causal))

        self.decoder_mag, self.decoder_phase = nn.ModuleList([]), nn.ModuleList([])
        for i in range(self.nband):
            self.decoder_mag.append(
                nn.Sequential(
                    nn.LayerNorm(self.feature_dim),
                    nn.Linear(self.feature_dim, 4 * self.feature_dim),
                    nn.GELU(),
                    nn.Linear(4 * self.feature_dim, int(self.band_width[i]))
                )
            )
            self.decoder_phase.append(
                nn.Sequential(
                    nn.LayerNorm(self.feature_dim),
                    nn.Linear(self.feature_dim, 4 * self.feature_dim),
                    nn.GELU(),
                    nn.Linear(4 * self.feature_dim, int(self.band_width[i] * 2)),
                )
            )

    def forward(self, inpt):
        """
        input: (B, F, T)
        return: 
            logamp: (B, F, T)
            pha: (B, F, T)
            rea: (B, F, T)
            imag: (B, F, T)
            audio: (B, 1, L)
        """
        subband_spec_list = []
        band_idx = 0
        for i in range(len(self.band_width)):
            cur_subband_spec = inpt[:, band_idx: band_idx + self.band_width[i]].transpose(-2, -1).contiguous()  # (B, T, fw)
            subband_spec_list.append(self.encoder[i](cur_subband_spec.view([*cur_subband_spec.shape[:2], -1])))
            band_idx += self.band_width[i]

        subband_feature = torch.stack(subband_spec_list, dim=1)  # (B, nband, T, C)

        # enhancer
        x = subband_feature
        for layer in self.separator:
            x = layer(x)
        feature_output = x  # (B, nband, T, C)

        # decoder
        decode_resi_mag_list, decode_phase_list = [], []
        for i in range(len(self.band_width)):
            # mag
            this_resi_mag = self.decoder_mag[i](feature_output[:, i].contiguous())
            # phase
            this_comp = self.decoder_phase[i](feature_output[:, i].contiguous())
            this_real, this_imag = this_comp.chunk(2, dim=-1)
            this_phase = torch.atan2(this_imag, this_real)

            decode_resi_mag_list.append(this_resi_mag)
            decode_phase_list.append(this_phase)
        decode_resi_mag, decode_phase = torch.cat(decode_resi_mag_list, dim=-1), torch.cat(decode_phase_list, dim=-1)  # (B, T, F)

        decode_mag, decode_phase = torch.exp(decode_resi_mag.transpose(-2, -1) + inpt).contiguous(), decode_phase.transpose(-2, -1)  # (B, F, T)

        # output
        logamp = torch.log(torch.clamp_min_(decode_mag, 1e-5))
        pha = decode_phase
        rea = decode_mag * torch.cos(decode_phase)
        imag = decode_mag * torch.sin(decode_phase)

        out_spec = torch.complex(rea, imag)  # complex-tensor, (B, F, T)
        out_wav = torch.istft(out_spec,
                              n_fft=self.n_fft,
                              hop_length=self.win_shift,
                              win_length=self.win_size,
                              window=torch.hann_window(self.win_size).to(inpt.device),
                              )

        return logamp, pha, rea, imag, out_wav
