import torch
import torch.nn as nn

from dataset import inverse_mel


class GLSTM(nn.Module):
    def __init__(self, in_features=None, out_features=None, mid_features=None, hidden_size=896, groups=2):
        super(GLSTM, self).__init__()
   
        hidden_size_t = hidden_size // groups
     
        self.lstm_list1 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList([nn.LSTM(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
     
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
     
        self.groups = groups
        self.mid_features = mid_features
     
    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
    
        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack([self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)
    
        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat([self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()

        return out


class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=(0,0)):
        super(GluConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=output_padding)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        output_padding=output_padding)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out

# here we also adopt the resi-mask + phase strategy
class GCRN(nn.Module):
    def __init__(self, h):
        super(GCRN, self).__init__()
        self.h = h
        self.win_size = h.win_size
        self.hop_size = h.hop_size
        self.n_fft = h.n_fft

        self.conv1 = GluConv2d(in_channels=1, out_channels=16, kernel_size=(1,5), stride=(1,2))
        self.conv2 = GluConv2d(in_channels=16, out_channels=32, kernel_size=(1,3), stride=(1,2))
        self.conv3 = GluConv2d(in_channels=32, out_channels=64, kernel_size=(1,3), stride=(1,2))
        self.conv4 = GluConv2d(in_channels=64, out_channels=128, kernel_size=(1,3), stride=(1,2))
        self.conv5 = GluConv2d(in_channels=128, out_channels=128, kernel_size=(1,3), stride=(1,2))
        self.conv6 = GluConv2d(in_channels=128, out_channels=128, kernel_size=(1,3), stride=(1,2))
        
        self.glstm = GLSTM(groups=2)

        self.conv6_t_1 = GluConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(1,3), stride=(1,2))
        self.conv5_t_1 = GluConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(1,3), stride=(1,2))
        self.conv4_t_1 = GluConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1,3), stride=(1,2))
        self.conv3_t_1 = GluConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1,3), stride=(1,2))
        self.conv2_t_1 = GluConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1,3), stride=(1,2))
        self.conv1_t_1 = GluConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1,5), stride=(1,2))  # mag-branch
        
        self.conv6_t_2 = GluConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(1,3), stride=(1,2))
        self.conv5_t_2 = GluConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(1,3), stride=(1,2))
        self.conv4_t_2 = GluConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1,3), stride=(1,2))
        self.conv3_t_2 = GluConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1,3), stride=(1,2))
        self.conv2_t_2 = GluConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1,3), stride=(1,2))
        self.conv1_t_2 = GluConvTranspose2d(in_channels=32, out_channels=2, kernel_size=(1,5), stride=(1,2))  # phase-branch
        
        self.bn1 = nn.LayerNorm(255)
        self.bn2 = nn.LayerNorm(127)
        self.bn3 = nn.LayerNorm(63)
        self.bn4 = nn.LayerNorm(31)
        self.bn5 = nn.LayerNorm(15)
        self.bn6 = nn.LayerNorm(7)
                   
        self.bn6_t_1 = nn.LayerNorm(15)
        self.bn5_t_1 = nn.LayerNorm(31)
        self.bn4_t_1 = nn.LayerNorm(63)
        self.bn3_t_1 = nn.LayerNorm(127)
        self.bn2_t_1 = nn.LayerNorm(255)
        self.bn1_t_1 = nn.LayerNorm(513)

        self.bn6_t_2 = nn.LayerNorm(15)
        self.bn5_t_2 = nn.LayerNorm(31)
        self.bn4_t_2 = nn.LayerNorm(63)
        self.bn3_t_2 = nn.LayerNorm(127)
        self.bn2_t_2 = nn.LayerNorm(255)
        self.bn1_t_2 = nn.LayerNorm(513)

        self.elu = nn.ELU(inplace=True)
        
        self.fc1 = nn.Linear(in_features=513, out_features=513)
        self.fc2 = nn.Linear(in_features=513, out_features=513)

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
                    self.h.n_fft,
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
        
        inv_amp_ = inv_amp.log()

        inpt = inv_amp_.transpose(-2, -1).contiguous()  # (B, T, F)
        e1 = self.elu(self.bn1(self.conv1(inpt.unsqueeze(1))))
        e2 = self.elu(self.bn2(self.conv2(e1)))
        e3 = self.elu(self.bn3(self.conv3(e2)))
        e4 = self.elu(self.bn4(self.conv4(e3)))
        e5 = self.elu(self.bn5(self.conv5(e4)))
        e6 = self.elu(self.bn6(self.conv6(e5)))
        
        out = e6
        
        out = self.glstm(out)

        out = torch.cat((out, e6), dim=1)

        d6_1 = self.elu(torch.cat((self.bn6_t_1(self.conv6_t_1(out)), e5), dim=1))
        d5_1 = self.elu(torch.cat((self.bn5_t_1(self.conv5_t_1(d6_1)), e4), dim=1))
        d4_1 = self.elu(torch.cat((self.bn4_t_1(self.conv4_t_1(d5_1)), e3), dim=1))
        d3_1 = self.elu(torch.cat((self.bn3_t_1(self.conv3_t_1(d4_1)), e2), dim=1))
        d2_1 = self.elu(torch.cat((self.bn2_t_1(self.conv2_t_1(d3_1)), e1), dim=1))
        d1_1 = self.elu(self.bn1_t_1(self.conv1_t_1(d2_1)))
        
        d6_2 = self.elu(torch.cat((self.bn6_t_2(self.conv6_t_2(out)), e5), dim=1))
        d5_2 = self.elu(torch.cat((self.bn5_t_2(self.conv5_t_2(d6_2)), e4), dim=1))
        d4_2 = self.elu(torch.cat((self.bn4_t_2(self.conv4_t_2(d5_2)), e3), dim=1))
        d3_2 = self.elu(torch.cat((self.bn3_t_2(self.conv3_t_2(d4_2)), e2), dim=1))
        d2_2 = self.elu(torch.cat((self.bn2_t_2(self.conv2_t_2(d3_2)), e1), dim=1))
        d1_2 = self.elu(self.bn1_t_2(self.conv1_t_2(d2_2)))
        
        resi_mask = self.fc1(d1_1).transpose(-2, -1).contiguous().squeeze(1)  # (B, F, T)
        mag = torch.exp(resi_mask + inv_amp_)

        real, imag = self.fc2(d1_2).transpose(-2, -1).contiguous().chunk(2, dim=1)
        phase = torch.atan2(imag.squeeze(1), real.squeeze(1))  # (B, F, T)

        rea_, imag_ = mag * torch.cos(phase), mag * torch.sin(phase)

        # output
        decode_mag = mag
        decode_phase = phase
        logamp = torch.log(decode_mag + 1e-5)
        out_spec = torch.complex(rea_, imag_)
        out_wav = torch.istft(out_spec,
                              n_fft=self.n_fft,
                              hop_length=self.hop_size,
                              win_length=self.win_size,
                              window=torch.hann_window(self.win_size).to(mel.device),
                              )

        return logamp, decode_phase, rea_, imag_, out_wav

 
if __name__ == '__main__':
    net = GCRN().cuda()
    x = torch.abs(torch.rand([3, 513, 31])).cuda()
    y = net(x)
    print('{} -> {}'.format(x.shape, y.shape))