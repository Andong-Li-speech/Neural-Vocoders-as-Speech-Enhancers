from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import glob
import os, sys
import json
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import AttrDict
from dataset import mel_spectrogram
import soundfile as sf
import numpy as np
import soundfile as sf
import librosa as lib
import time
import tqdm

#
from Models import iSTFTNet

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(h):
    generator = eval(h.model_name)(h).to(device)

    state_dict_g = load_checkpoint(h.checkpoint_file_load, device)
    generator.load_state_dict(state_dict_g['generator'])

    post_str = os.path.splitext(h.test_input_wavs_dir)[-1]
    if post_str in ['.txt', '.scp']:
        filelist = []
        lines = open(h.test_input_wavs_dir, 'r').readlines()
        for l in lines:
            cur_filename = l.strip().split('/')[1].split('|')[0]
            filelist.append(os.path.join(h.raw_wavfile_path, cur_filename))
    else:  # dir
        filelist = sorted(os.listdir(h.test_input_mels_dir if h.test_mel_load else h.test_input_wavs_dir))

    os.makedirs(h.test_output_dir, exist_ok=True)

    generator.eval()
    try:
        generator.remove_weight_norm()
    except:
        pass
    l = 0
    with torch.no_grad():
        starttime = time.time()
        for filename in tqdm.tqdm(filelist):
            # if h.test_mel_load:
            if h.test_mel_load:
                mel = np.load(os.path.join(h.test_input_wavs_dir, filename))
                x = torch.FloatTensor(mel).to(device)
                x = x.transpose(1,2)
            else:
                if post_str in ['.txt', '.scp']:
                    raw_wav, orig_sr = sf.read(filename)
                else:
                    raw_wav, orig_sr = sf.read(os.path.join(h.test_input_wavs_dir, filename))
                if orig_sr != h.sampling_rate:
                    raw_wav = lib.core.resample(raw_wav, orig_sr=orig_sr, target_sr=h.sampling_rate)

                raw_wav = torch.FloatTensor(raw_wav.astype('float32')).to(device)
                x = get_mel(raw_wav.unsqueeze(0))
            
            y_g_list = generator(x)
            if isinstance(y_g_list, torch.Tensor):  # 对于时域方法
                y_g = y_g_list
            else:
                y_g = y_g_list[-1]  # 对于频域方法
            audio = y_g.squeeze()
            audio = audio.cpu().numpy()
            audiolen=len(audio)
            if post_str in ['.txt', '.scp']:
                sf.write(os.path.join(h.test_output_dir, filename.split('/')[-1]), audio, h.sampling_rate, 'PCM_16')
            else:
                sf.write(os.path.join(h.test_output_dir, filename.split('.')[0] + '.wav'), audio, h.sampling_rate, 'PCM_16')

            # print(pp)
            l += audiolen
        end=time.time()
        print(end-starttime)
        print(l/22050)
        print(l/22050/(end-starttime)) 


def main():
    print("Initializing Training Process...")
    parse = argparse.ArgumentParser('Vocoder configs.')
    parse.add_argument('--cfg_filename', type=str, required=True, default='cfgs/istftnet_config.json',
                        help='Json for configurations.')
    args = parse.parse_args()
    dir_path = os.getcwd()
    config_file = os.path.join(dir_path, args.cfg_filename)
    
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    inference(h)


if __name__ == '__main__':
    main()
