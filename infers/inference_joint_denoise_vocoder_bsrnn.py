from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import glob
import os, sys
import json
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import AttrDict
from dataset_joint_denoise_vocoder import mel_spectrogram, inverse_mel, amp_pha_specturm
import soundfile as sf
import numpy as np
import soundfile as sf
import librosa as lib
import time
import tqdm

from Models.bsrnn_24k import BSRNN_24k

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

def get_inv_mel(x):
    return inverse_mel(x, n_fft=h.n_fft, num_mels=h.num_mels, sampling_rate=h.sampling_rate, hop_size=h.hop_size, win_size=h.win_size, fmin=h.fmin, fmax=h.fmax)

def get_log_mag(x):
    return amp_pha_specturm(x, n_fft=h.n_fft, hop_size=h.hop_size, win_size=h.win_size)[0]

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(h, task_mode=None):
    generator = eval(h.model_name)(h).to(device)

    state_dict_g = load_checkpoint(h.checkpoint_file_load, device)
    generator.load_state_dict(state_dict_g['generator'])

    post_str = os.path.splitext(h.test_input_wavs_dir)[-1]
    if post_str in ['.txt', '.scp']:
        filelist = []
        lines = open(h.test_input_wavs_dir, 'r').readlines()
        for l in lines:
            if task_mode == 'vocoder':
                cur_filename = l.strip().split('|')[0]
            elif task_mode == 'denoise':
                cur_filename = os.path.splitext(l.strip().split('\t')[0])[0]
            filelist.append(os.path.join(h.raw_wavfile_path, f'{cur_filename}.wav'))
    else:  # dir
        filelist = sorted(os.listdir(h.test_input_mels_dir if h.test_mel_load else h.test_input_wavs_dir))

    os.makedirs(h.test_output_dir, exist_ok=True)

    generator.eval()
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
                if task_mode == 'vocoder':
                    x = get_mel(raw_wav.unsqueeze(0))
                    x = get_inv_mel(x).abs().clamp_min(1e-5).log()
                elif task_mode == 'denoise':
                    x = amp_pha_specturm(raw_wav.unsqueeze(0), n_fft=h.n_fft, hop_size=h.hop_size, win_size=h.win_size)[0]
            
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
            
        #     write(output_file, h.sampling_rate, audio)
        #     print(output_file)
        end=time.time()
        print(end-starttime)
        print(l/22050)
        print(l/22050/(end - starttime)) 


def main():
    print("Initializing Inference Process...")
    parse = argparse.ArgumentParser('')
    parse.add_argument('--cfg_filename', type=str, required=True, default='cfgs/bsrnn_joint_denoise_vocoder_config.json',
                        help='Json for configurations.')
    parse.add_argument("--processing_mode", type=str, required=True, default="denoise", choices=["denoise", "vocoder"],
                       help="Processing mode.")
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
    inference(h, task_mode=args.processing_mode)


if __name__ == '__main__':
    main()

