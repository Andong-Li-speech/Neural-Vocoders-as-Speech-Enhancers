import argparse
from typing import *
import numpy as np
import os
import soundfile as sf
import librosa as lib
from tqdm import tqdm
from glob import glob

def pad_shorter_np(audio1: np.ndarray, audio2: np.ndarray) -> tuple:
    """pad the shorter audio input to length of the longer one

    Args:
        audio1 (np.ndarray): shape: (num_samples,)
        audio2 (np.ndarray): shape: (num_samples,)

    Returns:
        (np.ndarray, np.ndarray): padded audios
    """
    padding_size = abs(audio1.shape[0] - audio2.shape[0])
    if padding_size > 0:
        padding = np.zeros(shape=(padding_size,))
        if audio1.shape[0] < audio2.shape[0]:
            audio1 = np.concatenate((audio1, padding))
        else:
            audio2 = np.concatenate((audio2, padding))
    return audio1, audio2


def las_rmse(reference_wav: str, synthesized_wav: str, sr: int, n_fft=1024, hop_length=256) -> float:
    """compute LAS-RMSE

    Args:
        reference_wav (str): reference .wav filepath
        synthesized_wav (str): synthesized .wav filepath
        sr (int, optional): sample rate. Defaults to SAMPLE_RATE.
        n_fft (int, optional): n_fft used for STFT. Defaults to N_FFT.
        hop_length (int, optional): hop_length used for STFT. Defaults to HOP_LENGTH.

    Returns:
        float: LAS-RMSE
    """
    ref_audio, _ = lib.load(reference_wav, sr=sr)
    syn_audio, _ = lib.load(synthesized_wav, sr=sr)
    
    ref_audio, syn_audio = pad_shorter_np(ref_audio, syn_audio)
    
    ref_stft = lib.stft(ref_audio, n_fft=n_fft, hop_length=hop_length)
    syn_stft = lib.stft(syn_audio, n_fft=n_fft, hop_length=hop_length)
    
    ref_amplitude = np.abs(ref_stft)
    syn_amplitude = np.abs(syn_stft)
    
    epsilon = 1e-10     # Convert to log scale (add a small constant to avoid log(0))
    ref_log_amplitude = np.log(ref_amplitude + epsilon)
    syn_log_amplitude = np.log(syn_amplitude + epsilon)
    
    return np.sqrt(np.mean((ref_log_amplitude - syn_log_amplitude) ** 2))


def cal_lsd_wrapper(ref_dir_dict, deg_dir, sr):
    input_files = glob(f"{deg_dir}/*.wav")

    lsd_score_list = []
    cnt = 0
    for deg_wav in tqdm(input_files):
        ref_dirname = ref_dir_dict[os.path.basename(deg_wav)]
        ref_wav = os.path.join(ref_dirname, os.path.basename(deg_wav))

        try:
            cur_lsd = las_rmse(ref_wav, deg_wav, sr=sr)
            lsd_score_list.append(cur_lsd)
            cnt += 1
        except:
            pass
    
    mean_lsd_score = np.mean(lsd_score_list)
    std_lsd_score = np.std(lsd_score_list)

    return mean_lsd_score, std_lsd_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute LAS-RMSE measure.")

    parser.add_argument('--ref_dir', required=False, 
                        default="/data4/liandong/datasets/LJSpeech-1.1",
                        help="Reference wav folder.")
    parser.add_argument('--deg_dir', required=False, 
                        default="/data4/liandong/PROJECTS/FreeV/File_Decodes/LJSpeech/ConvTasNet/griffin_lim",
                        help="Degraded wav folder.")
    parser.add_argument('--sr', required=False,
                        default=22050,
                        help="sampling rate.")

    args = parser.parse_args()
    # 
    ref_full_path_dict = {}
    if isinstance(args.ref_dir, List):
        for cur_ref_dir in args.ref_dir:
            cur_ref_list = glob(f'{cur_ref_dir}/*.wav') + \
                           glob(f'{cur_ref_dir}/*/*.wav') + \
                           glob(f'{cur_ref_dir}/*/*/*.wav') + \
                           glob(f'{cur_ref_dir}/*/*/*/*.wav')
            for cur_cur_ref_path in cur_ref_list:
                cur_dirname, cur_filename = os.path.split(cur_cur_ref_path)
                ref_full_path_dict[cur_filename] = cur_dirname
    elif isinstance(args.ref_dir, str):
        ref_list = glob(f'{args.ref_dir}/*.wav') + \
                   glob(f'{args.ref_dir}/*/*.wav') + \
                   glob(f'{args.ref_dir}/*/*/*.wav')
        for cur_ref_path in ref_list:
            cur_dirname, cur_filename = os.path.split(cur_ref_path)
            ref_full_path_dict[cur_filename] = cur_dirname

    mean_, std_ = cal_lsd_wrapper(ref_full_path_dict, args.deg_dir, args.sr)
    print("LSD score: mean->{:4f}, std->{:4f}.".format(mean_, std_))
