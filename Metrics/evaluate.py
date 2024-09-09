from typing import *
import argparse
import functools
import numpy as np
import os
import torch
from glob import glob
import torchaudio as ta


from cargan.evaluate.objective.metrics import Pitch
from cargan.preprocess.pitch import from_audio
from scipy.io.wavfile import read
from tqdm import tqdm

SR_TARGET = 22050  # input sampling rate
MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, audio = read(full_path)
    if sampling_rate != SR_TARGET:
        raise IOError(
            f'Sampling rate of the file {full_path} is {sampling_rate} Hz, but the model requires {SR_TARGET} Hz'
        )

    audio = audio / MAX_WAV_VALUE

    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    return audio


def evaluate(ref_full_path_dict, deg_filelist):
    """Perform objective evaluation"""
    gpu = 2 if torch.cuda.is_available() else None
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    torch.cuda.empty_cache()

    resampler_22k = ta.transforms.Resample(SR_TARGET, 22050).to(device)

    # Modules for evaluation metrics
    batch_metrics_periodicity = Pitch()
    periodicity_fn = functools.partial(from_audio, gpu=gpu)

    with torch.no_grad():
        for cur_deg_filename in tqdm(deg_filelist): 
            ref_dirname = ref_full_path_dict[os.path.basename(cur_deg_filename)]
            y = load_wav(os.path.join(ref_dirname, os.path.basename(cur_deg_filename)))
            y_g_hat = load_wav(cur_deg_filename)
            y = y.to(device)
            y_g_hat = y_g_hat.to(device)

            y_22k = resampler_22k(y)
            y_g_hat_22k = resampler_22k(y_g_hat)

            min_ = min(y_22k.shape[-1], y_g_hat_22k.shape[-1])
            y_22k = y_22k[:, :min_]
            y_g_hat_22k = y_g_hat_22k[:, :min_]

            # Periodicity calculation
            true_pitch, true_periodicity = periodicity_fn(y_22k)
            pred_pitch, pred_periodicity = periodicity_fn(y_g_hat_22k)
            batch_metrics_periodicity.update(true_pitch, true_periodicity, pred_pitch, pred_periodicity)

    results = batch_metrics_periodicity()

    return {
        'Periodicity': results['periodicity'],
        'V/UV F1': results['f1'],
        'Pitch': results['pitch'],
        'Periodicity_std': results['periodicity_std'],
        'V/UV F1_std': results['f1_std'],
        'Pitch_std': results['pitch_std']
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', required=False, 
                        default="/data4/liandong/datasets/LJSpeech-1.1",
                        help="Reference wav folder.")
    parser.add_argument('--deg_dir', required=False, 
                        default="/data4/liandong/PROJECTS/FreeV/File_Decodes/LJSpeech/HD-DEMUCAS/rand_phase",
                        help="Degraded wav folder.")
    args = parser.parse_args()
    ref_dir, deg_dir = args.ref_dir, args.deg_dir

    deg_filelist = glob(f"{deg_dir}/*.wav")
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

    # Evaluate waveforms
    results_tot = {
        'Periodicity': 0.0,
        'V/UV F1': 0.0,
        'Pitch': 0.0,
        'Periodicity_std': 0.0,
        'V/UV F1_std': 0.0,
        'Pitch_std': 0.0,
        'dir_results': {},
    }
    results = evaluate(ref_full_path_dict, deg_filelist)

    # Print to stdout
    print(results)


if __name__ == '__main__':
    main()
    