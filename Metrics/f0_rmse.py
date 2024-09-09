import argparse
from typing import *
import numpy as np
from tqdm import tqdm
from glob import glob
import os
import librosa
import soundfile as sf
import multiprocessing as mp
from tqdm import tqdm
from typing import Dict
import pysptk
import pyworld as pw
from fastdtw import fastdtw
from scipy import spatial


def world_extract(
    x: np.ndarray,
    fs: int,
    f0min: int = 40,
    f0max: int = 800,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
) -> np.ndarray:
    """Extract World-based acoustic features.

    Args:
        x (ndarray): 1D waveform array.
        fs (int): Minimum f0 value (default=40).
        f0 (int): Maximum f0 value (default=800).
        n_shift (int): Shift length in point (default=256).
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).

    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).
        ndarray: F0 sequence (N,).

    """
    # extract features
    x = x.astype(np.float64)
    f0, time_axis = pw.harvest(
        x,
        fs,
        f0_floor=f0min,
        f0_ceil=f0max,
        frame_period=n_shift / fs * 1000,
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=n_fft)
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)

    return mcep, f0


def _get_best_mcep_params(fs: int):
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


def cal_f0_wrapper(ref_dir_dict, deg_files, sr, n_fft=1024, n_shift=256, f0_dict: Dict=None, f0min=None, f0max=None):
    cnt = 0
    for deg_wav in tqdm(deg_files):
        ref_dirname = ref_dir_dict[os.path.basename(deg_wav)]
        ref_wav = os.path.join(ref_dirname, os.path.basename(deg_wav))

        ref, ref_sr = sf.read(ref_wav)
        deg, deg_sr = sf.read(deg_wav)

        if ref_sr != sr:
            ref = librosa.resample(ref, orig_sr=ref_sr, target_sr=sr)
        if deg_sr != sr:
            deg = librosa.resample(deg, orig_sr=deg_sr, target_sr=sr)

        min_len = min(len(ref), len(deg))
        ref, deg = ref[:min_len], deg[:min_len]

        # extract ground truth and converted features
        gen_mcep, gen_f0 = world_extract(
            x=deg,
            fs=deg_sr,
            f0min=f0min,
            f0max=f0max,
            n_fft=n_fft,
            n_shift=n_shift,
            mcep_dim=None,
            mcep_alpha=None,
        )
        gt_mcep, gt_f0 = world_extract(
            x=ref,
            fs=ref_sr,
            f0min=f0min,
            f0max=f0max,
            n_fft=n_fft,
            n_shift=n_shift,
            mcep_dim=None,
            mcep_alpha=None,
        )

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_f0_dtw = gen_f0[twf[0]]
        gt_f0_dtw = gt_f0[twf[1]]

        # Get voiced part
        nonzero_idxs = np.where((gen_f0_dtw != 0) & (gt_f0_dtw != 0))[0]
        gen_f0_dtw_voiced = gen_f0_dtw[nonzero_idxs]
        gt_f0_dtw_voiced = gt_f0_dtw[nonzero_idxs]

        # F0 RMSE
        f0_rmse = np.sqrt(np.mean((gen_f0_dtw_voiced - gt_f0_dtw_voiced) ** 2))
        f0_dict[f"{os.path.basename(deg_wav)}"] = f0_rmse

        cnt += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute F0-RMSE measure.")

    parser.add_argument('--ref_dir', required=False, 
                        default="/data4/liandong/datasets/LJSpeech-1.1",
                        help="Reference wav folder.")
    parser.add_argument('--deg_dir', required=False, 
                        default="/data4/liandong/PROJECTS/FreeV/File_Decodes/LJSpeech/ConvTasNet/griffin_lim",
                        help="Degraded wav folder.")
    parser.add_argument(
        '--sr', required=False,
        default=22050,
        help="Target sampling rate."
    )
    parser.add_argument(
        '--n_fft', required=False, 
        default=1024,
        help="N-FFT."
    )
    parser.add_argument(
        '--n_shift', required=False, 
        default=256,
        help="Window shift."
    )
    parser.add_argument(
        "--f0min",
        default=40,
        type=int,
        help="Minimum f0 value.",
    )
    parser.add_argument(
        "--f0max",
        default=800,
        type=int,
        help="Maximum f0 value.",
    )
    parser.add_argument(
        '--jb', required=False,
        default=16,
        help='The number or proceeds.'
    )

    args = parser.parse_args()

    gen_file_lists = glob(f"{args.deg_dir}/*.wav")
    file_lists = np.array_split(gen_file_lists, args.jb)
    file_lists = [f_list.tolist() for f_list in file_lists]

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

    # multi processing
    with mp.Manager() as manager:
        f0_dict = manager.dict()
        processes = []
        for f in file_lists:
            p = mp.Process(target=cal_f0_wrapper, args=(ref_full_path_dict, f, args.sr, args.n_fft, args.n_shift, f0_dict, args.f0min, args.f0max))
            p.start()
            processes.append(p)
        
        # wait for all process
        for p in processes:
            p.join()
        
        # convert to standard list
        f0_dict = dict(f0_dict)

        mean_f0 = np.mean(np.array([v for v in f0_dict.values()]))
        std_f0 = np.std(np.array([v for v in f0_dict.values()]))
        print("F0-RMSE score: mean->{:4f}, std->{:4f}.".format(mean_f0, std_f0))
