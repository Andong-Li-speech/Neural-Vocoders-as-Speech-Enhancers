import argparse
from typing import *
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from glob import glob
from typing import Dict
from pymcd.mcd import Calculate_MCD


def cal_mcd_wrapper(ref_dir_dict, deg_files, sr, n_fft=1024, n_shift=256, mcd_dict: Dict=None):

    cnt = 0
    for deg_wav in tqdm(deg_files):
        ref_dirname = ref_dir_dict[os.path.basename(deg_wav)]
        ref_wav = os.path.join(ref_dirname, os.path.basename(deg_wav))
        mcd_toolbox = Calculate_MCD(MCD_mode="plain")
        cur_mcd = mcd_toolbox.calculate_mcd(ref_wav, deg_wav)

        cnt += 1
        mcd_dict[f"{os.path.basename(deg_wav)}"] = cur_mcd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute MCD measure.")

    parser.add_argument('--ref_dir', required=False, 
                        default=["/data4/liandong/datasets/LibriTTS/LibriTTS/dev-clean", "/data4/liandong/datasets/LibriTTS/LibriTTS/dev-other"],
                        help="Reference wav folder.")
    parser.add_argument('--deg_dir', required=False, 
                        default="/data4/liandong/PROJECTS/FreeV/File_Decodes/LibriTTS/joint_denoise_vocoder/vocoder/2M_steps",
                        help="Degraded wav folder.")
    parser.add_argument(
        '--sr', required=False,
        default=24000,
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
        mcd_dict = manager.dict()
        processes = []
        for f in file_lists:
            p = mp.Process(target=cal_mcd_wrapper, args=(ref_full_path_dict, f, args.sr, args.n_fft, args.n_shift, mcd_dict))
            p.start()
            processes.append(p)
        
        # wait for all process
        for p in processes:
            p.join()
        
        # convert to standard list
        mcd_dict = dict(mcd_dict)

        mean_mcd = np.mean(np.array([v for v in mcd_dict.values()]))
        std_mcd = np.std(np.array([v for v in mcd_dict.values()]))
        print("MCD score: mean->{:4f}, std->{:4f}.".format(mean_mcd, std_mcd))
