import argparse
from typing import *
import numpy as np
from glob import glob
import os
import soundfile as sf
import librosa as lib
from pystoi import stoi
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict


def cal_stoi_wrapper(ref_dir_dict, deg_files, stoi_dict:Dict=None, estoi_dict: Dict=None):
    cnt = 0
    for deg_wav in tqdm(deg_files):
        ref_dirname = ref_dir_dict[os.path.basename(deg_wav)]
        ref_wav = os.path.join(ref_dirname, os.path.basename(deg_wav))
        ref, ref_sr = sf.read(ref_wav)
        deg, deg_sr = sf.read(deg_wav)

        min_len = min(len(ref), len(deg))
        ref, deg = ref[:min_len], deg[:min_len]

        try:
            cur_stoi_score = stoi(ref, deg, fs_sig=ref_sr, extended=False)
            cur_estoi_score = stoi(ref, deg, fs_sig=deg_sr, extended=True)

            stoi_dict[f"{os.path.basename(deg_wav)}"] = cur_stoi_score
            estoi_dict[f"{os.path.basename(deg_wav)}"] = cur_estoi_score
            cnt += 1
        except:
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute STOI/ESTOI measure.")

    parser.add_argument('--ref_dir', required=False, 
                        default=["/data4/liandong/datasets/LibriTTS/LibriTTS/dev-clean", "/data4/liandong/datasets/LibriTTS/LibriTTS/dev-other"],
                        help="Reference wav folder.")
    parser.add_argument('--deg_dir', required=False, 
                        default="/data4/liandong/PROJECTS/FreeV/File_Decodes/LibriTTS/joint_denoise_vocoder/vocoder/1.75M_steps",
                        help="Degraded wav folder.")
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
        stoi_dict = manager.dict()
        estoi_dict = manager.dict()
        processes = []
        for f in file_lists:
            p = mp.Process(target=cal_stoi_wrapper, args=(ref_full_path_dict, f, stoi_dict, estoi_dict))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()
        
        # convert to standard list
        stoi_dict, estoi_dict = dict(stoi_dict), dict(estoi_dict)

        mean_stoi = np.mean(np.array([v for v in stoi_dict.values()]))
        std_stoi = np.std(np.array([v for v in stoi_dict.values()]))
        mean_estoi = np.mean(np.array([v for v in estoi_dict.values()]))
        std_estoi = np.std(np.array([v for v in estoi_dict.values()]))
        print("STOI score: mean->{:4f}, std->{:4f}.".format(mean_stoi, std_stoi))
        print("ESTOI score: mean->{:4f}, std->{:4f}.".format(mean_estoi, std_estoi))
