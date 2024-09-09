import argparse
from typing import *
import numpy as np
import os
import soundfile as sf
import librosa as lib
from pesq import pesq as pesq
from tqdm import tqdm
from glob import glob
import multiprocessing as mp


def cal_pesq_wrapper(ref_dir_dict, deg_files, sr=None, pesq_dict=None):

    cnt = 0
    for deg_wav in tqdm(deg_files):
        ref_dirname = ref_dir_dict[os.path.basename(deg_wav)]
        ref_wav = os.path.join(ref_dirname, os.path.basename(deg_wav))
        ref, ref_sr = sf.read(ref_wav)
        deg, deg_sr = sf.read(deg_wav)
        if sr is not None:
            if ref_sr != sr:
                ref = lib.core.resample(ref, orig_sr=ref_sr, target_sr=sr)
            if deg_sr != sr:
                deg = lib.core.resample(deg, orig_sr=ref_sr, target_sr=sr)

        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        try:
            cur_wb_score = pesq(16000, ref, deg, 'wb')
            pesq_dict[f"{os.path.basename(deg_wav)}"] = cur_wb_score
            cnt += 1
        except:
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute PESQ measure.")

    parser.add_argument('--ref_dir', required=False, 
                        default="/data4/liandong/datasets/LJSpeech-1.1",
                        help="Reference wav folder.")
    parser.add_argument('--deg_dir', required=False, 
                        default="/data4/liandong/PROJECTS/FreeV/File_Decodes/LJSpeech/BSRNN_feat128_mask",
                        help="Degraded wav folder.")
    parser.add_argument('--sr', default=16000, required=False, help="Target sampling rate, 16000 by default."
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
        pesq_dict = manager.dict()
        processes = []
        for f in file_lists:
            p = mp.Process(target=cal_pesq_wrapper, args=(ref_full_path_dict, f, args.sr, pesq_dict))
            p.start()
            processes.append(p)
        
        # wait for all process
        for p in processes:
            p.join()
        
        # convert to standard list
        pesq_dict = dict(pesq_dict)

        mean_pesq = np.mean(np.array([v for v in pesq_dict.values()]))
        std_pesq = np.std(np.array([v for v in pesq_dict.values()]))
        print("PESQ score: mean->{:4f}, std->{:4f}.".format(mean_pesq, std_pesq))
