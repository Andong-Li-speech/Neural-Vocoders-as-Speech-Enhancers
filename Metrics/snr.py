import argparse
import numpy as np
import glob
import os
import soundfile as sf
import librosa as lib
from tqdm import tqdm


def cal_snr_wrapper(ref_dir, deg_dir, sr):
    input_files = glob.glob(f"{deg_dir}/*.wav")

    snr_score_list = []
    cnt = 0
    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
        ref, ref_sr = sf.read(ref_wav)
        deg, deg_sr = sf.read(deg_wav)
        if sr is not None:
            if ref_sr != sr:
                ref = lib.core.resample(ref, orig_sr=ref_sr, target_sr=sr)
            if deg_sr != sr:
                deg = lib.core.resample(deg, orig_sr=ref_sr, target_sr=sr)

        min_len = min(len(ref), len(deg))
        ref, deg = ref[:min_len], deg[:min_len]
        ref_mean, deg_mean = np.mean(ref), np.mean(deg)
        ref, deg = ref - ref_mean, deg - deg_mean

        try:
            cur_snr = 10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - deg) ** 2)) + 1e-10)
            snr_score_list.append(cur_snr)
            cnt += 1
        except:
            pass
    
    mean_snr_score = np.mean(snr_score_list)
    std_snr_score = np.std(snr_score_list)

    return mean_snr_score, std_snr_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute SNR measure.")

    parser.add_argument('--ref_dir', required=False, 
                        default="/data4/liandong/datasets/LJSpeech-1.1/wavs", 
                        help="Reference wav folder.")
    parser.add_argument('--deg_dir', required=False, 
                        default="/data4/liandong/PROJECTS/FreeV/File_Decodes/LJSpeech/APNet2_ori_authors",
                        help="Degraded wav folder.")
    parser.add_argument('--sr', required=False,
                        type=int,
                        default=22050,
                        help="sampling rate.")

    args = parser.parse_args()

    mean_, std_ = cal_snr_wrapper(args.ref_dir, args.deg_dir, args.sr)
    print("SNR score: mean->{:4f}, std->{:4f}.".format(mean_, std_))
