import os
import random
import glob
import torch
import torch.utils.data
import soundfile as sf
import numpy as np
from librosa.filters import mel as librosa_mel_fn
import librosa
import pyloudnorm as pyln
from typing import List


def load_wav(full_path, sample_rate):
    data, orig_sr = sf.read(full_path)
    if orig_sr != sample_rate:
        data = librosa.core.resample(data, orig_sr=orig_sr, target_sr=sample_rate)
    return data


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_window = {}
inv_mel_window = {}


def param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device):
    return f"{sampling_rate}-{n_fft}-{num_mels}-{fmin}-{fmax}-{win_size}-{device}"


def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=True,
    in_dataset=False,
):
    global mel_window
    device = torch.device("cpu") if in_dataset else y.device
    ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if ps in mel_window:
        mel_basis, hann_window = mel_window[ps]
        # print(mel_basis, hann_window)
        # mel_basis, hann_window = mel_basis.to(y.device), hann_window.to(y.device)
    else:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis = torch.from_numpy(mel).float().to(device)
        hann_window = torch.hann_window(win_size).to(device)
        mel_window[ps] = (mel_basis.clone(), hann_window.clone())

    spec = torch.stft(
        y.to(device),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window.to(device),
        center=True,
        return_complex=True,
    )

    spec = mel_basis.to(device) @ spec.abs()
    spec = spectral_normalize_torch(spec)

    return spec  # [batch_size,n_fft/2+1,frames]


def inverse_mel(
    mel,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    in_dataset=False,
):
    global inv_mel_window, mel_window
    device = torch.device("cpu") if in_dataset else mel.device
    ps = param_string(sampling_rate, n_fft, num_mels, fmin, fmax, win_size, device)
    if ps in inv_mel_window:
        inv_basis = inv_mel_window[ps]
    else:
        if ps in mel_window:
            mel_basis, _ = mel_window[ps]
        else:
            mel_np = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            mel_basis = torch.from_numpy(mel_np).float().to(device)
            hann_window = torch.hann_window(win_size).to(device)
            mel_window[ps] = (mel_basis.clone(), hann_window.clone())
        inv_basis = mel_basis.pinverse()
        inv_mel_window[ps] = inv_basis.clone()
    return inv_basis.to(device) @ spectral_de_normalize_torch(mel.to(device))


def amp_pha_specturm(y, n_fft, hop_size, win_size):
    hann_window = torch.hann_window(win_size).to(y.device)

    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=True,
        return_complex=True,
    )  # [batch_size, n_fft//2+1, frames, 2]

    log_amplitude = torch.log(
        stft_spec.abs() + 1e-5
    )  # [batch_size, n_fft//2+1, frames]
    phase = torch.atan2(stft_spec.imag, stft_spec.real)  # [batch_size, n_fft//2+1, frames]

    return log_amplitude, phase, stft_spec.real, stft_spec.imag


def get_dataset_filelist(input_training_wav_list, input_validation_wav_list, raw_wavfile_path, input_noise_wav_list):
    noise_all_files = []
    lines = open(input_noise_wav_list, 'r').readlines()
    for l in lines:
        cur_filename = l.strip()
        noise_all_files.append(cur_filename)

    noise_num = len(noise_all_files)
    # 9: 1 division in the noise clips by default
    training_noise_files, validation_noise_files = noise_all_files[:int(0.9 * noise_num)], \
                                                   noise_all_files[int(0.9 * noise_num):]
    actual_all_files = glob.glob(f'{raw_wavfile_path}/*.wav') + \
                       glob.glob(f'{raw_wavfile_path}/*/*.wav') + \
                       glob.glob(f'{raw_wavfile_path}/*/*/*.wav') + \
                       glob.glob(f'{raw_wavfile_path}/*/*/*/*.wav')
    training_files = []
    lines = open(input_training_wav_list, 'r').readlines()
    for l in lines:
        cur_filename = l.strip().split('|')[0]
        cur_wavfile_path = os.path.join(raw_wavfile_path, f'{cur_filename}.wav')
        if cur_wavfile_path in actual_all_files:
            training_files.append(cur_wavfile_path)

    validation_files = []
    lines = open(input_validation_wav_list, 'r').readlines()
    for l in lines:
        cur_filename = l.strip().split('|')[0]
        cur_wavfile_path = os.path.join(raw_wavfile_path, f'{cur_filename}.wav')
        if cur_wavfile_path in actual_all_files:
            validation_files.append(cur_wavfile_path)

    return training_files, validation_files, training_noise_files, validation_noise_files


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        noise_files,
        snr_range,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        batch_size,
        fmin,
        fmax,
        meloss,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        task_dict=None,
    ):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.noise_files = noise_files
        self.snr_range = snr_range
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.meloss = meloss
        self.rng = random.Random(1234)
        self.meter = pyln.Meter(self.sampling_rate)
        self.task_dict = task_dict

    def __getitem__(self, index):
        # determine task: denoise/vocoder
        if isinstance(self.task_dict, List):
            if len(self.task_dict) == 2:
                task_id = self.rng.choices([0, 1], weights=[0.5, 0.5], k=1)[0]
                task_type = self.task_dict[task_id]
            elif len(self.task_dict) == 1:
                task_type = self.task_dict[0]
        elif isinstance(self.task_dict, str):
            task_type = self.task_dict
        else:
            raise RuntimeError('Only list and str are supported! Please check it carefully.')

        inpt_list, log_amplitude_list, phase_list, rea_list, imag_list, audio_list, melloss1_list = [], [], [], [], [], [], []
        for cur_index in range(self.batch_size):
            idx = index * self.batch_size + cur_index
            audio_filename = self.audio_files[idx]
            if task_type == 'denoise':  # denoise task
                noise_idx = random.choice(list(range(len(self.noise_files))))
                noise_filename = self.noise_files[noise_idx]
                if self._cache_ref_count == 0:
                    audio = load_wav(audio_filename, self.sampling_rate)
                    noise = load_wav(noise_filename, self.sampling_rate)
                    noise = 100 * noise # pre-amplify to decrease the risk of inf/nan 
                    self.cached_wav = audio
                    self._cache_ref_count = self.n_cache_reuse
                else:
                    audio = self.cached_wav
                    self._cache_ref_count -= 1

                if self.split:
                    if len(audio) >= self.segment_size:
                        max_audio_start = len(audio) - self.segment_size
                        audio_start = random.randint(0, max_audio_start)
                        audio = audio[audio_start: audio_start + self.segment_size]  # (T)
                    else:
                        nrep = int(np.ceil(self.segment_size / len(audio)))
                        audio = np.tile(audio, nrep)[:self.segment_size]

                    if len(noise) >= self.segment_size:
                        while True:
                            noise_start = random.randint(0, len(noise) - self.segment_size)
                            noise_ = noise[noise_start: noise_start + self.segment_size]  # (T)
                            if (noise_ ** 2.0).sum() > 1e-2:
                                break
                        noise = noise_
                    else:
                        nrep = int(np.ceil(self.segment_size / len(noise)))
                        noise = np.tile(noise, nrep)[:self.segment_size]
                        if (noise ** 2.0).sum() <= 1e-2:
                            noise = noise + 0.1 * np.random.randn(*noise.shape)

                if self.batch_size == 1:  # only->validation
                    if len(noise) >= len(audio):
                        while True:
                            noise_start = random.randint(0, len(noise) - len(audio))
                            noise_ = noise[noise_start: noise_start + len(audio)]  # (T)
                            if (noise_ ** 2.0).sum() > 1e-2:
                                break
                        noise = noise_
                    else:
                        nrep = int(np.ceil(len(audio) / len(noise)))
                        noise = np.tile(noise, nrep)[:len(audio)]

                snr_dB = np.round(np.random.uniform(self.snr_range[0], self.snr_range[1]), decimals=1)
                loudness_audio = self.meter.integrated_loudness(audio)
                loudness_noise = self.meter.integrated_loudness(noise)
                target_loudness = loudness_audio - snr_dB
                delta_loudness = target_loudness - loudness_noise
                gain = np.power(10.0, delta_loudness / 20.0)
                # 如果gain为inf,表明loudness_noise过小，即noise整体为0
                if np.isinf(np.array(gain)) or np.isnan(np.array(gain)):
                    gain = 1.0
                noise_scaled = gain * noise
                inpt = audio + noise_scaled

                # adjust scale to avoid clipping effect
                while np.max(np.abs(inpt)) >= 1.0:
                    max_scale = np.random.uniform(0.3, 0.9)
                    c = max_scale / (np.max(np.abs(inpt)) + 1e-5)

                    inpt, audio = inpt * c, audio * c
                    # snr_dB += 1
                    # target_loudness = loudness_audio - snr_dB
                    # delta_loudness = target_loudness - loudness_noise
                    # gain = np.power(10.0, delta_loudness / 20.0)
                    # noise_scaled = gain * noise
                    # inpt = audio + noise_scaled
                #
                inpt = torch.FloatTensor(inpt.astype('float32')).unsqueeze(0)  # (1, T)
                audio = torch.FloatTensor(audio.astype('float32')).unsqueeze(0)  # (1, T)

            elif task_type == 'vocoder':
                if self._cache_ref_count == 0:
                    audio = load_wav(audio_filename, self.sampling_rate)
                    self.cached_wav = audio
                    self._cache_ref_count = self.n_cache_reuse
                else:
                    audio = self.cached_wav
                    self._cache_ref_count -= 1

                if self.split:
                    if len(audio) >= self.segment_size:
                        max_audio_start = len(audio) - self.segment_size
                        audio_start = random.randint(0, max_audio_start)
                        audio = audio[audio_start: audio_start + self.segment_size]  # (T)
                    else:
                        nrep = int(np.ceil(self.segment_size / len(audio)))
                        audio = np.tile(audio, nrep)[:self.segment_size]
                #
                inpt = torch.FloatTensor(audio.astype('float32')).unsqueeze(0)  # (1, T)
                audio = torch.FloatTensor(audio.astype('float32')).unsqueeze(0)  # (1, T)

            # extract features
            if task_type == 'denoise':
                inpt_log_amplitude = amp_pha_specturm(inpt, self.n_fft, self.hop_size, self.win_size)[0]
                inpt_list.append(inpt_log_amplitude)
                log_amplitude, phase, rea, imag = amp_pha_specturm(audio, self.n_fft, self.hop_size, self.win_size)
                log_amplitude_list.append(log_amplitude)
                phase_list.append(phase)
                rea_list.append(rea)
                imag_list.append(imag)
            elif task_type == 'vocoder':
                # mel->inv_mel: spectrum corrupted
                mel = mel_spectrogram(inpt, 
                                      self.n_fft,
                                      self.num_mels,
                                      self.sampling_rate,
                                      self.hop_size,
                                      self.win_size,
                                      self.fmin,
                                      self.fmax,
                                      center=True,
                                      in_dataset=True)
                inv_mel = inverse_mel(
                                      mel,
                                      self.n_fft,
                                      self.num_mels,
                                      self.sampling_rate,
                                      self.hop_size,
                                      self.win_size,
                                      self.fmin,
                                      self.fmax,
                                      in_dataset=True).abs().clamp_min(1e-5)
                inv_mel = inv_mel.log()
                inpt_list.append(inv_mel)
                log_amplitude, phase, rea, imag = amp_pha_specturm(audio, self.n_fft, self.hop_size, self.win_size)
                log_amplitude_list.append(log_amplitude)
                phase_list.append(phase)
                rea_list.append(rea)
                imag_list.append(imag)

            meloss1 = mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.meloss,
                center=True,
                in_dataset=True,  # True->CPU else device
            )
            melloss1_list.append(meloss1)
            audio_list.append(audio)

        # pad_sequence
        inpts = torch.cat(inpt_list, dim=0)  # (B, F, T)
        log_amplitudes = torch.cat(log_amplitude_list, dim=0)  # (B, F, T)
        phases = torch.cat(phase_list, dim=0)  # (B, F, T)
        reals = torch.cat(rea_list, dim=0)  # (B, F, T)
        imags = torch.cat(imag_list, dim=0)  # (B, F, T)
        audios = torch.cat(audio_list, dim=0)  # (B, L)
        melloss1s = torch.cat(melloss1_list, dim=0)  # (B, F, T)

        return inpts, log_amplitudes, phases, reals, imags, audios, melloss1s

    def __len__(self):

        return len(self.audio_files) // self.batch_size
