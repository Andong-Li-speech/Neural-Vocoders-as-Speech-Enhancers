import argparse
import warnings
import numpy as np
import torch 
torch.autograd.set_detect_anomaly(True)
warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
import time
from tqdm import tqdm
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# from ptflops import get_model_complexity_info
from dataset import Dataset, mel_spectrogram, get_dataset_filelist
from Models.models import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    hinge_generator_loss,
    hinge_discriminator_loss,
    ls_generator_loss,
    ls_discriminator_loss
)

from Models import (HiFiGAN,
                    iSTFTNet,
                    HDDemucas, 
                    ConvTasNet,
                    )

from utils import (
    AttrDict,
    build_env,
    plot_spectrogram,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    remove_older_checkpoint
)

def train(h):
    torch.cuda.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    torch.manual_seed(h.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:{:d}".format(0))

    generator = eval(h.model_name)(h).to(device)
    # get_model_complexity_info(generator, (80, 86))
    mpd = MultiPeriodDiscriminator(mpd_reshapes=h.mpd_reshapes).to(device)
    print("Using MultiPeriodDiscriminator")
    mrd = MultiScaleDiscriminator().to(device)
    print("Using MultiScaleDiscriminator.")

    print(generator)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_g = scan_checkpoint(h.checkpoint_path, "g_")
        cp_do = scan_checkpoint(h.checkpoint_path, "do_")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    training_filelist, validation_filelist = get_dataset_filelist(
        h.input_training_wav_list, h.input_validation_wav_list, h.raw_wavfile_path
    )

    trainset = Dataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        n_cache_reuse=0,
        shuffle=True,
        device=device,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=True,
        sampler=None,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    validset = Dataset(
        validation_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        False,
        False,
        n_cache_reuse=0,
        device=device,
    )
    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )

    sw = SummaryWriter(os.path.join(h.checkpoint_path, "logs"))

    generator.train()
    mpd.train()
    mrd.train()

    for epoch in range(max(0, last_epoch), h.training_epochs):
        start = time.time()
        print("Epoch: {}".format(epoch + 1))

        for i, batch in enumerate(tqdm(train_loader)):
            start_b = time.time()
            x, logamp, pha, rea, imag, y, meloss = map(
                lambda x: x.to(device, non_blocking=True), batch
            )

            # for waveform-domain methods...
            y_g = generator(x)
            # 对齐长度
            y_min = np.min([y_g.shape[-1], y.shape[-1]])
            y_g, y = y_g[..., :y_min], y[..., :y_min]
            y_g_mel = mel_spectrogram(
                y_g,
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.sampling_rate / 2,
            )
            optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g.detach())
            if h.model_name in ['HiFiGAN', 'iSTFTNet']:
                # for HiFiGAN, and iSTFTNet, we follow the original loss-setting (ls)
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = ls_discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )
            else:
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = hinge_discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g.detach())
            if h.model_name in ['HiFiGAN', 'iSTFTNet']:
                # for HiFiGAN, and iSTFTNet, we follow the original loss-setting (ls)
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = ls_discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )
            else:
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = hinge_discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

            L_D = loss_disc_s + loss_disc_f
            L_D.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_g)
            y_ds_r, y_ds_g, fmap_s_r, fmap_s_g = mrd(y, y_g)

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            if h.model_name in ['HiFiGAN', 'iSTFTNet']:
                loss_gen_f, losses_gen_f = ls_generator_loss(y_df_g)
                loss_gen_s, losses_gen_s = ls_generator_loss(y_ds_g)
            else:
                loss_gen_f, losses_gen_f = hinge_generator_loss(y_df_g)
                loss_gen_s, losses_gen_s = hinge_generator_loss(y_ds_g)
            L_GAN_G = loss_gen_s + loss_gen_f
            L_FM = loss_fm_s + loss_fm_f
            L_Mel = F.l1_loss(meloss, y_g_mel) * 45

            L_G = L_GAN_G + L_FM + L_Mel

            L_G.backward()
            optim_g.step()

            # STDOUT logging
            if steps % h.stdout_interval == 0:
                with torch.no_grad():
                    Mel_error = F.l1_loss(x, y_g_mel).item()
                # print(
                #     "Steps : {:d}, Gen Loss Total : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, s/b : {:4.3f}".format(
                #         steps,
                #         L_G,
                #         Mel_error,
                #         time.time() - start_b,
                #     )
                # )

            # checkpointing
            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {"generator": generator.state_dict()})
                remove_older_checkpoint(checkpoint_path, pre='g', max_to_keep=h.max_to_keep)
                checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(
                    checkpoint_path,
                    {
                        "mpd": mpd.state_dict(),
                        "mrd": mrd.state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    },
                )
                remove_older_checkpoint(checkpoint_path, pre='d', max_to_keep=h.max_to_keep)

            # Tensorboard summary logging
            if steps % h.summary_interval == 0:
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", Mel_error, steps)

            # Validation
            if steps % h.validation_interval == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_Mel_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, logamp, pha, rea, imag, y, meloss = map(
                            lambda x: x.to(device, non_blocking=True), batch
                        )
                        y_g = generator(x.to(device))

                        # 对齐长度
                        y_min = np.min([y_g.shape[-1], y.shape[-1]])
                        y_g, y = y_g[..., :y_min], y[..., :y_min]

                        y_g_mel = mel_spectrogram(
                            y_g,
                            h.n_fft,
                            h.num_mels,
                            h.sampling_rate,
                            h.hop_size,
                            h.win_size,
                            h.fmin,
                            h.sampling_rate / 2,
                        )
                        val_Mel_err_tot = val_Mel_err_tot + F.l1_loss(meloss, y_g_mel).item()

                        if j <= 4:
                            if steps == 0:
                                sw.add_audio(
                                    "gt/y_{}".format(j), y[0], steps, h.sampling_rate
                                )
                                sw.add_figure(
                                    "gt/y_spec_{}".format(j),
                                    plot_spectrogram(x[0].cpu()),
                                    steps,
                                )

                            sw.add_audio(
                                "generated/y_g_{}".format(j),
                                y_g[0],
                                steps,
                                h.sampling_rate,
                            )
                            y_g_spec = mel_spectrogram(
                                y_g,
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax,
                            )
                            sw.add_figure(
                                "generated/y_g_spec_{}".format(j),
                                plot_spectrogram(y_g_spec.squeeze(0).cpu().numpy()),
                                steps,
                            )
                    val_Mel_err = val_Mel_err_tot / (j + 1)
                    sw.add_scalar("Validation/Mel_Spectrogram_loss", val_Mel_err, steps)

                generator.train()

            steps = steps + 1

        scheduler_g.step()
        scheduler_d.step()

        print(
            "Time taken for epoch {} is {} sec\n".format(
                epoch + 1, int(time.time() - start)
            )
        )

def main():
    print("Initializing Training Process..")
    parse = argparse.ArgumentParser('Vocoder configs.')
    parse.add_argument('--cfg_filename', type=str, required=True, default="./cfgs/istftnet_config.json", help='Json for configurations.')
    args = parse.parse_args()

    config_file = args.cfg_filename
    try:
        config_file_list = os.path.split(config_file)
        if len(config_file_list) > 1:
            config_filename = config_file_list[-1]
    except:
        config_filename = config_file

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, config_filename, h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    train(h)

if __name__ == "__main__":
    main()
