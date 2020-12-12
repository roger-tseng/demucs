import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from .vc_model import AE
#from vc_utils import cc
from functools import reduce
import json
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from scipy.io.wavfile import write
import random
#from preprocess.tacotron.utils import melspectrogram2wav
#from preprocess.tacotron.utils import get_spectrograms
import librosa 
#from torch.multiprocessing import set_start_method
#set_start_method('spawn')

def cc(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)

class Hyperparams:
    '''
    Hyperparameters
    By kyubyong park. kbpark.linguist@gmail.com.
    https://www.github.com/kyubyong/dc_tts
    '''
    
    # pipeline
    prepro = False  # if True, run `python prepro.py` first before running `python train.py`.

    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence

    # data
    data = "/data/private/voice/LJSpeech-1.0"
    # data = "/data/private/voice/nick"
    test_data = 'harvard_sentences.txt'
    max_duration = 10.0
    top_db = 15

    # signal processing
    #sr = 24000 # Sample rate.
    n_fft = 4096 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    #hop_length = int(sr*frame_shift) # samples.
    #win_length = int(sr*frame_length) # samples.
    n_mels = 512 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 100 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 256 # alias = E
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5

    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "logdir/01"
    sampledir = 'samples'
    batch_size = 32
    
    def __init__(self, sr):
        self.sr = sr
        self.hop_length = int(sr*0.0125)
        self.win_length = int(sr*0.05)

def get_spectrograms_direct(y, sr):
    '''
    Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed

    By kyubyong park. kbpark.linguist@gmail.com.
    https://www.github.com/kyubyong/dc_tts

    '''
    # num = np.random.randn()
    # if num < .2:
    #     y, sr = librosa.load(fpath, sr=hp.sr)
    # else:
    #     if num < .4:
    #         tempo = 1.1
    #     elif num < .6:
    #         tempo = 1.2
    #     elif num < .8:
    #         tempo = 0.9
    #     else:
    #         tempo = 0.8
    #     cmd = "ffmpeg -i {} -y ar {} -hide_banner -loglevel panic -ac 1 -filter:a atempo={} -vn temp.wav".format(fpath, hp.sr, tempo)
    #     os.system(cmd)
    #     y, sr = librosa.load('temp.wav', sr=hp.sr)

    # Loading sound file
    hp = Hyperparams(sr)
    #y, sr = librosa.load(fpath, sr=hp.sr)
    #print("y shape", y.shape)
    # Trimming
    #y, _ = librosa.effects.trim(y, top_db=hp.top_db)
    #print("y shape after trim", y.shape)
    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])
    
    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)
    
    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)
    
    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)
    #print("mel specto shape: ", mel.shape)
    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

class Inferencer(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        #print(config)
        # args store other information
        self.args = args
        #print(self.args)

        # init the model with config
        self.build_model()

        # load model
        self.load_model()

        with open(self.args.attr, 'rb') as f:
            self.attr = pickle.load(f)

    def load_model(self):
        #print(f'Load model from {self.args.model}')
        self.model.load_state_dict(torch.load(f'{self.args.model}'))
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        #print(self.model)
        self.model.eval()
        return

    def utt_make_frames(self, x):
        frame_size = self.config['data_loader']['frame_size']
        remains = x.size(0) % frame_size 
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out

    def inference_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)
        x_cond = self.utt_make_frames(x_cond)
        dec = self.model.inference(x, x_cond)
        dec = dec.transpose(1, 2).squeeze(0)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)
        wav_data = melspectrogram2wav(dec)
        return wav_data, dec

    def denormalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr['mean'], self.attr['std']
        print("m: ", m.shape)
        print("First 10 elements of m: ", m[:10])
        print("s: ", s.shape)
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        write(output_path, rate=self.args.sample_rate, data=wav_data)
        return

    def inference_from_path(self):
        src_mel, _ = get_spectrograms(self.args.source)
        tar_mel, _ = get_spectrograms(self.args.target)
        src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
        conv_wav, conv_mel = self.inference_one_utterance(src_mel, tar_mel)
        self.write_wav_to_file(conv_wav, self.args.output)
        return

    def infer_content_test(self):
        src_mel, _ = get_spectrograms(self.args.source)
        print("spectogram shape: ", src_mel.shape)
        src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        src_mel = self.utt_make_frames(src_mel)
        dec = self.model.inference(src_mel, src_mel)
        print("1st dec size is: ", dec.size())
        dec = dec.transpose(1, 2).squeeze(0)
        print("2nd dec size is: ", dec.size())
        dec = dec.detach().cpu().numpy()
        print("3rd dec shape is: ", dec.shape)
        dec = self.denormalize(dec)
        wav_data = melspectrogram2wav(dec)
        return wav_data, dec
    '''
    def infer_content_from_path(self):
        src_mel, _ = get_spectrograms(self.args.source)
        #src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        src_mel = torch.from_numpy(src_mel).cuda()
        #print("mel size: ", src_mel.size())
        x = self.utt_make_frames(src_mel)
        #print("x size: ", x.size())
        emb = self.model.get_content_embeddings(x)
        emb = emb.detach().cpu()
        #emb = self.denormalize(emb)
        emb = emb.squeeze(0)
        return emb
    '''
    def infer_content(self, streams, samplerate):
        src_mel, _ = get_spectrograms_direct(streams, samplerate)
        #src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        src_mel = torch.from_numpy(src_mel).cuda()
        print("mel size: ", src_mel.size())
        x = self.utt_make_frames(src_mel)
        #print("x size: ", x.size())
        emb = self.model.get_content_embeddings(x)
        emb = emb.detach().cpu()
        #emb = self.denormalize(emb)
        emb = emb.squeeze(0)
        return emb

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('-attr', '-a', help='attr file path')
    parser.add_argument('-config', '-c', help='config file path')
    parser.add_argument('-model', '-m', help='model path')
    parser.add_argument('-source', '-s', help='source wav path')
    parser.add_argument('-sample_rate', '-sr', help='sample rate', default=24000, type=int)
    args = parser.parse_args()
    
    # load config file 
    #args = Namespace(attr='./attr.pkl', config='./config.yaml', model='vctk_model.ckpt', sample_rate=24000, source='/home/itsmagicrt0612/musdb18/test/~/The Easton Ellises - Falcon 69_0.wav')
    with open(args.config) as f:
        config = yaml.load(f)
    inferencer = Inferencer(config=config, args=args)
    print()
    print("========================================================")
    print()
    print(f"using {inferencer.args.source}")
    print()
    print("content embedding size: ", inferencer.infer_content().size())
    #print("content embedding:\n", inferencer.infer_content())
    print()
    print("Note: no spectogram normalization currently")
    print()
    print("========================================================")
    print()

