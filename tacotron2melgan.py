import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""

cwd = os.getcwd()

import sys

sys.path.append(cwd)

import wave
from datetime import datetime

import numpy as np
import tensorflow as tf
from tacotron.datasets import audio
from tacotron.utils.infolog import log
from librosa import effects
from tacotron.models import create_model
from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence
import os
from tacotron_hparams import hparams
import shutil
import hashlib
import time
from tacotron.pinyin.parse_text_to_pyin import get_pyin


def padding_targets(target, r, padding_value):
    lens = target.shape[0]
    if lens % r == 0:
        return target
    else:
        target = np.pad(target, [(0, r - lens % r), (0, 0)], mode='constant', constant_values=padding_value)
        return target


class Synthesizer:
    def load(self, checkpoint_path, hparams, gta=False, model_name='Tacotron'):
        log('Constructing model: %s' % model_name)
        # Force the batch size to be known in order to use attention masking in batch synthesis
        inputs = tf.placeholder(tf.int32, (1, None), name='inputs')
        input_lengths = tf.placeholder(tf.int32, (1), name='input_lengths')

        targets = None  # tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
        target_lengths = None  # tf.placeholder(tf.int32, (1), name='target_length')
        # gta = True

        with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams)
            self.model.initialize(inputs=inputs, input_lengths=input_lengths)
            # mel_targets=targets,  targets_lengths=target_lengths, gta=gta, is_evaluating=True)

            self.mel_outputs = self.model.mel_outputs
            self.alignments = self.model.alignments
            if hparams.predict_linear:
                self.linear_outputs = self.model.linear_outputs
            self.stop_token_prediction = self.model.stop_token_prediction

        self._hparams = hparams

        self.inputs = inputs
        self.input_lengths = input_lengths
        # self.targets = targets
        # self.target_lengths = target_lengths

        log('Loading checkpoint: %s' % checkpoint_path)
        # Memory allocation on the GPUs as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, text):
        hparams = self._hparams

        T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (
        0, hparams.max_abs_value)

        # pyin, text = get_pyin(text)
        print(text.split(' '))

        inputs = [np.asarray(text_to_sequence(text.split(' ')))]
        print(inputs)
        input_lengths = [len(inputs[0])]

        feed_dict = {
            self.inputs: np.asarray(inputs, dtype=np.int32),
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
        }

        mels, alignments, stop_tokens = self.session.run([self.mel_outputs,
                                                          self.alignments, self.stop_token_prediction],
                                                         feed_dict=feed_dict)

        mel = mels[0]
        alignment = alignments[0]

        print('pred_mel.shape', mel.shape)
        stop_token = np.round(stop_tokens[0]).tolist()
        target_length = stop_token.index(1) if 1 in stop_token else len(stop_token)

        mel = mel[:target_length, :]
        mel = np.clip(mel, T2_output_range[0], T2_output_range[1])
        # 保存合成的音频
        out_dir = os.path.join(cwd, 'tacotron_inference_output')
        # if os.path.exists(out_dir):
        #    shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        strftime = time.strftime('%Y%m%d-%H%M%S')  # 给音频加入时间
        # wav_path = os.path.join(out_dir, 'step-{}-{}-wav-from-mel.wav'.format(step, hparams.dataset))
        wav_path = os.path.join(out_dir, 'step-{}-{}-wav-from-mel.wav'.format(strftime, hparams.dataset))
        wav = audio.inv_mel_spectrogram(mel.T, hparams)
        audio.save_wav(wav, wav_path, sr=hparams.sample_rate)

        wav_output, sr_output = librosa.core.load(wav_path, sr=None)

        return wav_output, sr_output
"""
melgan_inference
"""
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(Path(__name__).stem)

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--folder", type=Path, default=Path('tacotron_inference_output'),
                        help='输入音频文件的目录路径')
    parser.add_argument("-o", "--save_path", type=Path, default=Path("results"),
                        help='输出生成语音的目录路径')
    parser.add_argument("-m", "--load_path", type=Path,
                        default=Path("models/melgan_multi_speaker.pt"),
                        help='模型路径')
    parser.add_argument("--args_path", type=str, default='',
                        help='设置模型参数的文件')
    parser.add_argument("--mode", type=str, default='default',
                        help='模型模式')
    parser.add_argument("--n_samples", type=int, default=10,
                        help='需要实验多少个音频')
    parser.add_argument("--save_model_path", type=str, default='',
                        help='保存模型为可以直接torch.load的格式')
    parser.add_argument("--cuda", type=str, default='-1',
                        help='设置CUDA_VISIBLE_DEVICES')
    # 合成器的参数解析
    parser.add_argument('--text', default='', help='text to synthesis.')
    args_origin = parser.parse_args()

    return args_origin

#
# args = parse_args()
#
# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

import traceback
import numpy as np

import librosa
import torch
import time
from tqdm import tqdm
from scipy.io import wavfile

from melgan.inference import MelVocoder, get_default_device, save_model

_device = get_default_device()


if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    vocoder = MelVocoder(args.load_path, github=args.mode == 'default', args_path=args.args_path,
                         device=_device, mode=args.mode)
    if args.save_model_path:
        save_model(vocoder, args.save_model_path)
    args.save_path.mkdir(exist_ok=True, parents=True)

   # 合成器的参数解析
   #  parser = argparse.ArgumentParser()
    # parser.add_argument('--text', default='', help='text to synthesis.')


    past = time.time()

    synth = Synthesizer()

    ckpt_path = f'logs-Tacotron-2/{hparams.dataset}/taco_pretrained'  # finetune(D8)
    # ckpt_path = 'logs-Tacotron-2/taco_pretrained' # pretrained_tacotron
    checkpoint_path = tf.train.get_checkpoint_state(ckpt_path).model_checkpoint_path

    synth.load(checkpoint_path, hparams)
    print('succeed in loading checkpoint')

    # text = '空气质量良。'
    text = '这是一段测试语音哈哈哈。'

    # 解析文字
    text = args.text if args.text != '' else text
    pyin, text = get_pyin(text)

    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    idx = m.hexdigest()
    step = checkpoint_path.split('/')[-1].split('-')[-1].strip()
    # 阶段一：Tacotron V2将文字合成了纯信号处理还原的语音

    wav, sr = synth.synthesize(pyin)

    # 使用MelGAN解码器第一步，音频转mel
    mel = vocoder(torch.from_numpy(wav[None]))

    recons = vocoder.inverse(mel.to(_device)).squeeze().cpu().numpy()

    # 音频保存格式的设置
    strftime = time.strftime('%Y%m%d-%H%M%S')  # 给音频加入时间
    outdir = Path(args.save_path)  # 设置音频保存的路径，默认从参数解析器里面选
    # .joinpath(f'{args.load_path.stem}_{args.mode}')
    outdir.mkdir(exist_ok=True, parents=True)  # 假如没有路径，自己创建？
    filename = str(outdir.joinpath(f'audio_{strftime}_{hparams.dataset}_syn.wav'))  # 为合成的音频创建路径
    wavfile.write(filename=filename, rate=sr, data=recons)






