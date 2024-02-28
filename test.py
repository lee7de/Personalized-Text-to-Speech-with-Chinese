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

        wav_path = os.path.join(out_dir, 'step-{}-{}-wav-from-mel.wav'.format(step, idx))
        wav = audio.inv_mel_spectrogram(mel.T, hparams)
        # audio.save_wav(wav, wav_path, sr=hparams.sample_rate)

        # pred_mel_path = os.path.join(out_dir, 'step-{}-{}-mel-pred.npy'.format(step, idx))
        # new_mel = np.clip((mel + T2_output_range[1]) / (2 * T2_output_range[1]), 0, 1)
        # # np.save(pred_mel_path, new_mel, allow_pickle=False)
        #
        # # pred_mel_path = os.path.join(out_dir, 'step-{}-{}-mel-pred.png'.format(step, idx))
        # # plot.plot_spectrogram(mel, pred_mel_path, title=datetime.now().strftime('%Y-%m-%d %H:%M'))
        #
        # # alignment_path = os.path.join(out_dir, 'step-{}-{}-align.npy'.format(step, idx))
        # # np.save(alignment_path, alignment, allow_pickle=False)
        # alignment_path = os.path.join(out_dir, 'step-{}-{}-align.png'.format(step, idx))
        # # plot.plot_alignment(alignment, alignment_path,
        # #     title=datetime.now().strftime('%Y-%m-%d %H:%M'), split_title=True, max_len=target_length)

        return wav


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='', help='text to synthesis.')
    args = parser.parse_args()

    past = time.time()

    synth = Synthesizer()

    ckpt_path = f'logs-Tacotron-2/{hparams.dataset}/taco_pretrained'  # finetune(D8)
    # ckpt_path = 'logs-Tacotron-2/taco_pretrained' # pretrained_tacotron
    checkpoint_path = tf.train.get_checkpoint_state(ckpt_path).model_checkpoint_path

    synth.load(checkpoint_path, hparams)
    print('succeed in loading checkpoint')

    out_dir = os.path.join(cwd, 'tacotron_inference_output')
    # if os.path.exists(out_dir):
    #    shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    text = '哈尔滨今天晴，十度到二十二度，南风三级，空气质量良。'

    text = args.text if args.text != '' else text
    pyin, text = get_pyin(text)

    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    idx = m.hexdigest()
    step = checkpoint_path.split('/')[-1].split('-')[-1].strip()

    strftime = time.strftime('%Y%m%d-%H%M%S')

    wav_path = os.path.join(out_dir, 'step-{}-{}-wav-from-mel.wav'.format(strftime, text))
    wav = synth.synthesize(pyin)
    audio.save_wav(wav, wav_path, sr=hparams.sample_rate)
    import librosa
    wav, sr = librosa.core.load(wav_path, sr=None)
    print(type(wav))


    print('last: {} seconds'.format(time.time() - past))

