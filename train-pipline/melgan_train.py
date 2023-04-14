#!usr/bin/env python
# -*- coding: utf-8 -*-
import os

# from utils.argutils import print_args
from melgan.train import train_melgan, parse_args

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

if __name__ == "__main__":
    try:
        from setproctitle import setproctitle

        setproctitle('melgan-train')
    except ImportError:
        pass

    # print_args(args, parser)
    train_melgan(args)
