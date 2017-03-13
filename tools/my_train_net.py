#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import os

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

if __name__ == '__main__':

    cfg_file = os.path.join(cfg.ROOT_DIR,'experiments/cfgs/faster_rcnn_end2end.yml')
    set_cfgs = None
    imdb_name = 'voc_2007_trainval'
    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)


    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_cpu()
    #caffe.set_device(0)

    imdb, roidb = combined_roidb(imdb_name)
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = 'output/my_output'
    print 'Output will be saved to `{:s}`'.format(output_dir)

    solver = os.path.join(cfg.ROOT_DIR,'models/pascal_voc/ZF/my_net/solver.prototxt')
    train_net(solver, roidb, output_dir,
              pretrained_model = os.path.join(cfg.ROOT_DIR, 'data/imagenet_models/ZF.v2.caffemodel'),
              max_iters = 20000)
