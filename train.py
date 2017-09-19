#!/usr/bin/env python
from __future__ import print_function

import os
import pickle
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
#from builtins import range
from tensorboard_logger import configure, log_value
from torch.autograd import Variable

import opts
from evaluate import evaluate
from misc.data import get_train_loader
from misc.models import *
from misc.utils import blockPrint, enablePrint
from misc.caption import *
sys.path.append('./coco-caption/')
from pycocotools.coco import COCO


def load_vocabulary(opt):
    # Load vocabulary
    vocab_pkl_path = os.path.join(opt.feat_dir, opt.ds + '_vocab.pkl')
    with open(vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    return vocab, vocab_size

def main(opt):
    opt, infos, iteration, epoch = utils.history_infos(opt)
    configure(opt.log_environment, flush_secs=10)
    vocab, vocab_size = load_vocabulary(opt)

    model, crit, optimizer, infos = build_models(opt, vocab, infos, {'split': 'train'})

    # Initialize data loader
    train_loader = get_train_loader(opt, opt.train_caption_pkl_path, opt.feature_h5_path, opt.batch_size)
    total_step = len(train_loader)

    # Prepare ground-truth for evaluation
    reference_json_path = '{0}.json'.format(opt.test_reference_txt_path)
    reference = COCO(reference_json_path)

    # Start training model
    best_meteor = 0
    loss_count = 0
    for epoch in range(opt.num_epochs):
        epsilon = max(0.6, opt.ss_factor / (opt.ss_factor + np.exp(epoch / opt.ss_factor)))
        print('epoch:%d\tepsilon:%.8f' % (epoch, epsilon))
        log_value('epsilon', epsilon, epoch)
        for i, (videos, captions, cap_lens, video_ids) in enumerate(train_loader, start=1):
            # Construct mini batch Variable
            videos = Variable(videos)
            targets = Variable(captions)
            if opt.use_cuda:
                videos = videos.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            outputs, _ = model(videos, targets, 1)
            # As we may can not sample a entire batch after one epoch
            # We need to re-calcuate the batch size
            bsz = len(captions)
            # Straighten the compressed (remove padding) output
            outputs = torch.cat([outputs[j][:cap_lens[j]] for j in range(bsz)], 0)
            outputs = outputs.view(-1, vocab_size)
            # Straighten the compressed (remove padding) target
            targets = torch.cat([targets[j][:cap_lens[j]] for j in range(bsz)], 0)
            targets = targets.view(-1)
            loss = crit(outputs, targets)
            log_value('loss', loss.data[0], epoch * total_step + i)
            loss_count += loss.data[0]
            loss.backward()
            optimizer.step()

            if i % 10 == 0 or bsz < opt.batch_size:
                loss_count /= 10 if bsz == opt.batch_size else i % 10
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %
                      (epoch, opt.num_epochs, i, total_step, loss_count,
                       np.exp(loss_count)))
                loss_count = 0
                tokens, _ = model.sample(videos)
                tokens = tokens.data[0].squeeze()
                we = model.decode_tokens(tokens)
                gt = model.decode_tokens(captions[0].squeeze())
                #print('[vid:%d]' % video_ids[0])
                #print('WE: %s\nGT: %s' % (we, gt))

        torch.save(model.state_dict(), opt.decoder_pth_path)
        torch.save(optimizer.state_dict(), opt.optimizer_pth_path)
        # Calcuate the performance on validation set
        blockPrint()
        model.eval()
        metrics = evaluate(opt, vocab, model, opt.test_range, opt.test_prediction_txt_path, reference)
        enablePrint()
        for k, v in metrics.items():
            log_value(k, v, epoch)
            print('%s: %.6f' % (k, v))
            if k == 'METEOR' and v > best_meteor:
                # Backup the best model on Val set
                shutil.copy2(opt.decoder_pth_path, opt.best_decoder_pth_path)
                shutil.copy2(opt.optimizer_pth_path, opt.best_optimizer_pth_path)
                best_meteor = v
        model.train()


'''
Main function: Start from here !!!
'''
opt = opts.parse_opt()
main(opt)
