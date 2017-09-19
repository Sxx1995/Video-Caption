'''
Generate predict results on the given dataset split, and calcuate the scores.
'''

from __future__ import absolute_import
from __future__ import unicode_literals

import pickle
import sys
import opts
import torch
from torch.autograd import Variable
from misc.data import get_eval_loader
from misc.models import *
from misc.utils import CocoResFormat

sys.path.append('./coco-caption/')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def measure(prediction_txt_path, reference):
    # Transform predicted results(txt format) to the format required by evaluation function
    prediction_json_path = prediction_txt_path.replace('txt', 'json')
    crf = CocoResFormat()
    crf.read_file(prediction_txt_path, True)
    crf.dump_json(prediction_json_path)
    # crf.res is the transformed predict results
    cocoRes = reference.loadRes(prediction_json_path)
    #cocoRes = reference.loadRes(crf.res)
    cocoEval = COCOEvalCap(reference, cocoRes)

    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))
    return cocoEval.eval


def evaluate(opt, vocab, decoder, eval_range, prediction_txt_path, reference):
    # Load test dataset
    eval_loader = get_eval_loader(opt, eval_range, opt.feature_h5_path)

    result = {}
    for i, (videos, video_ids) in enumerate(eval_loader):
        # Create mini batch Variable
        videos = Variable(videos)

        if opt.use_cuda:
            videos = videos.cuda()

        outputs, attens = decoder.sample(videos)
        #outputs = outputs.data.squeeze(2)
        outputs = outputs.data
        for (tokens, vid) in zip(outputs, video_ids):
            s = decoder.decode_tokens(tokens)
            result[vid] = s

    prediction_txt = open(prediction_txt_path, 'w')
    for vid, s in result.items():
        prediction_txt.write('%d\t%s\n' % (vid, s))  # Note that the video name of MSVD is start from 1

    prediction_txt.close()

    # calculate scores for the generated results
    metrics = measure(prediction_txt_path, reference)
    return metrics


if __name__ == '__main__':
    opt = opts.parse_opt()

    opt, infos, iteration, epoch = utils.history_infos(opt)
    with open(opt.vocab_pkl_path, 'rb') as f:
        vocab = pickle.load(f)

    # Load pre-trained model
    model, crit, _, infos = model(opt, vocab, infos)
    reference_json_path = '{0}.json'.format(opt.test_reference_txt_path)
    reference = COCO(reference_json_path)
    evaluate(opt, vocab, decoder, opt.test_range, opt.test_prediction_txt_path, reference)
