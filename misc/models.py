import itertools
import os

import torch.optim as optim

from cnn.EncoderCNN import *
from lm.LangugeModel import *
from lm.variants.SSTA import *
from Criterion import *

def build_lm(opt, vocab):
    opt.n_gpus = getattr(opt, 'n_gpus', 1)

    if opt.model_name == 'SSTA':
        lm_model = SSTA(opt, vocab)
    else:
        lm_model = LanguageModel(opt)
    if opt.n_gpus>1:
        print('Construct multi-gpu model ...')
        model = nn.DataParallel(lm_model, device_ids=opt.gpus, dim=0)
    else:
        model = lm_model

    # check compatibility if training is continued from previously saved model
    if len(opt.start_from) != 0:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        lm_info_path = os.path.join(opt.start_from, os.path.basename(opt.start_from) + '.infos-best.pkl')
        lm_pth_path = os.path.join(opt.start_from, os.path.basename(opt.start_from) + '.model-best.pth')
        assert os.path.isfile(lm_info_path), "infos.pkl file does not exist in path %s" % opt.start_from
        model.load_state_dict(torch.load(lm_pth_path))
    if opt.use_cuda:
        model.cuda()
    else:
        raise AssertionError('Hey, get a GPU!!!')
    return model

def build_optimizer(opt, model, infos):
    model_parameters = itertools.ifilter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_parameters, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if len(opt.start_from) != 0:
        if os.path.isfile(os.path.join(opt.start_from, opt.model_id + '.optimizer.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, opt.model_id + '.optimizer.pth')))

    return optimizer

def build_models(opt, vocab, infos, model_kwargs):
    split = model_kwargs.get('split', 'train')
    model = build_lm(opt, vocab)
    crit = LanguageModelCriterion(opt)

    if split == 'train':
        model.train()  # Assure in training mode
        optimizer = build_optimizer(opt, model, infos)
    else:
        model.eval()  # Assure in testing or validation mode
        optimizer = None

    # Pring training parameter
    print('Learning rate: %.4f' % opt.learning_rate)
    print('Batch size: %d' % opt.batch_size)

    return model, crit, optimizer, infos
