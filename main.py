# -*- coding: utf-8 -*-

# This project is for Roberta model.

import time
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import tqdm
from datetime import datetime
from load_data import load_data
from transformers import RobertaTokenizer, RobertaForMaskedLM
from torch.optim import AdamW
from parameter import parse_args
from util import savePredict
from dataset_processor import DatasetProcessor
from tools import calculate, get_batch
import random
from Model.model import MLP


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

if True:
    args = parse_args()
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    if not os.path.exists(args.outmodel):
        os.mkdir(args.outmodel)
    if not os.path.exists('./predict/'):
        os.mkdir('./predict/')
    if not os.path.exists(args.train_file):
        os.mkdir(args.train_file)
    if not os.path.exists(args.dev_file):
        os.mkdir(args.dev_file)
    if not os.path.exists(args.test_file):
        os.mkdir(args.test_file)
    t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
    args.log = args.log + f"mavenLGN_{args.graph_enhance}_{args.graph_dropout}_{args.t_lr}_{args.lr_decay}_" + t + '.txt'


    # 创建日志文件
    logging.basicConfig(format='%(message)s', level=logging.INFO,
                        filename=args.log,
                        filemode='w')
    logger = logging.getLogger(__name__)

    def printlog(message: object, printout: object = True):
        message = '{}: {}'.format(datetime.now(), message)
        if printout:
            print(message)
        logger.info(message)
    # load Roberta model
    printlog('Passed args:')
    printlog('log path: {}'.format(args.log))
    printlog('transformer model: {}'.format(args.model_name))

    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    setup_seed(args.seed)

if True:
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    encoder = RobertaForMaskedLM.from_pretrained(args.model_name).to(device)
#
# if True:
#     printlog('Loading data')
#     train_data, dev_data, test_data = load_data(args)
#     train_size = len(train_data)
#     dev_size = len(dev_data)
#     test_size = len(test_data)
#     print('Data loaded')
#
if True:
    printlog('Start processing the dataset using DatasetProcessor...')
    dataset_processor = DatasetProcessor(args, device, tokenizer, encoder)

    #
    # train_data, dev_data, test_data = dataset_processor.process_datasets(train_data, dev_data, test_data)
    #
    # data_sizes = dataset_processor.get_data_sizes()
    # vocab_info = dataset_processor.get_vocab_info()
    # conversion_tables = dataset_processor.get_conversion_tables()
    # to_add = conversion_tables['to_add']
    #
    # printlog(f"The dataset processing is complete!")
    # printlog(f"Train set size: {data_sizes['train']}")
    # printlog(f"Dev set size: {data_sizes['dev']}")
    # printlog(f"Test set size: {data_sizes['test']}")
    # printlog(f"Vocabulary list size: {vocab_info['original_vocab_size']} -> {vocab_info['current_vocab_size']}")
    # printlog(f"The number of added event tokens: {vocab_info['added_tokens_count']}")

    train_data, dev_data, test_data, to_add, tokenizer = dataset_processor.load_processed_data()
    train_size = len(train_data)
    dev_size = len(dev_data)
    test_size = len(test_data)
    args = dataset_processor.args
    printlog(f"tokenizer now has {len(tokenizer)} tokens")
# ---------- network ----------

net = MLP(args).to(device)
net.handler(to_add, tokenizer)
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.t_lr)

from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)

cross_entropy = nn.CrossEntropyLoss().to(device)

# save model and result
best_mrr, best_hit1, best_hit3, best_hit10, best_hit20, best_hit50 = 0, 0, 0, 0, 0, 0
dev_best_mrr, dev_best_hit1, dev_best_hit3, dev_best_hit10, dev_best_hit20, dev_best_hit50 = 0, 0, 0, 0, 0, 0
best_mrr_epoch, best_hit1_epoch, best_hit3_epoch, best_hit10_epoch, best_hit20_epoch, best_hit50_epoch = 0, 0, 0, 0, 0, 0
state = {}

best_epoch = 0

printlog(args)
printlog('Start training ...')


##################################  epoch  #################################
for epoch in range(args.num_epoch):
    args.model = args.outmodel + f"epoch{epoch}_mavenLGN_{args.graph_enhance}_{args.graph_dropout}_{args.t_lr}_{args.lr_decay}_" + t + '.pth'
    print('=' * 20)
    printlog('Epoch: {}'.format(epoch))
    torch.cuda.empty_cache()

    all_indices = torch.randperm(train_size).split(args.batch_size)
    loss_epoch = 0.0

    Mrr, Hit1, Hit3, Hit10, Hit20, Hit50 = [], [], [], [], [], []
    all_Mrr, all_Hit1, all_Hit3, all_Hit10, all_Hit20, all_Hit50 = [], [], [], [], [], []

    start = time.time()

    printlog('lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
    printlog('t_lr:{}'.format(optimizer.state_dict()['param_groups'][1]['lr']))
    t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()+8*3600))
    train_predict_file = open(args.train_file + f"mavenLGN_{epoch}_train_{args.t_lr}_{args.lr_decay}_{args.graph_enhance}_{args.graph_dropout}_" + t + '.txt', "w")
    dev_predict_file = open(args.dev_file + f"mavenLGN_{epoch}_dev_{args.t_lr}_{args.lr_decay}_{args.graph_enhance}_{args.graph_dropout}_" + t + '.txt', "w")
    test_predict_file = open(args.test_file + f"mavenLGN_{epoch}_test_{args.t_lr}_{args.lr_decay}_{args.graph_enhance}_{args.graph_dropout}_" + t + '.txt', "w")

    ############################################################################
    ##################################  train  #################################
    ############################################################################
    net.train()
    progress = tqdm.tqdm(total=len(train_data) // args.batch_size + 1, ncols=75, desc='Train {}'.format(epoch))

    for ii, batch_indices in enumerate(all_indices, 1):
        mode = 'train'
        progress.update(1)
        batch_arg, mask_arg, coevent_arg, causal_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, labels, \
            candiSet = get_batch(train_data, args, batch_indices, tokenizer, deactivate=args.graph_dropout, linearization_mode=args.linearization_mode)
        candiLabels = [] + labels
        for tt in range(len(labels)):
            candiLabels[tt] = candiSet[tt].index(labels[tt])
        batch_arg, mask_arg, mask_indices= batch_arg.to(device), mask_arg.to(device), mask_indices.to(device)
        for sent in sentences:
            for k in sent.keys():
                sent[k]['input_ids'] = sent[k]['input_ids'].to(device)
                sent[k]['attention_mask'] = sent[k]['attention_mask'].to(device)
        # fed data into network
        prediction = net(mode, batch_arg, mask_arg, coevent_arg, causal_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, args.batch_size)

        label = torch.LongTensor(labels).to(device)
        # loss
        loss = cross_entropy(prediction, label)
        if torch.isnan(loss):
            printlog('nan loss occured: epoch-{}, data idx-{}'.format(epoch, batch_indices))

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        savePredict(batch_indices, train_predict_file, prediction, candiSet, labels)

        mrr, hit1, hit3, hit10, hit20, hit50 = calculate(prediction, candiSet, labels, args.batch_size)
        Mrr += mrr
        Hit1 += hit1
        Hit3 += hit3
        Hit10 += hit10
        Hit20 += hit20
        Hit50 += hit50

        all_Mrr += mrr
        all_Hit1 += hit1
        all_Hit3 += hit3
        all_Hit10 += hit10
        all_Hit20 += hit20
        all_Hit50 += hit50

        if ii % (500 // args.batch_size) == 0:
            printlog('loss={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit20={:.4f} hit50={:.4f}'.format(
                loss_epoch / (30 // args.batch_size),
                sum(Hit1) / len(Hit1),
                sum(Hit3) / len(Hit3),
                sum(Hit10) / len(Hit10),
                sum(Hit20) / len(Hit20),
                sum(Hit50) / len(Hit50)))
            loss_epoch = 0.0
            Mrr, Hit1, Hit3, Hit10, Hit20, Hit50 = [], [], [], [], [], []
    end = time.time()
    progress.close()

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################

    all_indices = torch.arange(0, dev_size, dtype=torch.int32, device=device).split(args.batch_size)

    mode = 'dev'
    Mrr_d, Hit1_d, Hit3_d, Hit10_d, Hit20_d, Hit50_d = [], [], [], [], [], []

    progress = tqdm.tqdm(total=len(dev_data) // args.batch_size + 1, ncols=75, desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_arg, mask_arg, coevent_arg, causal_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, labels, \
            candiSet = get_batch(dev_data, args, batch_indices, tokenizer, linearization_mode=args.linearization_mode)

        candiLabels = [] + labels
        for tt in range(len(labels)):
            candiLabels[tt] = candiSet[tt].index(labels[tt])
        batch_arg, mask_arg, mask_indices= batch_arg.to(device), mask_arg.to(device), mask_indices.to(device)
        for sent in sentences:
            for k in sent.keys():
                sent[k]['input_ids'] = sent[k]['input_ids'].to(device)
                sent[k]['attention_mask'] = sent[k]['attention_mask'].to(device)
        # fed data into network
        prediction = net(mode, batch_arg, mask_arg, coevent_arg, causal_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, args.batch_size)
        savePredict(batch_indices, dev_predict_file, prediction, candiSet, labels)

        mrr, hit1, hit3, hit10, hit20, hit50 = calculate(prediction, candiSet, labels, args.batch_size)
        Mrr_d += mrr
        Hit1_d += hit1
        Hit3_d += hit3
        Hit10_d += hit10
        Hit20_d += hit20
        Hit50_d += hit50

    progress.close()

    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    all_indices = torch.arange(0, test_size, dtype=torch.int32, device=device).split(args.batch_size)

    mode = 'test'
    Mrr_t, Hit1_t, Hit3_t, Hit10_t, Hit20_t, Hit50_t = [], [], [], [], [], []

    progress = tqdm.tqdm(total=len(test_data) // args.batch_size + 1, ncols=75, desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)
        # get a batch of dev_data
        batch_arg, mask_arg, coevent_arg, causal_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, labels, \
            candiSet = get_batch(test_data, args, batch_indices, tokenizer, linearization_mode=args.linearization_mode)

        candiLabels = [] + labels
        for tt in range(len(labels)):
            candiLabels[tt] = candiSet[tt].index(labels[tt])
        batch_arg, mask_arg, mask_indices= batch_arg.to(device), mask_arg.to(device), mask_indices.to(device)
        for sent in sentences:
            for k in sent.keys():
                sent[k]['input_ids'] = sent[k]['input_ids'].to(device)
                sent[k]['attention_mask'] = sent[k]['attention_mask'].to(device)
        # fed data into network
        prediction = net(mode, batch_arg, mask_arg, coevent_arg, causal_arg, event_tokenizer_pos, event_key_pos, mask_indices, sentences, args.batch_size)

        savePredict(batch_indices, test_predict_file, prediction, candiSet, labels)

        mrr, hit1, hit3, hit10, hit20, hit50 = calculate(prediction, candiSet, labels, args.batch_size)
        Mrr_t += mrr
        Hit1_t += hit1
        Hit3_t += hit3
        Hit10_t += hit10
        Hit20_t += hit20
        Hit50_t += hit50

    progress.close()

    ############################################################################
    ##################################  result  ##################################
    ############################################################################
    ######### Train Results Print #########
    printlog('-------------------')
    printlog("TIME: {}".format(time.time() - start))
    printlog('EPOCH : {}'.format(epoch))
    printlog("TRAIN:")
    printlog('loss={:.4f} mrr={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit2-={:.4f} hit50={:.4f}'.format(
        loss_epoch / (30 // args.batch_size),
        sum(all_Mrr) / len(all_Mrr),
        sum(all_Hit1) / len(all_Hit1),
        sum(all_Hit3) / len(all_Hit3),
        sum(all_Hit10) / len(all_Hit10),
        sum(all_Hit20) / len(all_Hit20),
        sum(all_Hit50) / len(all_Hit50)))

    ######### Dev Results Print #########
    printlog("DEV:")
    printlog('loss={:.4f} mrr={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit2-={:.4f} hit50={:.4f}'.format(
        loss_epoch / (30 // args.batch_size),
        sum(Mrr_d) / len(Mrr_d),
        sum(Hit1_d) / len(Hit1_d),
        sum(Hit3_d) / len(Hit3_d),
        sum(Hit10_d) / len(Hit10_d),
        sum(Hit20_d) / len(Hit20_d),
        sum(Hit50_d) / len(Hit50_d)))

    ######### Test Results Print #########
    printlog("TEST:")
    printlog('loss={:.4f} mrr={:.4f} hit1={:.4f}, hit3={:.4f}, hit10={:.4f}, hit2-={:.4f} hit50={:.4f}'.format(
        loss_epoch / (30 // args.batch_size),
        sum(Mrr_t) / len(Mrr_t),
        sum(Hit1_t) / len(Hit1_t),
        sum(Hit3_t) / len(Hit3_t),
        sum(Hit10_t) / len(Hit10_t),
        sum(Hit20_t) / len(Hit20_t),
        sum(Hit50_t) / len(Hit50_t)))

    # record the best result
    if sum(Mrr_d) / len(Mrr_d) > dev_best_mrr:
        dev_best_mrr = sum(Mrr_d) / len(Mrr_d)
        best_mrr = sum(Mrr_t) / len(Mrr_t)
        best_mrr_epoch = epoch
    if sum(Hit1_d) / len(Hit1_d) > dev_best_hit1:
        dev_best_hit1 = sum(Hit1_d) / len(Hit1_d)
        best_hit1 = sum(Hit1_t) / len(Hit1_t)
        best_hit1_epoch = epoch
    if sum(Hit3_d) / len(Hit3_d) > dev_best_hit3:
        dev_best_hit3 = sum(Hit3_d) / len(Hit3_d)
        best_hit3 = sum(Hit3_t) / len(Hit3_t)
        best_hit3_epoch = epoch
    if sum(Hit10_d) / len(Hit10_d) > dev_best_hit10:
        dev_best_hit10 = sum(Hit10_d) / len(Hit10_d)
        best_hit10 = sum(Hit10_t) / len(Hit10_t)
        best_hit10_epoch = epoch
    if sum(Hit20_d) / len(Hit20_d) > dev_best_hit20:
        dev_best_hit20 = sum(Hit20_d) / len(Hit20_d)
        best_hit20 = sum(Hit20_t) / len(Hit20_t)
        best_hit20_epoch = epoch
    if sum(Hit50_d) / len(Hit50_d) > dev_best_hit50:
        dev_best_hit50 = sum(Hit50_d) / len(Hit50_d)
        best_hit50 = sum(Hit50_t) / len(Hit50_t)
        best_hit50_epoch = epoch

    printlog('=' * 20)
    printlog('Best result at mrr epoch: {}'.format(best_mrr_epoch))
    printlog('Best result at hit1 epoch: {}'.format(best_hit1_epoch))
    printlog('Best result at hit3 epoch: {}'.format(best_hit3_epoch))
    printlog('Best result at hit10 epoch: {}'.format(best_hit10_epoch))
    printlog('Best result at hit20 epoch: {}'.format(best_hit20_epoch))
    printlog('Best result at hit50 epoch: {}'.format(best_hit50_epoch))
    printlog('Eval mrr: {}'.format(best_mrr))
    printlog('Eval hit1: {}'.format(best_hit1))
    printlog('Eval hit3: {}'.format(best_hit3))
    printlog('Eval hit10: {}'.format(best_hit10))
    printlog('Eval hit20: {}'.format(best_hit20))
    printlog('Eval hit50: {}'.format(best_hit50))


    train_predict_file.close()
    dev_predict_file.close()
    test_predict_file.close()

    scheduler.step()
    state = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': args,
    }
    torch.save(state, args.model)