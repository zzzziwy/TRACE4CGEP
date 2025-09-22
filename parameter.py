# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    # Dataset
    parser.add_argument('--len_arg_text', default=200, type=int, help='Sentence length')
    parser.add_argument('--len_arg_template', default=200, type=int, help='Template length')
    parser.add_argument('--len_temp', default=0, type=int, help='Template length')
    parser.add_argument('--dataset_path', default='MAVEN', type=str, help='')
    parser.add_argument('--dataset_type', default='train', type=str, help='')
    # Few-shot
    parser.add_argument('--Sample_rate', default=0.0, type=float, help='Few shot rate')

    # Model
    parser.add_argument('--model_name', default='roberta-base', type=str, help='Model used to be encoder')
    parser.add_argument('--init_vocab_size', default=50265, type=int, help='Original Size of RoBERTa vocab')
    parser.add_argument('--vocab_size', default=50265, type=int, help='Size of RoBERTa vocab')
    parser.add_argument('--label_smoothing', default=0, type=float, help='label_smoothing')
    parser.add_argument('--no-graph_enhance', action="store_false", dest="graph_enhance", default=True,
                        help='disable graph enhancement')
    parser.add_argument('--no-graph_dropout', action="store_false", dest="graph_dropout", default=True,
                        help='disable graph dropout')
    parser.add_argument('--linearization_mode', default="sequential", type=str, help='')
    # sequential, reverse, random
    # train
    parser.add_argument('--num_epoch', default=10, type=int, help='Number of total epochs to run prompt learning')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for prompt learning')
    parser.add_argument('--t_lr', default=2e-6, type=float, help='Initial lr')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
    parser.add_argument('--lr_step', default=5, type=int, help='lr step')
    parser.add_argument('--lr_decay', default=1, type=float, help='lr decay')

    # Others
    parser.add_argument('--seed', default=209, type=int, help='Seed for reproducibility')
    parser.add_argument('--log', default='./out/', type=str, help='Log result file name')
    parser.add_argument('--outmodel', default='outmodel', type=str, help='Model parameters result file folder')
    parser.add_argument('--model', default='', type=str, help='Model parameters result file name')
    parser.add_argument('--train_file', default='./predict/train/', type=str, help='Saved train file')
    parser.add_argument('--dev_file', default='./predict/dev/', type=str, help='Saved dev file')
    parser.add_argument('--test_file', default='./predict/test/', type=str, help='Saved test file')
    parser.add_argument('--padding_idx', default=1, type=int, help='the padding index of the PLM')

    args = parser.parse_args()
    return args
