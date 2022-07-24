import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()

    # used for auto_train.py to split training data
    parser.add_argument('-data', type=str, default='dataset_3_learning.json', help='dataset json file')
    parser.add_argument('-train_ratio', type=float, default=0.6,
                        help='proportion of data used for training (default 0.6)')
    parser.add_argument('-val_ratio', type=float, default=0.2,
                        help='proportion of data used for validation (default 0.2)')
    parser.add_argument('-no_cuda', action='store_true', default=False)

    parser.add_argument('-input_dim', type=int, default=4, help='')
    parser.add_argument('-output_dim', type=int, default=1, help='')

    parser.add_argument('-vocab', default='vocab.json', type=str)
    parser.add_argument('-epoch_number', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=2)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-num_workers', type=int, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    return args
