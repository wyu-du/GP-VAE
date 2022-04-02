import os
import argparse
import torch
import random
from torch import optim
from math import exp

from model import Copynet, Normal, MoG, Vamp, GP_Full
from training import train, evaluate
from utils import reader
from decode import BeamSearch


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    # Training config
    p.add_argument('-t', '--task', type=str, default="train", 
                   help="specify the task to do: (train)ing, (eval)uation, decode")
    p.add_argument('-epochs', type=int, default=200,
                   help='number of epochs for train')
    p.add_argument('-bs', '--batch_size', type=int, default=64,
                   help='number of epochs for train')
    p.add_argument('-optim', type=int, default=2,
                   help="optimization method. 0: SGD; 1: Adagrad, 2: Adam")
    p.add_argument('-lr', type=float, default=0.001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=5.0,
                   help='in case of gradient explosion')
    p.add_argument('-val_step', type=int, default=1,
                   help='the number of epoches between two validations')
    p.add_argument('-print_step', type=int, default=100,
                   help='the number of batches between two tf.summary')
    p.add_argument('-kw', type=float, default=0.,
                   help="scaler for kl annealing function")
    p.add_argument('-x0', type=int, default=25000,
                   help="mean of kl annealing function")
    # Model config
    p.add_argument('-mt', '--model_type', type=str, default='gp_full',
                   help='type of model: copynet/normal/mog/vamp/gp_full')
    p.add_argument('-e', '--embed_size', type=int, default=300,
                   help='embedding size')
    p.add_argument('-hs', '--hidden_size', type=int, default=512,
                   help='output dimension of the recognition network')
    p.add_argument('-k', '--latent_size', type=int, default=256,
                   help="dimension of the latent variable z")
    p.add_argument('-wd', '--word_dropout', type=float, default=0,
                   help='decoder input sentence dropout rate')
    p.add_argument('--embed_dropout', type=float, default=0.5,
                   help='word embedding dropout rate')
    p.add_argument('-c', '--components', type=int, default=10,
                   help="number of components for mixture gaussian")
    p.add_argument('-kld_sampled', type=int, default=0,
                   help="how to compute kld. 1: sampled kld; 0: analytical kld")
    p.add_argument('-v', '--kernel_v', type=float, default=65.0,
                   help="Hyper-parameter for prior kernel,  control the signal variance")
    p.add_argument('-r', '--kernel_r', type=float, default=0.0001,
                   help="Hyper-parameter for prior kernel.")
    p.add_argument('-vocab', '--vocab_size', type=int, default=20000,
                   help="vocab size")
    p.add_argument('-max_len', type=int, default=30,
                   help="max sequence length")
    p.add_argument('-using_cuda', type=bool, default=True,
                   help="whether to use cuda")
    # Generation - decoding strategy
    p.add_argument('-samn', '--sample_num', type=int, default=10,
                   help="Number of samples from latent codes.")
    p.add_argument('-bms', '--beam_size', type=int, default=10,
                   help="beam size, if 0, use greedy search; otherwise, using beam search.")
    p.add_argument('-df', '--data_file', type=str, default='../../data/GYAFC/em',
                   help='path to dataset')
    p.add_argument('-mf', '--model_file', type=str, default=None,
                   help='path to a pretrained model')
    # Generation - sampling latent codes
    p.add_argument('-topn', type=int, default=10,
                   help="number of generated samples for the decoder")
    p.add_argument('-std', type=float, default=1.,
                   help="Standard deviation for sampling z.")
    p.add_argument('-decf', '--decode_from', type=str, default='sample',
                   help="sample: decode from sampled latent codes; mean: decode from mean")
    p.add_argument('--seed', type=int, default=123)
    return p.parse_args()


def main():
    # --------------------------------------------
    # print("*** Parsing arguments ***")
    args = parse_arguments()
    args.using_cuda = args.using_cuda and torch.cuda.is_available()
    
    # --------------------------------------------
    print("*** Preparing dataset ... ***")
    train_iter, val_iter, test_iter, txtfield = reader(suffix=".tsv", 
                                                       rpath=args.data_file, 
                                                       batch_size=args.batch_size)
    print("[TRAIN]:%d (dataset:%d)\t[VAL]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(val_iter), len(val_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[vocab]:%d" % (len(txtfield.vocab)))

    # --------------------------------------------
    print("*** Initializing the model ... ***")
    if args.model_type == "copynet":
        model = Copynet(args, txtfield)
    elif args.model_type == "normal":
        model = Normal(args, txtfield)
    elif args.model_type == "mog":
        model = MoG(args, txtfield)
    elif args.model_type == "vamp":
        model = Vamp(args, txtfield)
    elif args.model_type == "gp_full":
        model = GP_Full(args, txtfield)
    else:
        raise ValueError("Unrecognized model type: %s" % args.model_type)
        
    # --------------------------------------------
    if (args.model_file is not None) and os.path.isfile(args.model_file):
        print("Loading the pretrained model from: {}".format(args.model_file))
        if args.using_cuda:
            model.load_state_dict(torch.load(args.model_file))
        else:
            model.load_state_dict(torch.load(args.model_file, map_location='cpu'))
    else:
        print("Couldn't find the pretrained model from: {}".format(args.model_file))
    if args.using_cuda: model.cuda()

    # --------------------------------------------
    optimizer = None
    params = model.parameters()
    if args.optim == 0:
        print("*** Choosing SGD as the optimization method ... ***")
        optimizer = optim.SGD(params, lr=args.lr)
    elif args.optim == 1:
        print("*** Choosing Adagrad as the optimization method ... ***")
        optimizer = optim.Adagrad(params, lr=args.lr)
    elif args.optim == 2:
        print("*** Choosing Adam as the optimization method ... ***")
        optimizer = optim.Adam(params, lr=args.lr)
    else:
        raise ValueError("Unspecified optimization method")

    # --------------------------------------------
    if args.task == "train":
        print("*** Training with validation ... ***")
        train(model, optimizer, train_iter, val_iter, txtfield, args)
    elif args.task == "eval":
        print("*** Testing ... ***")
        test_nll, test_kld = evaluate(model, test_iter, txtfield, args)
        print("[TEST] NLL: %5.4f KLD: %5.4f PPLx: %5.4f" % (test_nll, test_kld, exp(test_nll+test_kld)))
    elif args.task == "decode":
        print("*** Decode ... ***")
        # set random seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        beam_searcher = BeamSearch(model, txtfield, args)
        if args.decode_from == 'sample':
            beam_searcher.decode_sample(test_iter, txtfield, args, num_samples=30000)
        else:
            beam_searcher.decode_beam(test_iter, txtfield, args, num_samples=30000)        
    else:
        raise ValueError("Unrecognized task: {}".format(args.task))    


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
