import torch
import argparse
import numpy as np

from utils import *
from torch.utils.data import DataLoader
from solver import Solver
from solver_text import Solver_Text
from solver_fusion import Solver_Fusion
from solver_gb import Solver_GB
from config import get_args, get_config, output_dim_dict, criterion_dict
from data_loader import get_loader

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

if __name__ == '__main__':
    args = get_args()
    dataset = str.lower(args.dataset.strip())

    set_seed(args.seed)
    print("Start loading the data....")
    if args.test:
        args.num_epochs = 0

    train_config = get_config(dataset, mode='train', batch_size=args.batch_size, bert_model=args.bert_model)
    valid_config = get_config(dataset, mode='valid', batch_size=args.batch_size, bert_model=args.bert_model)
    test_config = get_config(dataset, mode='test',  batch_size=args.batch_size, bert_model=args.bert_model)

    # pretrained_emb saved in train_config here
    train_loader = get_loader(args, train_config, shuffle=True)
    print('Training data loaded!')
    valid_loader = get_loader(args, valid_config, shuffle=False)
    print('Validation data loaded!')
    test_loader = get_loader(args, test_config, shuffle=False)
    print('Test data loaded!')
    print('Finish loading the data....')

    torch.autograd.set_detect_anomaly(True)

    # addintional appending
    args.word2id = train_config.word2id

    # architecture parameters
    args.d_tin, args.d_vin, args.d_ain = train_config.tva_dim
    args.dataset = args.data = dataset
    args.when = args.when
    args.n_class = output_dim_dict.get(dataset, 1)
    args.criterion = criterion_dict.get(dataset, 'MSELoss')

    if args.fusion == 'none': # MMIM
        solver = Solver(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    elif args.fusion == 'text': # bert
        solver = Solver_Text(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    elif args.fusion == 'early' or args.fusion == 'late': # early or late fusion
        solver = Solver_Fusion(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    elif args.fusion == 'audio' or args.fusion == 'video': # audio or video uni-modal
        solver = Solver_Fusion(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    elif args.fusion == 'gb': # gradient blending
        solver = Solver_GB(args, train_loader=train_loader, dev_loader=valid_loader,
                    test_loader=test_loader, is_train=True)
    if not args.test:
        solver.train_and_eval()
    else:
        solver.test()
