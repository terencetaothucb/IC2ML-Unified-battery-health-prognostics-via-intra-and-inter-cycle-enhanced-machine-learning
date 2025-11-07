import argparse
import os
import torch
from exp.exp_forecasting import Exp_Long_Term_Forecast1
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='IC2ML')

    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model', type=str,  default='IC2ML',
                        help='')

    parser.add_argument('--dataset', type=str, default='NCM', help='')
    parser.add_argument('--condition', type=str, default='CY25-05_1', 
                        help='NCA:CY45-05_1,CY25-05_1,CY25-025_1,CY25-1_1,CY35-05_1'
                        'NCM:CY45-05_1,CY25-05_1,CY35-05_1,NCMNCA:CY25-05_1,CY25-05_2,CY25-05_4')

    parser.add_argument('--hidden_dim', type=int, default=256, help='')
    parser.add_argument('--horizon', type=int, default=50, help='')
    parser.add_argument('--context', type=int, default=10, help='')
    parser.add_argument('--dataaccess', type=int, default=100, help='100,80,60,40,20')

    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=2000, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--checkpoints', type=str, default='./checkpoint', help='location of model checkpoints')
    parser.add_argument('--patience', type=str, default=100, help='location of model checkpoints')
    parser.add_argument('--inverse', type=str, default='no', help='s')
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')

    Exp = Exp_Long_Term_Forecast1

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args) 
            setting = '{}_dataset{}_context{}_horizon{}_hidden_dim{}_epoc{}_dm{}'.format(
                args.model,               
                args.dataset,             
                args.context,            
                args.horizon,             
                args.hidden_dim, 
                args.train_epochs,            
                ii                        
            )   
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_dataset{}_context{}_horizon{}_hidden_dim{}_epoc{}_dm{}'.format(
            args.model,               
            args.dataset,             
            args.context,            
            args.horizon,             
            args.hidden_dim, 
            args.train_epochs,            
            ii                        
        )   

        exp = Exp(args)  
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test1(setting, test=1)
        torch.cuda.empty_cache()
