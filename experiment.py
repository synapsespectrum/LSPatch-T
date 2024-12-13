import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

import mlflow
import mlflow.pytorch


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Transformer models based family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--is_pretrain', type=int, required=False, default=0, help='status')
    parser.add_argument('--is_finetune', type=int, required=False, default=0, help='status')
    parser.add_argument('--pretrained_model', type=str, required=False, default=None, help='pretrained model path')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Transformer',
                        help='model name, options: [Transformer, Informer, Autoformer, PatchTST, LSPatchT, iTransformer, Crossformer]')
    parser.add_argument('--use_mlflow', type=bool, default=True, help='use mlflow')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1',
                        help='dataset type for choosing Data Loader')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./experiments/model_saved/checkpoints/',
                        help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, '
                             'S:univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    # parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    # parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
    parser.add_argument('--patch_len', type=int, default=12, help='patch length')
    parser.add_argument('--stride', type=int, default=12, help='stride between patch')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # For debugging
    data_parser = {
        'ETTh1': {'data_path': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        # 'T' is target, example here is 'OT'
        'ETTh2': {'data_path': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data_path': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data_path': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data_path': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'Solar': {'data_path': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1],
                  'MS': [137, 137, 1]},
    }

    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    print('Args in experiment:')
    print(args)

    if args.is_training:
        # setting record of experiments
        setting = get_exp_setting(args)
        if args.use_mlflow:
            mlflow.set_experiment(f"{args.model_id}_{args.model}_{args.data}_{args.seq_len}_{args.pred_len}")
            if args.is_pretrain:
                args.itr = 1
            elif args.is_finetune:  # fine tuning or supervised
                if args.pretrained_model is None:
                    raise ValueError("Please provide the path of the pretrained model")
                if not os.path.exists(args.pretrained_model):
                    print('>>>>>>>supervised : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            best_model_path = tracking_exp(setting, args)
            print('>>>>>>>best model path : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(best_model_path))
            if args.is_finetune and args.is_pretrain:  # fine tuning
                args.itr = 2
                args.is_pretrain = 0
                args.batch_size = 32
                args.train_epochs = 10
                args.patch_len = args.seq_len
                args.stride = args.seq_len
                args.model_id = "FineTune"
                args.pretrained_model = best_model_path
                setting = get_exp_setting(args)
                print('>>>>>>>fine tuning : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                print("Loading pretrained model from: ", best_model_path)
                tracking_exp(setting, args)

        else:
            for ii in range(args.itr):
                setting_name = f'{setting}_{ii}'
                exp = Exp_Main(args, None, setting_name)
                run_experiment(exp, setting_name, args)
    else:
        setting = f'{get_exp_setting(args)}_{0}'
        exp = Exp_Main(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


def get_exp_setting(args):
    if args.is_pretrain:
        return '{}_{}_{}_sl{}'.format(
            args.model_id, args.model, args.data, args.seq_len)  # set the name of pretrain model
    return '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.model_id, args.model, args.data, args.features, args.seq_len, args.label_len,
        args.pred_len, args.d_model, args.n_heads, args.e_layers, args.d_layers,
        args.d_ff, args.factor, args.embed, args.distil, args.des)


def tracking_exp(setting, args):
    # Start a parent run for each configuration
    with mlflow.start_run(run_name=setting) as parent_run:
        mlflow.log_params(vars(args))  # Log parameters in the parent run
        mlflow.log_artifact(f"models/{args.model}.py")
        for ii in range(args.itr):
            setting_name = '{}_{}'.format(setting, ii)
            with mlflow.start_run(run_name=f"{args.model}_Iteration_{ii}", nested=True) as child_run:
                # Enable system metrics logging
                mlflow.system_metrics.enable_system_metrics_logging()  # Log system metrics
                exp = Exp_Main(args, mlflow, setting_name)
                best_model_path = run_experiment(exp, setting_name, args)
    return best_model_path


def run_experiment(exp, setting_name, args):
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting_name))
    best_model_path = exp.train(setting_name)
    if args.is_pretrain:
        return best_model_path

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting_name))
    exp.test(setting_name)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting_name))
        exp.predict(setting_name, True)

    torch.cuda.empty_cache()
    return best_model_path


if __name__ == "__main__":
    main()
