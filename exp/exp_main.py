from data_provider.data_factory import *
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, PatchTST, LSPatchT, iTransformer, Crossformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, get_model_size
from utils.metrics import metric
from utils.losses import MaskedLoss, DownstreamLoss
from fvcore.nn import FlopCountAnalysis


import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args, mlflow=None, setting=None):
        self.mlflow = mlflow
        super(Exp_Main, self).__init__(args)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # Writer will output to ./tensorflow/ directory by default
        if os.name == 'posix':
            self.writer = SummaryWriter(path + '/tensorflow/')
        else:
            self.writer = None

    def _calculate_flops(self):
        # Define the input shapes correctly
        input_shape = (1, self.args.seq_len, self.args.enc_in)  # Adjust num_features as needed
        x_mark_enc_shape = (1, self.args.seq_len, self.args.enc_in)  # Example shape, adjust as needed
        x_dec_shape = (1, self.args.label_len + self.args.pred_len, self.args.enc_in)  # Example shape, adjust as needed
        x_mark_dec_shape = (
        1, self.args.label_len + self.args.pred_len, self.args.enc_in)  # Example shape, adjust as needed

        # Create dummy inputs
        dummy_input = torch.randn(input_shape).to(self.device)
        dummy_x_mark_enc = torch.randn(x_mark_enc_shape).to(self.device)
        dummy_x_dec = torch.randn(x_dec_shape).to(self.device)
        dummy_x_mark_dec = torch.randn(x_mark_dec_shape).to(self.device)

        # Perform a forward pass to calculate FLOPS
        flops = FlopCountAnalysis(self.model, (dummy_input, dummy_x_mark_enc, dummy_x_dec, dummy_x_mark_dec))
        print(f"FLOPS: {flops.total()}")
        if self.mlflow:
            self.mlflow.log_metric("FLOPS", flops.total())

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'PatchTST': PatchTST,
            'LSPatchT': LSPatchT,
            'iTransformer': iTransformer,
            'Crossformer': Crossformer
            # 'Reformer': Reformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        # Log model summary.
        if os.name == 'posix':
            with open("mlflow_model_summary.txt", "w") as f:
                f.write(str(summary(model)))
        if self.mlflow: self.mlflow.log_artifact("mlflow_model_summary.txt")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.is_pretrain:
            return MaskedLoss()
        elif self.args.is_finetune:
            return DownstreamLoss()
        else:
            return nn.MSELoss()

    def _process_batch(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = [
            tensor.float().to(self.device) for tensor in batch
        ]
        return batch_x, batch_y, batch_x_mark, batch_y_mark

    def _prepare_decoder_input(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        return dec_inp

    def _compute_loss(self, batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, criterion):
        if self.args.is_pretrain:
            outputs, batch_y, mask = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            return criterion(outputs, batch_y, mask)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            return criterion(outputs, batch_y)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._process_batch(batch)
                # decoder input
                dec_inp = self._prepare_decoder_input(batch_y)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self._compute_loss(batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, criterion)
                else:
                    loss = self._compute_loss(batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, criterion)
                total_loss.append(loss.item())  # Convert loss to a Python number
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        # Initialize training steps, early stopping, optimizer, and loss criterion
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.epoch_time = time.time()
            # Log weight distribution at the start of each epoch
            self._log_weight_distribution(epoch)

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark = self._process_batch(batch)

                """
OUTPUT:                                               (216) (217) ... (240)
                                                                ↑
                            |---------------------------------------------------------------------------|
                            |                                 Full Architecture                         |
                            |---------------------------------------------------------------------------|
                                ↑                                                                   ↑
                              Encoder                                                            Decoder
[TOKENS index]:       (1) (2) (3) ... (168)                                 (169) (170) ... (215)   |   (216) x x x x x (240)
                          [ENCODER INPUT]                                      [DECODER INPUT]      | [DECODER OUTPUT (MASKED)]
                """

                # decoder input
                dec_inp = self._prepare_decoder_input(batch_y)

                # encoder - decoder
                if self.args.use_amp:
                    with (torch.cuda.amp.autocast()):
                        loss = self._compute_loss(batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, criterion)
                else:
                    loss = self._compute_loss(batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, criterion)
                train_loss.append(loss.item())
                self._log_training_progress(i, epoch, loss, len(train_loader))
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - self.epoch_time))
            train_loss = np.average(train_loss)
            vali_start_time = time.time()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            vali_time = time.time() - vali_start_time

            test_start_time = time.time()
            test_loss = self.vali(test_data, test_loader, criterion)
            test_time = time.time() - test_start_time

            # Log epoch-level metrics
            self._log_epoch_metrics(epoch, train_loss, vali_loss, test_loss, model_optim, vali_time, test_time)

            early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # Log the best model checkpoint path for MLFlow
        if self.mlflow:
            self.mlflow.log_param("model_checkpoint_path", os.path.abspath(best_model_path))
            self.mlflow.log_param("model_size_mb", get_model_size(self.model))

        return best_model_path

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './experiments/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape)

        # result save
        folder_path = './experiments/model_saved/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # log the metrics for MLFlow
        if self.mlflow:
            self._log_test_metrics(mae, mse, rmse, mape, mspe)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _log_training_progress(self, i, epoch, loss, num_iters):
        """
        Log training progress to console and MLFlow.
        """
        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            if self.mlflow:
                self._log_batch_metrics(loss, epoch * num_iters + i)
            # Log weight distribution into TensorBoard
            self._log_weight_distribution(epoch * num_iters + i)

    # log batch-level metrics
    def _log_batch_metrics(self, loss, step):
        self.mlflow.log_metrics({
            'batch_loss': loss.item(),
            'batch_time': time.time() - self.epoch_time
        }, step=step)
        return

    def _log_epoch_metrics(self, epoch, train_loss, vali_loss, test_loss, model_optim, vali_time, test_time):
        print(
            f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
        if self.writer:  # tensorboard
            self.writer.add_scalars('loss', {
                'train': train_loss,
                'vali': vali_loss,
                'test': test_loss
            }, epoch)

        if self.mlflow:
            self.mlflow.log_metrics({
                'train_loss': train_loss,
                'vali_loss': vali_loss,
                'test_loss': test_loss,
                'learning_rate': model_optim.param_groups[0]['lr'],
                'vali_time': vali_time,
                'test_time': test_time,
            }, step=epoch)

    def _log_test_metrics(self, mae, mse, rmse, mape, mspe):
        self.mlflow.log_metrics({
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        })

    def _log_weight_distribution(self, step):
        for name, param in self.model.named_parameters():
            if 'weight' in name and self.writer:  # Log only weights, not biases
                    self.writer.add_histogram(f'weights/{name}', param.data, step)
                    if param.grad is not None:
                        self.writer.add_histogram(f'gradients/{name}', param.grad, step)
