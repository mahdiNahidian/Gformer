from data.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Pred,
)
from exp.exp_basic import Exp_Basic
from models.gformer import Gformer, GformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import warnings

warnings.filterwarnings('ignore')


class Exp_Gformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Gformer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'gformer': Gformer,
            'gformerstack': GformerStack,
        }

        if self.args.model in model_dict:
            e_layers = (
                self.args.e_layers
                if self.args.model == 'gformer'
                else self.args.s_layers
            )

            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device,
            ).float()
        else:
            raise ValueError(f"Unknown model: {self.args.model}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }

        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        if flag == 'train':
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag in ('val', 'test'):
            # Important for small custom datasets: do not silently drop
            # the whole validation/test split when len(split) < batch_size.
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            raise ValueError(f"Unknown data flag: {flag}")

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
        )

        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
        )

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion, epoch=None):
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                pred, true = self._process_one_batch(
                    vali_data,
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    epoch=epoch,
                )
                loss = criterion(
                    pred.detach().cpu(),
                    true.detach().cpu(),
                )
                total_loss.append(loss.item())

        if len(total_loss) == 0:
            raise RuntimeError(
                "Validation loader produced zero batches. "
                "Check split size and batch_size."
            )

        total_loss = float(np.average(total_loss))
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
        )
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch_idx in range(self.args.train_epochs):
            epoch_num = epoch_idx + 1
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                pred, true = self._process_one_batch(
                    train_data,
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    epoch=epoch_num,
                )

                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch_num, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch_idx)
                        * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(
                "Epoch: {} cost time: {}".format(
                    epoch_num, time.time() - epoch_time
                )
            )

            train_loss = float(np.average(train_loss))
            vali_loss = self.vali(
                vali_data,
                vali_loader,
                criterion,
                epoch=epoch_num,
            )

            print(
                "Epoch: {0}, Steps: {1} | "
                "Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch_num,
                    train_steps,
                    train_loss,
                    vali_loss,
                )
            )

            # Model selection uses validation only. Test is evaluated once
            # after training, in exp.test().
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch_num, self.args)

        best_model_path = path + '/checkpoint.pth'
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                pred, true = self._process_one_batch(
                    test_data,
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    epoch=None,
                )
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(
            folder_path + 'metrics.npy',
            np.array([mae, mse, rmse, mape, mspe]),
        )
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/checkpoint.pth'
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                pred, _ = self._process_one_batch(
                    pred_data,
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    epoch=None,
                )
                preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

    def _process_one_batch(
        self,
        dataset_object,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        epoch=None,
    ):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if self.args.padding == 0:
            dec_inp = torch.zeros(
                [
                    batch_y.shape[0],
                    self.args.pred_len,
                    batch_y.shape[-1],
                ]
            ).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones(
                [
                    batch_y.shape[0],
                    self.args.pred_len,
                    batch_y.shape[-1],
                ]
            ).float()
        else:
            raise ValueError(
                f"Unsupported padding value: {self.args.padding}"
            )

        dec_inp = torch.cat(
            [
                batch_y[:, :self.args.label_len, :],
                dec_inp,
            ],
            dim=1,
        ).float().to(self.device)

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                model_out = self.model(
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                    epoch=epoch,
                )
        else:
            model_out = self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                epoch=epoch,
            )

        outputs = (
            model_out[0]
            if self.args.output_attention
            else model_out
        )

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[
            :, -self.args.pred_len:, f_dim:
        ].to(self.device)

        # Prevent the silent broadcasting that occurred in the AAPL notebook
        # when c_out=1 but the target tensor had multiple variables.
        if outputs.shape != batch_y.shape:
            raise RuntimeError(
                "Prediction/target shape mismatch: "
                f"outputs={tuple(outputs.shape)}, "
                f"target={tuple(batch_y.shape)}. "
                "For multivariate-input/single-target forecasting use "
                "features='MS' and c_out=1; for M forecasting make c_out "
                "match the number of target variables."
            )

        return outputs, batch_y
