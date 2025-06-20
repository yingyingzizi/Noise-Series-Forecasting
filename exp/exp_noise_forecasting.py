import matplotlib.pyplot as plt
from click.core import batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw

warnings.filterwarnings('ignore')


class Exp_Noise_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Noise_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()

        # 输出模型结构和参数量至Config_Result.txt
        with open(os.path.join(self.save_path, 'Config_Result.txt'), 'a') as f:
            f.write("--- Model Structure ---\n")
            f.write(str(model) + '\n')
            # 输出模型参数量，换算成MB
            f.write(f"--- Model Parameters ---\n")
            total_params = sum(p.numel() for p in model.parameters())
            total_params_mb = total_params / (1024 * 1024)  # 转换为MB
            f.write(f"Total parameters: {total_params_mb:.2f} MB\n")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, param_save_path=self.save_path)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.Adam(self.model.parameters(),
                                 lr=self.args.learning_rate,
                                 weight_decay=1e-4)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss(beta=0.5)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        tgt = vali_data.target_idx
        total_loss_norm = []
        total_loss_denorm = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:

                # 兼容 decompose / 不 decompose 两种返回格式
                if self.args.decompose:
                    (batch_x, batch_y, batch_x_mark,
                     batch_y_mark, _, _) = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # ---------- forward ----------
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                                    ).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # 只取目标列
                outputs = outputs[:, -self.args.pred_len:, tgt:tgt + 1]
                batch_y = batch_y[:, -self.args.pred_len:, tgt:tgt + 1]
                # # === 反归一化（与 test 一致）====
                # if vali_data.scale and self.args.inverse:
                #     shp = outputs.shape  # [B, pred_len, 1]
                #     # -> numpy
                #     out_np = outputs.detach().cpu().numpy().reshape(-1, shp[-1])
                #     y_np = batch_y.detach().cpu().numpy().reshape(-1, shp[-1])
                #     out_np = vali_data.inverse_transform(out_np).reshape(shp)
                #     y_np = vali_data.inverse_transform(y_np).reshape(shp)
                #     # <- tensor, 放回GPU再算loss
                #     outputs = torch.from_numpy(out_np).float().to(self.device)
                #     batch_y = torch.from_numpy(y_np).float().to(self.device)
                # # =================================
                # loss = criterion(outputs, batch_y)
                # total_loss.append(loss.item())
                # === 计算两种 loss: 归一化 & 反归一化 ==================
                loss_norm = criterion(outputs, batch_y)  # 用这个 early-stop
                if self.args.inverse and vali_data.scale:
                    shp = outputs.shape
                    out_np = outputs.detach().cpu().numpy().reshape(-1, 1)
                    y_np = batch_y.detach().cpu().numpy().reshape(-1, 1)
                    out_np = vali_data.inverse_transform(out_np).reshape(shp)
                    y_np = vali_data.inverse_transform(y_np).reshape(shp)
                    loss_denorm = criterion(torch.from_numpy(out_np).to(self.device),
                                            torch.from_numpy(y_np).to(self.device))
                else:
                    loss_denorm = loss_norm.detach() * 0

                total_loss_norm.append(loss_norm.item())
                total_loss_denorm.append(loss_denorm.item())

        self.model.train()
        return np.mean(total_loss_norm), np.mean(total_loss_denorm)

    def train(self, setting):
        """
        训练过程：构造Dataset/DataLoader -> 模型训练 -> 记录loss -> 可视化 -> 保存模型
        """
        # 1) 构造训练集和验证集
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = ReduceLROnPlateau(  # 学习率衰减
            model_optim,
            mode='min',  # 监控的指标越小越好
            factor=self.args.scheduler_factor,  # 每次衰减到原来的 1/2
            patience=self.args.patience,  # 连续 3 个 epoch 没提升就衰减
            verbose=True
        )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 这两个列表用来存 epoch 级别的 train_loss 和 val_loss，用于画图
        epoch_train_loss_list = []
        epoch_val_loss_list = []

        train_target_idx = train_data.target_idx

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
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
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # ---------------- target 取列 ----------------
                        outputs = outputs[:, -self.args.pred_len:, train_target_idx:train_target_idx + 1]
                        batch_y = batch_y[:, -self.args.pred_len:, train_target_idx:train_target_idx + 1]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # ---------------- target 取列 ----------------
                    outputs = outputs[:, -self.args.pred_len:, train_target_idx:train_target_idx + 1]
                    batch_y = batch_y[:, -self.args.pred_len:, train_target_idx:train_target_idx + 1]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 每个epoch结束，计算训练集平均loss
            train_loss_epoch = np.average(train_loss)
            epoch_train_loss_list.append(train_loss_epoch)

            # 验证集loss
            vali_loss_norm, vali_loss_denorm = self.vali(vali_data, vali_loader, criterion)
            epoch_val_loss_list.append(vali_loss_denorm)
            # 测试集loss
            test_loss, _ = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Cost Time: {1}".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_epoch, vali_loss_norm, test_loss))

            # ========== Early Stopping ==========
            early_stopping(vali_loss_norm, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # # 调整学习率
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            # 根据验证集 loss 调整学习率
            scheduler.step(vali_loss_norm)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        # 绘制训练集和验证集的loss曲线
        self._plot_loss_curve(epoch_train_loss_list, epoch_val_loss_list,
                              title='Train/Val Loss',
                              save_name=os.path.join(self.save_path, 'loss_epoch.png')
                              )

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        tgt = test_data.target_idx
        if test:
            best_model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            print('loading model from', best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        folder_path = os.path.join(self.save_path, 'test_results')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if self.args.decompose:
                    (batch_x, batch_y, batch_x_mark,
                     batch_y_mark, batch_trend, batch_season) = batch
                    batch_trend = batch_trend.float().to(self.device)
                    batch_season = batch_season.float().to(self.device)
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
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
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, tgt:tgt + 1]
                batch_y = batch_y[:, -self.args.pred_len:, tgt:tgt + 1]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()  # shape [B, label+pred, F]

                # inverse_transform
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        repeat_times = batch_y.shape[-1] // outputs.shape[-1]
                        outputs = np.tile(outputs, [1, 1, repeat_times])
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                if self.args.decompose and self.args.inverse:
                    # ----------------- 还原 resid → Noise -----------------
                    # batch_trend / season 仍是缩放前的原始值 → 不需要 inverse_transform
                    batch_trend = batch_trend[:, -self.args.pred_len:, :]  # 取最后 pred_len 步
                    batch_season = batch_season[:, -self.args.pred_len:, :]
                    outputs = outputs + batch_trend.cpu().numpy() + batch_season.cpu().numpy()
                    batch_y = batch_y + batch_trend.cpu().numpy() + batch_season.cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                # --- 预测值真实值可视化：对当前 batch 里的前 num_vis 条都画 ---
                if len(preds) == 1:  # 只在第一个 batch 里画
                    input_np = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shp = input_np.shape
                        input_np = test_data.inverse_transform(
                            input_np.reshape(shp[0]*shp[1], -1)).reshape(shp)
                    vis_cnt = min(self.args.num_vis, input_np.shape[0])
                    for idx in range(vis_cnt):
                        gt = np.concatenate((input_np[idx, :, tgt], batch_y[idx, :, 0]), axis=0)
                        pd_ = np.concatenate((input_np[idx, :, tgt], outputs[idx, :, 0]), axis=0)
                        visual(gt, pd_, os.path.join(folder_path, f'sample_{i}_{idx}.png'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        # 输出评价指标至Config_Result.txt
        with open(os.path.join(self.save_path, 'Config_Result.txt'), 'a') as f:
            f.write("--- Test Result ---\n")
            f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
            f.write('\n')
            f.close()

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        return

        # ======== 绘制函数 =========
    def _plot_loss_curve(self, train_loss, val_loss, title, save_name):
        plt.figure()
        plt.plot(range(len(train_loss)), train_loss, label='Train Loss')
        plt.plot(range(len(val_loss)), val_loss, label='Val Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_name)
        plt.close()
