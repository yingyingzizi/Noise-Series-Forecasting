import os
import json
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')

class Dataset_Noise(Dataset):
    def __init__(self, args, root_path, param_save_path, flag='train', size=None,
                 features='MS', data_path='Noise_1000Hz_4896.csv',
                 target='Noise', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # 默认窗口大小（基于 3 小时采样）
        if size is None:
            self.seq_len = 8 * 4  # 4 天（32 个时间步）
            self.label_len = 8  # 1 天（8 个时间步）
            self.pred_len = 8  # 1 天（8 个时间步）
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # 数据集划分
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.sample_step = self.args.sample_step

        self.root_path = root_path
        self.data_path = data_path
        self.save_path = param_save_path
        self.__read_data__()

    def __read_data__(self):
        # 根据args.scaler类型动态创建scaler
        if self.args.scaler.lower() == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ---------- 2.2 Spike removal on Noise ----------
        if self.args.spike_method != 'none':
            noise = df_raw['Noise'].copy()
            if self.args.spike_method == 'mad':
                med = noise.median()
                mad = 1.4826 * (noise - med).abs().median()
                mask = (noise - med).abs() > self.args.spike_thresh * mad
                df_raw['Noise'] = noise.where(~mask, med)  # replace by median
                # 可选：生成事件列
                df_raw['is_spike'] = mask.astype(int)
            elif self.args.spike_method == 'winsor':
                q_low, q_hi = noise.quantile([0.01, 0.99])
                df_raw['Noise'] = noise.clip(lower=q_low, upper=q_hi)
                df_raw['is_spike'] = ((noise < q_low) | (noise > q_hi)).astype(int)

        # ---------- 2.3 STL decomposition (trend/season/resid) ----------
        if self.args.decompose:
            from statsmodels.tsa.seasonal import STL
            # STL 期望 datetime index
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            df_raw = df_raw.set_index('date')
            stl = STL(df_raw['Noise'], period=self.args.stl_period)
            res = stl.fit()
            df_raw['Noise_trend'] = res.trend
            df_raw['Noise_season'] = res.seasonal
            df_raw['Noise_resid'] = res.resid
            self.trend = df_raw['Noise_trend'].values
            self.season = df_raw['Noise_season'].values
            # **用 resid 作为新预测目标**
            self.target = 'Noise_resid'
            df_raw = df_raw.reset_index()  # 恢复

        # 对数据集做log1p变换（如果需要）
        if self.args.log_target:
            df_raw[self.target] = np.log1p(df_raw[self.target])

        # 读取数据集长度（时间步数）
        data_length = len(df_raw)  # 获取数据集的行数，即时间步数

        # 根据 4896 个时间步划分数据集（80%-10%-10%）
        train_size = int(data_length * self.args.train_ratio)
        val_size = int(data_length * self.args.valid_ratio)
        test_size = data_length - train_size - val_size

        border1s = [0, train_size - self.seq_len, train_size + val_size - self.seq_len]
        border2s = [train_size, train_size + val_size, data_length]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ---------- 选取特征列 ----------
        if self.features in ['M', 'MS']:
            # 基础列：除 date 外的所有原始列
            base_cols = df_raw.columns.drop('date')  # 省心、不会把 date 拉进来
            # 如果做了 STL 分解且希望把 trend/season 当特征
            if self.args.decompose:
                extra_cols = pd.Index(['Noise_trend', 'Noise_season'])
                cols_data = base_cols.union(extra_cols)  # 用 Index.union，保证去重 & 顺序稳定
            else:
                cols_data = base_cols
            df_data = df_raw[cols_data]

        elif self.features == 'S':  # 单变量
            df_data = df_raw[[self.target]]  # target 可能是 Noise 或 Noise_resid

        else:  # 万不得已的兜底
            df_data = df_raw.copy()


        # ============== 关键：对train/val/test三种flag，分别处理 ==============
        if self.scale:
            if self.set_type == 0:
                # 'train'模式：fit后transform，并保存scaler参数
                train_data = df_data[border1s[0]:border2s[0]]  # 训练集范围
                self.scaler.fit(train_data.values)  # 仅在训练集上fit

                # 训练集中得到的均值/方差或min/max，应用到整个df_data
                data = self.scaler.transform(df_data.values)

                # 保存Scaler参数（示例：保存到args.scaler_path）
                # 你也可以组合到实验文件夹下，比如args.checkpoints
                self._save_scaler_params()
            else:
                # 'val'或'test'模式：从文件中加载训练好的scaler，再transform
                self.scaler = self._load_scaler_params()
                data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 时间特征
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='3h')
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.col_map = {c: i for i, c in enumerate(cols_data)}  # 跳过 date
        self.target_idx = self.col_map[self.target]

    def __getitem__(self, index):
        index = index * self.sample_step  # 根据sample_step调整索引
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.args.decompose:
            # trend / season 与 seq_y 对齐（预测窗口）
            seq_trend = self.trend[r_begin:r_end].reshape(-1, 1)
            seq_season = self.season[r_begin:r_end].reshape(-1, 1)
            return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_trend, seq_season
        else:
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len) // self.sample_step + 1

    def _save_scaler_params(self):
        """
            将scaler的参数(Mean/Std或Min/Max)保存到json文件中。
        """
        params = {}
        if self.args.scaler.lower() == 'minmax':
            params = {
                'min': self.scaler.data_min_.tolist(),
                'max': self.scaler.data_max_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'min_': self.scaler.min_.tolist()
            }
        else:
            # standardScaler
            params = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        save_path = self.save_path
        # 输出归一化参数至json文件
        with open(os.path.join(self.save_path, 'scaler_params.json'), 'w') as f:
            json.dump(params, f, indent=4)


    def _load_scaler_params(self):
        """
            从json文件中加载scaler参数，构造并返回一个scaler对象。
        """
        load_path = self.save_path
        with open(os.path.join(load_path, 'scaler_params.json'), 'r') as f:
            params = json.load(f)
        if self.args.scaler.lower() == 'minmax':
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array(params['min'])
            scaler.data_max_ = np.array(params['max'])
            scaler.scale_ = np.array(params['scale'])
            scaler.min_ = np.array(params['min_'])
        else:
            # standardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(params['mean'])
            scaler.scale_ = np.array(params['scale'])

        # 需要加上特征数
        scaler.n_features_in = len(scaler.scale_)
        return scaler

    def inverse_transform(self, data):
        """
            data : np.ndarray, shape (..., C_out)  (C_out 可以是 1，也可以 = n_features)
            只对目标列做逆变换，其它列保持 0（不会影响结果）
            """
        orig_shape = data.shape
        data_2d = data.reshape(-1, orig_shape[-1])  # (N, C_out)

        n_total = self.scaler.scale_.shape[0]  # = 10
        if data_2d.shape[1] == n_total:
            inv = self.scaler.inverse_transform(data_2d)  # 直接还原
        else:
            # --- 把目标列“嵌入”到全列矩阵，再 inverse ---
            tmp = np.zeros((data_2d.shape[0], n_total))
            # 允许一次性还原多列（C_out>1）时保持顺序
            tgt_slice = slice(self.target_idx,
                              self.target_idx + data_2d.shape[1])
            tmp[:, tgt_slice] = data_2d

            inv_full = self.scaler.inverse_transform(tmp)
            inv = inv_full[:, tgt_slice]  # 取回目标列

        inv = inv.reshape(orig_shape)
        if self.args.log_target:
            inv = np.expm1(inv)  # 还原 log1p
        return inv


