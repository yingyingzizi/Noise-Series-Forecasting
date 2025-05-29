import os, json, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
import matplotlib
matplotlib.use('Agg')          # 彻底关闭交互 GUI
from datetime import datetime
import pathlib, itertools

class DataExplorer:
    _counter = itertools.count()  # 全局计数器，自动编号文件

    def __init__(self, csv_path: str, time_col: str = 'date',
                 scaler: str = 'standard', save_root='../data_fig',     # ► 总图保存目录
                 start_tag=None, end_tag=None):
        """
        csv_path : 原始 csv 文件
        time_col : 时间戳列名
        scaler   : 'standard' 或 'minmax'
        """
        self.save_root = pathlib.Path(save_root)
        self.start_tag = start_tag or 'full'
        self.end_tag = end_tag or 'full'
        # ---------- 创建本次会话的子目录 ----------
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        sub = f"{ts}_{self.start_tag}-{self.end_tag}"
        self.save_dir = self.save_root / sub
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # ----------------------------------------

        self.df_raw = pd.read_csv(csv_path)
        self.time_col = time_col
        self.df_raw[time_col] = pd.to_datetime(self.df_raw[time_col])
        self.df_raw.sort_values(time_col, inplace=True)

        self.features = [c for c in self.df_raw.columns if c != time_col]
        if scaler.lower() == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        self.norm = pd.DataFrame(
            self.scaler.fit_transform(self.df_raw[self.features]),
            columns=self.features,
            index=self.df_raw[time_col],
        )

    # ----------------------------------------------------------
    # 2) 单变量 / 多变量可视化
    # ----------------------------------------------------------
    def plot_series(self, cols=None, start=None, end=None,
                    normalized=False):
        """
        cols : list[str]  需要查看的列；None=全部
        start/end : 支持 datetime / str / int 索引
        normalized : True -> 归一化后曲线
        """
        df = self.norm if normalized else self.df_raw.set_index(self.time_col)
        if cols is None:
            cols = self.features
        df = df[cols]
        if start is not None or end is not None:
            df = df.loc[start:end]

        plt.figure(figsize=(12,4))
        for col in cols:
            plt.plot(df.index, df[col], label=col, linewidth=1)
        plt.legend()
        plt.title(f"{'Normalized' if normalized else 'Raw'} Series")
        plt.title(f"{'Norm' if normalized else 'Raw'} {'/'.join(cols)}")
        self._finalize_fig('plot_series')

    # ----------------------------------------------------------
    # 4) KDE 分布对比
    # ----------------------------------------------------------
    def kde_compare(self, train_idx, val_idx, test_idx, col):
        """
        train/val/test_idx: list or slice of indices
        col : 目标列
        """
        plt.figure(figsize=(6,4))
        sns.kdeplot(self.df_raw.loc[train_idx, col], label='train')
        sns.kdeplot(self.df_raw.loc[val_idx, col],   label='val')
        sns.kdeplot(self.df_raw.loc[test_idx, col],  label='test')
        plt.title(f'KDE of {col}')
        plt.legend()
        self._finalize_fig('kde_compare')

    # ----------------------------------------------------------
    # 5) 异常值检测 / 剔除
    # ----------------------------------------------------------
    def remove_outliers(self, col, method='zscore', thresh=3.5):
        """
        method: 'zscore' | 'iqr' | 'hampel'
        return: cleaned Series
        """
        s = self.df_raw[col].copy()
        if method == 'zscore':
            mask = np.abs(stats.zscore(s)) < thresh
        elif method == 'iqr':
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1
            mask = (s > q1 - 1.5*iqr) & (s < q3 + 1.5*iqr)
        elif method == 'hampel':
            med = np.median(s)
            mad = np.median(np.abs(s - med))
            mask = np.abs(s - med) < thresh * 1.4826 * mad
        else:
            raise ValueError('unknown method')
        return s[mask]

    # ----------------------------------------------------------
    # 6) 相关系数 & PCA
    # ----------------------------------------------------------
    def corr_heatmap(self):
        plt.figure(figsize=(6,5))
        sns.heatmap(self.df_raw[self.features].corr(),
                    cmap='coolwarm', annot=False)
        plt.title('Feature Correlation')
        self._finalize_fig('corr_heatmap')

    def pca_plot(self, n_comp=2):
        pca = PCA(n_components=n_comp)
        Z = pca.fit_transform(self.norm.values)
        plt.figure(figsize=(5,5))
        plt.scatter(Z[:,0], Z[:,1], s=3, alpha=0.6)
        plt.title(f'PCA (n={n_comp}) var={pca.explained_variance_ratio_.sum():.2%}')
        self._finalize_fig('pca_plot')

    # ============= 内部工具：统一保存 =============
    def _finalize_fig(self, fig_title):
        idx = next(DataExplorer._counter)
        fname = self.save_dir / f"{fig_title}_{idx:03d}.png"
        plt.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close()  # 不 show
        print(f"[DataExplorer] figure saved -> {fname}")

de = DataExplorer('C:/my/硕士/科研数据/all_datasets/Noise//Noise_1000Hz_4896.csv',
                  start_tag='2018-01-01', end_tag='2018-03-01')
de.plot_series(cols=['Noise'], start='2018-01-01', end='2018-03-01')
for c in de.features:
    de.plot_series(cols=[c], start='2018-01-01', end='2018-03-01')
de.plot_series(normalized=True, start='2018-01-01', end='2018-03-01')
de.kde_compare(range(0,3900), range(3900,4390), range(4390,4870),
               col='Noise')
de.corr_heatmap()
clean = de.remove_outliers('Noise', method='hampel')
