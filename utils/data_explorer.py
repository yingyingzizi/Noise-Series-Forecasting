"""
A lightweight utility class for quick EDA (exploratory-data-analysis)
on the Noise-Series-Forecasting project.

© 2025
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# --------------------------------------------------------------------------- #
#                               Helper utilities                              #
# --------------------------------------------------------------------------- #
def _ensure_dir(path: Path | str) -> Path:
    """Create parent directory of *path* if it doesn't exist, then return Path."""
    p = Path(path)
    if p.suffix:          # has filename
        p.parent.mkdir(parents=True, exist_ok=True)
    else:                 # path itself is a directory
        p.mkdir(parents=True, exist_ok=True)
    return p


def _auto_name(save_root: Path,
               subdir: str,
               file_name: str) -> Path:
    """Generate save path and guarantee directory exists."""
    return _ensure_dir(save_root / subdir / file_name)


# --------------------------------------------------------------------------- #
#                                Main class                                   #
# --------------------------------------------------------------------------- #
class DataExplorer:
    """
    Quick visual / statistical inspection for the Noise dataset.
    """

    def __init__(self,
                 csv_path: str | os.PathLike,
                 save_root: str | os.PathLike = "data_fig",
                 scaler: str = "standard",
                 start_tag: str = "train",
                 end_tag: str = "test"):
        """
        Parameters
        ----------
        csv_path
            Path to the raw CSV (must include a ``date`` column).
        save_root
            Root directory for all figures. Will be created if absent.
        scaler
            'standard' or 'minmax' – used when plotting normalized curves.
        start_tag, end_tag
            For labelling the figure title (e.g. "2018-01-01 – 2018-03-01")
        """
        self.csv_path = Path(csv_path)
        self.save_root = _ensure_dir(save_root)
        self.start_tag = start_tag
        self.end_tag = end_tag

        # read csv
        self.df: pd.DataFrame = pd.read_csv(self.csv_path, parse_dates=["date"])

        # optional scaler
        if scaler.lower().startswith("min"):
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(self.df.drop(columns=["date"]).values)

    # ----------------------------- STL plot -------------------------------- #
    def plot_stl(self,
                 feature: str,
                 period: int = 24,
                 train_border: float = 0.8,
                 val_border: float = 0.9,
                 save_path: Optional[str | os.PathLike] = None) -> None:
        """
        STL decomposition of *feature* and plot **train / val / test** parts.

        * If *save_path* is ``None``, file will be saved to
          ``<save_root>/stl_<feature>/<feature>_stl.png``.
        """
        if feature not in self.df.columns:
            raise ValueError(f"'{feature}' not found in CSV columns.")

        series = self.df.set_index("date")[feature]
        stl = STL(series, period=period)
        res = stl.fit()

        # split indices
        n = len(series)
        train_end = int(n * train_border)
        val_end = int(n * val_border)

        # --- plotting ---
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True,
                                 gridspec_kw=dict(hspace=0.3))
        titles = ["Original", "Trend", "Seasonal", "Residual"]
        comps = [series, res.trend, res.seasonal, res.resid]

        # Color maps for three partitions
        cmap = dict(train="#1f77b4", val="#ff7f0e", test="#2ca02c")

        for ax, data, title in zip(axes, comps, titles):
            ax.set_title(title, fontsize=11)

            # plot each partition separately for color legend
            ax.plot(data.iloc[:train_end], color=cmap["train"], label="train")
            ax.plot(data.iloc[train_end:val_end], color=cmap["val"], label="val")
            ax.plot(data.iloc[val_end:], color=cmap["test"], label="test")

            ax.legend(fontsize=8, ncol=3, loc="upper right")

        axes[-1].set_xlabel("Date")
        fig.suptitle(f"STL Decomposition of '{feature}' "
                     f"({self.start_tag} → {self.end_tag})", fontsize=13)

        # resolve save path
        if save_path is None:
            save_path = _auto_name(self.save_root,
                                   f"stl_{feature}",
                                   f"{feature}_stl.png")
        else:
            save_path = _ensure_dir(save_path)

        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    # --------------------------- Raw / normalized -------------------------- #
    def plot_series(self,
                    cols: Sequence[str],
                    normalized: bool = False,
                    sample_range: Optional[Tuple[int, int]] = None,
                    save_path: Optional[str | os.PathLike] = None) -> None:
        """
        Plot raw or normalized time-series curves for given columns.

        Parameters
        ----------
        cols : list[str]
            Column names to plot.
        normalized : bool
            If True, use the scaler (same stats as训练) 显示标准化/归一化曲线.
        sample_range : (start_idx, end_idx)
            Optional slice to zoom in.
        """
        base_df = self.df.set_index("date")[list(cols)]

        # ---- slice if needed ----
        if sample_range:
            s, e = sample_range
            base_df = base_df.iloc[s:e]

        if normalized:
            # -------------------- 方案 B : 手动缩放 -------------------- #
            full_cols = self.df.columns.drop("date")
            idx = [full_cols.get_loc(c) for c in cols]   # index of each col in scaler stats

            if isinstance(self.scaler, StandardScaler):
                μ = self.scaler.mean_[idx]
                σ = self.scaler.scale_[idx]
                scaled_vals = (base_df.values - μ) / σ
            else:  # MinMaxScaler
                dmin = self.scaler.data_min_[idx]
                dmax = self.scaler.data_max_[idx]
                scaled_vals = (base_df.values - dmin) / (dmax - dmin)

            df_plot = pd.DataFrame(scaled_vals,
                                   index=base_df.index,
                                   columns=base_df.columns)
            tag = "norm"
        else:
            df_plot = base_df
            tag = "raw"

        # --------------------- 绘图 --------------------- #
        fig, ax = plt.subplots(figsize=(11, 4))
        df_plot.plot(ax=ax)
        ax.set_title(f"{tag.capitalize()} " + "/".join(cols))
        ax.legend(fontsize=8, ncol=len(cols) // 3 + 1)

        # 生成/确保保存路径
        if save_path is None:
            name = f"{tag}_{'_'.join(cols)}.png"
            save_path = _auto_name(self.save_root, "series", name)
        else:
            save_path = _ensure_dir(save_path)

        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        # -------------------- 方案 A (更简单，但每次重 fit) --------------------
        # if normalized:
        #     tmp_scaler = StandardScaler() if isinstance(self.scaler, StandardScaler) else MinMaxScaler()
        #     df_plot = pd.DataFrame(tmp_scaler.fit_transform(base_df.values),
        #                            index=base_df.index,
        #                            columns=base_df.columns)


# ----------------------------- simple test -------------------------------- #
if __name__ == "__main__":
    csv = r"C:/my/硕士/科研数据/all_datasets/Noise/Noise_1000Hz_4896.csv"
    exp = DataExplorer(csv_path=csv,
                       save_root="data_fig",
                       start_tag="train", end_tag="test")

    exp.plot_stl("Wind", period=24, train_border=0.8, val_border=0.9)
    exp.plot_series(cols=["Noise"], normalized=False)
    exp.plot_series(cols=["Noise"], normalized=True)