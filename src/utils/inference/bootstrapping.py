from arch.bootstrap import (
    IIDBootstrap,
    MovingBlockBootstrap,
    CircularBlockBootstrap,
    StationaryBootstrap,
)
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from functools import wraps
from joblib import Parallel, delayed
from copy import deepcopy
from tqdm import tqdm
from typing import List, Dict, Tuple, Generator, Union


class OuterBootstrap:
    """Bootstrap regions (by geo_id)"""

    def __init__(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        T: pd.Series,
        exclude: List[str] = ["r_number"],
        block_size: int = 21,
    ):
        self.X = X
        self.Y = Y
        self.T = T
        self.exclude = exclude
        # bootstrap regions
        self.bs_regions = IIDBootstrap(x=self.X.index.levels[0].unique().to_numpy())
        # bootstrap within regions
        self.bs_time_series = dict.fromkeys(self.X.index.levels[0].unique())
        for r in self.bs_time_series.keys():
            self.bs_time_series[r] = TimeSeriesBootstrap(
                self.X.loc[
                    r,
                    (~self.X.columns.get_level_values("predictor").isin(self.exclude))
                    & (self.X.columns.get_level_values("step") != "static"),
                ],
                block_size=block_size,
            ).bootstrap(10000)

    def bootstrap(
        self, reps: int
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Series], None, None]:
        for i, (_, regions) in enumerate(self.bs_regions.bootstrap(reps)):
            X_list, Y_list, T_list = [], [], []
            for r in regions["x"]:
                # copy df for r (region last drawn with bootstrapping)
                X_r = self.X.loc[r].copy()
                Y_r = self.Y.loc[r].copy()
                T_r = self.T.loc[r].copy()
                # now bootstrap all time series within this region
                X_r.loc[
                    :,
                    (~self.X.columns.get_level_values("predictor").isin(self.exclude))
                    & (self.X.columns.get_level_values("step") != "static"),
                ] = next(self.bs_time_series[r])
                X_r.index = pd.MultiIndex.from_product(
                    [["{:03d}".format(len(X_list))], list(X_r.index)],
                    names=["geo_id", "time_id"],
                )
                Y_r.index = pd.MultiIndex.from_product(
                    [["{:03d}".format(len(X_list))], list(Y_r.index)],
                    names=["geo_id", "time_id"],
                )
                T_r.index = pd.MultiIndex.from_product(
                    [["{:03d}".format(len(X_list))], list(T_r.index)],
                    names=["geo_id", "time_id"],
                )
                X_list.append(X_r)
                Y_list.append(Y_r)
                T_list.append(T_r)
            # concat all bootstrapped regional timeseries
            resampled_X = pd.concat(X_list, axis=0)
            resampled_Y = pd.concat(Y_list, axis=0)
            resampled_T = pd.concat(T_list, axis=0)
            yield resampled_X, resampled_Y, resampled_T
            if i == reps - 1:
                break


class TimeSeriesBootstrap:
    """Blocked bootstrap on residual parts from seasonal-trend decomposition using LOESS"""

    def __init__(
        self, df: pd.DataFrame, block_size: int = 21, bs_method: str = "circular"
    ) -> None:
        if bs_method == "moving":
            BS = MovingBlockBootstrap
        elif bs_method == "circular":
            BS = CircularBlockBootstrap
        elif bs_method == "stationary":
            BS = StationaryBootstrap
        else:
            raise ValueError(
                "bs_method for TimeSeriesBootstrap must be 'moving', 'circular' or 'stationary'."
            )

        # Box-Cox tranformation
        # df = self.boxcox_transformation(df)
        # perform STL on each time series
        self.decomposed = df.apply(
            lambda col: self.seasonal_decomposition(col, df.index), axis=0
        )
        self.time_series_bs = [None] * len(self.decomposed)
        # bootstrapper for the residuals of each time series (seed ensures that blocks are consistent)
        for i in range(len(self.decomposed)):
            self.time_series_bs[i] = BS(
                block_size, self.decomposed[i].resid, seed=0
            ).bootstrap(10000)

    def seasonal_decomposition(self, s, index):
        old_index = s.index
        s.index = index
        # trend paramters with minimal smoothing
        if pd.infer_freq(old_index).startswith("W"):
            trend = 53
        else:
            trend = 9
        # season-trend decomposition
        stl = STL(s, trend=trend).fit()
        # import matplotlib.pyplot as plt
        # stl.plot()
        # plt.show()
        stl.resid.index = stl.trend.index = stl.seasonal.index = old_index
        return stl

    # def boxcox_transformation(self, df):
    #     self.min_values = df.min()
    #     for col in self.min_values.index:
    #         if self.min_values[col] < 0:
    #             df[col] += np.abs(self.min_values[col]) + 1
    #     self.lambdas = df.apply(lambda col: boxcox(col)[1])
    #     return df.apply(lambda col: boxcox(col)[0])

    # def inv_boxcox_transformation(self, df):
    #     for col in self.min_values.index:
    #         df[col] = inv_boxcox(df[col], self.lambdas[col])
    #         if self.min_values[col] < 0:
    #             df[col] +=  + self.min_values[col]
    #     return df

    def bootstrap(self, reps):
        for i, samples in enumerate(zip(*self.time_series_bs)):
            columns = []
            # bootstrap the residual parts of time series, than add trend and seasonal part back
            for j in range(len(samples)):
                sample = samples[j][0][0]
                sample.index = self.decomposed[0].trend.index
                columns.append(
                    (
                        sample + self.decomposed[j].seasonal + self.decomposed[j].trend
                    ).rename(self.decomposed.keys()[j])
                )

            resampled_df = pd.concat(columns, axis=1)
            yield resampled_df  # self.inv_boxcox_transformation(resampled_df) # resampled_df
            if i == reps - 1:
                break


class StaticBootstrap:
    """Simple standard bootstrap (not advisable for time series data)"""

    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, T: pd.Series) -> None:
        self.X = X
        self.Y = Y
        self.T = T

    def bootstrap(
        self, reps: int
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.Series], None, None]:
        for _ in range(reps):
            random_idx = np.random.randint(0, len(self.X), len(self.X))
            yield self.X.iloc[random_idx, :], self.Y.iloc[random_idx, :], self.T.iloc[
                random_idx
            ]


class BootstrapInference:
    PROGRESS_BAR_COLOR = "cyan"

    def __init__(
        self,
        wrapped,
        n_samples: int = 100,
        n_jobs: int = -1,
        show_progress_bar: bool = True,
    ) -> None:
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.show_progress_bar = show_progress_bar
        self.wrapped = wrapped
        self._bootstrap_instances = [deepcopy(wrapped) for _ in range(n_samples)]
        self.config = wrapped.config

    def get_outcome_distribution(
        self, X: pd.DataFrame, T: Union[pd.DataFrame, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """Uses bootstrapping to make self.n_samples outcome predicions for given X and T

        Parameters
        ----------
        X : pd.DataFrame
            Formatted confounders / effect modifiers
        T : Union[pd.DataFrame, pd.Series]
            Formatted treatment(s)

        Returns
        -------
        Dict[str, np.ndarray]
            self.n_samples outcome predictions from bootstrapping for each outcome
        """
        # use variational dropout to make n_samples outcome predicions for given X and T
        Y_pred_dict = {k: [] for k in self.wrapped.y_models.keys()}
        for m in self._bootstrap_instances:
            for k, y_model in m.y_models.items():
                Y_pred_dict[k].append(y_model.predict(X=X, T=T))
        return {k: np.stack(Y_pred_dict[k], -1) for k in Y_pred_dict.keys()}

    def get_cate_distribution(
        self,
        X: pd.DataFrame,
        levels: List[str],
        T: Union[pd.Series, pd.DataFrame] = pd.DataFrame(),
    ) -> Dict[str, np.ndarray]:
        """Uses bootstrapping to make self.n_samples CATE predicions for given X and T

        Parameters
        ----------
        X : pd.DataFrame
            Formatted confounders / effect modifiers
        T : Union[pd.DataFrame, pd.Series]
            Formatted treatment(s)

        Returns
        -------
        Dict[str, np.ndarray]
            self.n_samples CATE predictions from bootstrapping for each outcome
        """
        cate_dict = {k: [] for k in levels}
        for m in self._bootstrap_instances:
            cate = m.cate(X=X, T=T)
            for k in cate.keys():
                cate_dict[k].append(cate[k])
        return {k: np.stack(cate_dict[k], -1) for k in cate_dict.keys()}


class StaticBootstrapInference(BootstrapInference):
    """Encapsulates fitting of static bootstrap instances."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _fit_one_estimator(
        model, X: pd.DataFrame, Y: pd.DataFrame, T: pd.Series, **kwargs
    ):
        model.fit(X, Y, T, **kwargs, skip_checkpoint=True)
        return model

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, T: pd.Series, **kwargs):
        # df = pd.concat([X,Y,T], axis=1)
        bs = StaticBootstrap(X, Y, T)
        self._bootstrap_instances = Parallel(
            n_jobs=self.n_jobs, backend="loky", prefer="processes"
        )(
            delayed(self._fit_one_estimator)(
                obj, *next(bs.bootstrap(self.n_samples)), **kwargs
            )
            for obj in tqdm(
                self._bootstrap_instances,
                colour=BootstrapInference.PROGRESS_BAR_COLOR,
                disable=not self.show_progress_bar,
                desc="Bootstrapping (static): ",
            )
        )
        return self


class TimeSeriesBootstrapInference(BootstrapInference):
    """Encapsulates fitting of time series bootstrap instances."""

    def __init__(self, *args, **kwargs) -> None:
        self.block_size = kwargs.pop("block_size", 28)
        self.exclude = kwargs.pop("exclude", ["r_number"])
        super().__init__(*args, **kwargs)

    @staticmethod
    def _fit_one_estimator(
        model, X: pd.DataFrame, Y: pd.DataFrame, T: pd.Series, **kwargs
    ):
        model.fit(X, Y, T, **kwargs, skip_checkpoint=True)
        return model

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, T: pd.Series, **kwargs):
        bs = OuterBootstrap(X, Y, T, block_size=self.block_size, exclude=self.exclude)
        self._bootstrap_instances = Parallel(
            n_jobs=self.n_jobs, backend="loky", prefer="processes"
        )(
            delayed(self._fit_one_estimator)(
                obj, *next(bs.bootstrap(self.n_samples)), **kwargs
            )
            for obj in tqdm(
                self._bootstrap_instances,
                colour=BootstrapInference.PROGRESS_BAR_COLOR,
                disable=not self.show_progress_bar,
                desc="Bootstrapping (time series): ",
            )
        )
        return self
