from .causal_refuter import CausalRefuter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from src.causal_models import *
from typing import Dict, Union, Optional, List, Any, Tuple


class DataSubsetRefuter(CausalRefuter):
    """This refutation test randomly subsamples a defined portion of fitting / evaluation window pairs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._subset_fraction = kwargs.pop("subset_fraction", 0.8)
        self._num_simulations = kwargs.pop(
            "num_simulations", CausalRefuter.DEFAULT_NUM_SIMULATIONS
        )
        self._random_state = kwargs.pop("random_state", None)

    def refute_estimate(
        self,
        show_progress_bar: bool = False,
        alpha: float = 0.01,
        validation_start: Optional[str] = None,
    ) -> Tuple[
        Optional[Dict[str, Dict[str, pd.DataFrame]]],
        Optional[Dict[str, Dict[str, pd.DataFrame]]],
    ]:
        refute = refute_data_subset(
            X=self._X,
            Y=self._Y,
            T=self._T,
            X_eval=self._X_eval,
            Y_eval=self._Y_eval,
            T_eval=self._T_eval,
            causal_model_class=self._causal_model_class,
            causal_model_config=self._causal_model_config,
            num_simulations=self._num_simulations,
            subset_fraction=self._subset_fraction,
            random_state=self._random_state,
            show_progress_bar=show_progress_bar,
            n_jobs=self._n_jobs,
            validation_start=validation_start,
        )

        # create summary tables for all treatments for the training data
        summary_tables_all_treatments = self._summmary_tables_all_treatments(
            alpha=alpha, refute=refute, refute_index=0, estimate=self._estimate
        )

        # create summary tables for all treatments for the evaluation data
        summary_tables_all_treatments_eval = self._summmary_tables_all_treatments(
            alpha=alpha, refute=refute, refute_index=1, estimate=self._estimate_eval
        )

        return summary_tables_all_treatments, summary_tables_all_treatments_eval


def _refute_once(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    T: Union[pd.Series, pd.DataFrame],
    subset_fraction: float,
    est_class: str,
    est_config: Dict[str, Any],
    random_state: Optional[np.random.RandomState] = None,
    validation_start: Optional[str] = None,
    X_eval: pd.DataFrame = pd.DataFrame(),
    Y_eval: pd.DataFrame = pd.DataFrame(),
    T_eval: Union[pd.DataFrame, pd.Series] = pd.DataFrame(),
) -> Optional[Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]]:
    # create mask and add to estimator (but make sure that same time points are sampled for each region)
    max_len = X.groupby("geo_id").count().max().max()
    if random_state is None:
        mask_one_region = np.random.choice(
            [True, False], max_len, p=[subset_fraction, 1 - subset_fraction]
        ).tolist()
    else:
        mask_one_region = random_state.choice(
            [True, False], max_len, p=[subset_fraction, 1 - subset_fraction]
        ).tolist()

    mask = []
    for region in X.index.get_level_values("geo_id").unique():
        mask.extend(mask_one_region[-len(X.loc[region]) :])
    mask = pd.Series(mask, index=X.index)
    new_estimator = eval(est_class)(config=est_config)

    try:
        new_estimator.fit(
            X[mask],
            Y[mask],
            T[mask],
            skip_checkpoint=True,
            validation_start=validation_start,
        )

        cate_train = new_estimator.cate(X=X, T=T)

        if not X_eval.empty:
            cate_eval = new_estimator.cate(X=X_eval, T=T_eval)
        else:
            cate_eval = None

        return cate_train, cate_eval

    except Exception as e:
        logging.error("Delayed iteration stopped with exception: {}".format(e))
        return None


def refute_data_subset(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    T: pd.Series,
    subset_fraction: float,
    causal_model_class: str,
    causal_model_config: Dict[str, Any],
    num_simulations: int,
    random_state: Optional[Union[np.random.RandomState, int]] = None,
    show_progress_bar: bool = True,
    n_jobs: int = -1,
    validation_start: Optional[str] = None,
    X_eval: pd.DataFrame = pd.DataFrame(),
    Y_eval: pd.DataFrame = pd.DataFrame(),
    T_eval: Union[pd.DataFrame, pd.Series] = pd.DataFrame(),
) -> List[Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]]:
    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    # Run refutation in parallel
    sample_estimates = Parallel(
        n_jobs=n_jobs, backend="loky", prefer="processes", timeout=10000
    )(
        delayed(_refute_once)(
            X=X,
            Y=Y,
            T=T,
            X_eval=X_eval,
            Y_eval=Y_eval,
            T_eval=T_eval,
            subset_fraction=subset_fraction,
            est_class=causal_model_class,
            est_config=causal_model_config,
            random_state=random_state,
            validation_start=validation_start,
        )
        for _ in tqdm(
            range(num_simulations),
            colour=CausalRefuter.PROGRESS_BAR_COLOR,
            disable=not show_progress_bar,
            desc="Refuting estimates with random subset (subset_fraction={}): ".format(
                subset_fraction
            ),
        )
    )
    if sample_estimates is None:
        raise RuntimeError("Parallel execution of DataSubsetRefuter returned None.")
    sample_estimates = [i for i in sample_estimates if i is not None]
    return sample_estimates
