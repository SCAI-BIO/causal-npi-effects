from .causal_refuter import CausalRefuter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from copy import deepcopy
from src.causal_models import *
from typing import Dict, Union, Optional, List, Any, Tuple


class RandomCommonCauseRefuter(CausalRefuter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_common_cause_type = kwargs.pop(
            "random_commom_cause_type", "random walk"
        )
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
        refute = refute_random_common_cause(
            X=self._X,
            Y=self._Y,
            T=self._T,
            X_eval=self._X_eval,
            Y_eval=self._Y_eval,
            T_eval=self._T_eval,
            causal_model_class=self._causal_model_class,
            causal_model_config=self._causal_model_config,
            num_simulations=self._num_simulations,
            random_common_cause_type=self._random_common_cause_type,
            random_state=self._random_state,
            show_progress_bar=show_progress_bar,
            n_jobs=self._n_jobs,
            validation_start=validation_start,
        )

        # create summary tables for all treatmens for the training data
        summary_tables_all_treatments = self._summmary_tables_all_treatments(
            alpha=alpha, refute=refute, refute_index=0, estimate=self._estimate
        )

        # create summary tables for all treatmens for the evaluation data
        summary_tables_all_treatments_eval = self._summmary_tables_all_treatments(
            alpha=alpha, refute=refute, refute_index=1, estimate=self._estimate_eval
        )

        return summary_tables_all_treatments, summary_tables_all_treatments_eval


def _refute_once(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    T: pd.Series,
    random_common_cause_type: str,
    est_class: str,
    est_config: Dict[str, Any],
    random_state: Optional[np.random.RandomState] = None,
    validation_start: Optional[str] = None,
    X_eval: pd.DataFrame = pd.DataFrame(),
    Y_eval: pd.DataFrame = pd.DataFrame(),
    T_eval: Union[pd.DataFrame, pd.Series] = pd.DataFrame(),
) -> Optional[Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]]:
    # add new confounder to both X and X_eval
    X_all = pd.concat([X, X_eval])
    max_window_size = len(
        X_all.loc[:, X_all.columns.get_level_values("step") != "static"]
        .columns.get_level_values("step")
        .unique()
    )
    # get new X according to random_common_cause_type
    if random_common_cause_type == "random walk":
        if random_state is None:
            random_state = np.random.RandomState(np.random.randint(100))

        rs_copies = [
            deepcopy(random_state) for _ in range(len(X_all.index.levels[0].unique()))
        ]
        w_random = []

        # for each region, create a slightly different random walk and append shifts
        for i, region in enumerate(X_all.index.levels[0].unique()):
            size = len(X_all.loc[region]) + (max_window_size - 1)
            walk = np.cumsum(
                rs_copies[i].normal(0, 1, size)
                + random_state.normal(size=size, scale=0.5)
            )

            if max_window_size > 1:
                walk_df = pd.DataFrame(
                    walk[: -max_window_size + 1], index=X_all.loc[region].index
                )
                walk_df.columns = pd.MultiIndex.from_product(
                    [["random_w"], ["t"]], names=["predictor", "step"]
                )
                shifted_walk_list = []
                for s in range(1, max_window_size):
                    shifted_walk = pd.DataFrame(
                        np.roll(walk, s)[: -max_window_size + 1],
                        index=X_all.loc[region].index,
                    )
                    shifted_walk.columns = pd.MultiIndex.from_product(
                        [["random_w"], ["t-{}".format(s)]], names=["predictor", "step"]
                    )
                    shifted_walk_list.append(shifted_walk)

                walk_concat = pd.concat([walk_df] + shifted_walk_list, axis=1)
                walk_concat = walk_concat.sort_index(
                    axis=1, level=[0, 1], ascending=[True, False]
                )
            else:
                walk_concat = pd.DataFrame(walk, index=X_all.loc[region].index)
                walk_concat.columns = pd.MultiIndex.from_product(
                    [["random_w"], ["t"]], names=["predictor", "step"]
                )

            walk_concat.index = pd.MultiIndex.from_product(
                [[region], X_all.loc[region].index], names=["geo_id", "time_id"]
            )
            w_random.append(walk_concat)

        w_random = pd.concat(w_random)

    else:
        # independent random values
        if random_state is None:
            w_random = pd.DataFrame(np.random.randn(X_all.shape[0]), index=X_all.index)
            w_random.columns = pd.MultiIndex.from_product(
                [["random_w"], ["static"]], names=["predictor", "step"]
            )
        else:
            w_random = pd.DataFrame(random_state(X_all.shape[0]), index=X_all.index)
            w_random.columns = pd.MultiIndex.from_product(
                [["random_w"], ["static"]], names=["predictor", "step"]
            )

    X_new = X_all.join(w_random)
    est_config["input_size"] += 1
    new_estimator = eval(est_class)(config=est_config)

    try:
        new_estimator.fit(
            X_new.loc[X.index],
            Y,
            T,
            skip_checkpoint=True,
            validation_start=validation_start,
        )
        # CATE
        cate_train = new_estimator.cate(X=X_new.loc[X.index], T=T)
        if not X_eval.empty:
            cate_eval = new_estimator.cate(X=X_new.loc[X_eval.index], T=T_eval)
        else:
            cate_eval = None
        return cate_train, cate_eval
    except Exception as e:
        logging.error("Delayed iteration stopped with exception: {}".format(e))
        return None


def refute_random_common_cause(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    T: pd.Series,
    random_common_cause_type: str,
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
            random_common_cause_type=random_common_cause_type,
            est_class=causal_model_class,
            est_config=causal_model_config,
            random_state=random_state,
            validation_start=validation_start,
        )
        for _ in tqdm(
            range(num_simulations),
            colour=CausalRefuter.PROGRESS_BAR_COLOR,
            disable=not show_progress_bar,
            desc="Refuting estimates with random common cause ({}): ".format(
                random_common_cause_type
            ),
        )
    )
    if sample_estimates is None:
        raise RuntimeError(
            "Parallel execution of RandomCommonCauseRefuter returned None."
        )
    sample_estimates = [i for i in sample_estimates if i is not None]
    return sample_estimates
