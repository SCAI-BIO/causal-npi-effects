from .causal_refuter import CausalRefuter
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from src.causal_models import *
from typing import Dict, Optional, Any, Union, List, Tuple


class PlaceboTreatmentRefuter(CausalRefuter):
    """This refutation test manipulates the treatment by random permutation or cycle shifting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._placebo_type = kwargs.pop("placebo_type", "cyclic")
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
        refute = refute_placebo_treatment(
            X=self._X,
            Y=self._Y,
            T=self._T,
            X_eval=self._X_eval,
            Y_eval=self._Y_eval,
            T_eval=self._T_eval,
            causal_model_class=self._causal_model_class,
            causal_model_config=self._causal_model_config,
            num_simulations=self._num_simulations,
            placebo_type=self._placebo_type,
            random_state=self._random_state,
            show_progress_bar=show_progress_bar,
            n_jobs=self._n_jobs,
            validation_start=validation_start,
        )

        # set estimate to zero because we want to check if zero is contained in the simulated distribution
        zeros = {}
        for k in self._estimate.keys():
            zeros[k] = pd.DataFrame(
                np.zeros(self._estimate[k].shape), index=self._estimate[k].index
            )
        if self._estimate_eval is not None:
            zeros_eval = {}
            for k in self._estimate.keys():
                zeros_eval[k] = pd.DataFrame(
                    np.zeros(self._estimate_eval[k].shape),
                    index=self._estimate_eval[k].index,
                )
        else:
            zeros_eval = None

        # create summary tables for all treatmens for the training data
        summary_tables_all_treatments = self._summmary_tables_all_treatments(
            alpha=alpha, refute=refute, refute_index=0, estimate=zeros
        )

        # create summary tables for all treatmens for the evaluation data
        summary_tables_all_treatments_eval = self._summmary_tables_all_treatments(
            alpha=alpha, refute=refute, refute_index=1, estimate=zeros_eval
        )

        return summary_tables_all_treatments, summary_tables_all_treatments_eval


def _refute_once(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    T: Union[pd.DataFrame, pd.Series],
    placebo_type: str,
    est_class: str,
    est_config: Dict[str, Any],
    random_state: Optional[np.random.RandomState] = None,
    validation_start: Optional[str] = None,
    X_eval: pd.DataFrame = pd.DataFrame(),
    Y_eval: pd.DataFrame = pd.DataFrame(),
    T_eval: Union[pd.DataFrame, pd.Series] = pd.Series(dtype=object),
) -> Optional[Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]]:
    new_estimator = None
    # manipulate treatment for both T and T_eval
    T_all = pd.concat([T, T_eval])
    # get new T according to placebo_type
    if placebo_type == "permute":
        if random_state is None:
            permuted_idx = np.random.choice(
                T_all.shape[0], size=T_all.shape[0], replace=False
            )
        else:
            permuted_idx = random_state.choice(
                T_all.shape[0], size=T_all.shape[0], replace=False
            )
        T_new = T_all.iloc[permuted_idx]
    elif placebo_type == "cyclic":
        max_shift = T_all.groupby(level=0).count().max()
        if random_state is None:
            shift = np.random.randint(1, max_shift)
        else:
            shift = random_state.randint(1, max_shift)
        # apply cyclic shift to every region separately
        if type(T_all) == pd.DataFrame:
            # for several treatments in one df (shift all together)
            T_new = T_all.groupby(level=0).apply(
                lambda region: pd.DataFrame(np.roll(region, shift, axis=0))
            )
            T_new.index = T_all.index
            T_new.columns = T_all.columns
        else:
            # for one treatment in a series
            T_new = T_all.groupby(level=0).apply(
                lambda region: pd.Series(np.roll(region, shift))
            )
            T_new.index = T_all.index
    else:
        raise ValueError(
            "Non-valid value given for placebo_type option (must be 'permute', 'Random Data' or ' cyclic)"
        )

    new_estimator = eval(est_class)(config=est_config)

    try:
        new_estimator.fit(
            X,
            Y,
            T_new.loc[T.index],
            skip_checkpoint=True,
            validation_start=validation_start,
        )
        # CATE
        cate_train = new_estimator.cate(X=X, T=T_new.loc[T.index])
        if not X_eval.empty:
            cate_eval = new_estimator.cate(X=X_eval, T=T_new.loc[T_eval.index])
        else:
            cate_eval = None
        return cate_train, cate_eval
    except Exception as e:
        logging.error("Delayed iteration stopped with exception: {}".format(e))
        return None


def refute_placebo_treatment(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    T: pd.Series,
    placebo_type: str,
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
            placebo_type=placebo_type,
            est_class=causal_model_class,
            est_config=causal_model_config,
            random_state=random_state,
            validation_start=validation_start,
        )
        for _ in tqdm(
            range(num_simulations),
            colour=CausalRefuter.PROGRESS_BAR_COLOR,
            disable=not show_progress_bar,
            desc="Refuting estimates with placebo treatment ({}): ".format(
                placebo_type
            ),
        )
    )
    if sample_estimates is None:
        raise RuntimeError(
            "Parallel execution of PlaceboTreatmentRefuter returned None."
        )
    sample_estimates = [i for i in sample_estimates if i is not None and len(i) != 0]
    return sample_estimates
