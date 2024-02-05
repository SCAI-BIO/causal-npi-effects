from .causal_refuter import CausalRefuter
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Union, Optional, Tuple


class PropensityThresholdingTest(CausalRefuter):
    """
    Strictly speaking, this is not a refutation test because it does not create simulations with manipulated data.
    It only predicts T with the t_models from the causal model, and fails if the returned propensity is too close to 0 and 1.
    This can help to detect violations of the positivity assumption. However, note that the outputs of the t_models depend on
    the causal model specifications, and it must be decided from model to model which propensity scores are acceptable.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epsilon = kwargs.pop("epsilon", 0.1)
        try:
            self._model = kwargs.pop("model")
        except KeyError:
            raise KeyError("You must pass causal model to PropensityThresholdingTest.")

    def refute_estimate(
        self, show_progress_bar: bool = False, alpha: float = 0.01, **kwargs
    ) -> Tuple[
        Optional[Dict[str, Dict[str, pd.DataFrame]]],
        Optional[Dict[str, Dict[str, pd.DataFrame]]],
    ]:
        refute = perform_propensity_thresholding_test(
            X=self._X,
            Y=self._Y,
            T=self._T,
            X_eval=self._X_eval,
            Y_eval=self._Y_eval,
            T_eval=self._T_eval,
            epsilon=self._epsilon,
            model=self._model,
            show_progress_bar=show_progress_bar,
        )

        # create a summary tables similar to the other tests (significant if closer to 0 or 1 than allowed by epsilon)
        # training data
        if type(refute[0]) != dict:
            summary_tables_all_treatments = None
        else:
            summary_tables_all_treatments = {k: {} for k in refute[0].keys()}
            for k in refute[0].keys():
                summary_tables_all_treatments[k][
                    self._model.config["outcomes"][-1]
                ] = pd.DataFrame(
                    {
                        "point_estimate": np.full(refute[0][k].shape, np.nan),
                        "mean": np.full(refute[0][k].shape, np.nan),
                        "ci_lower": np.clip(refute[0][k], None, 0.5),
                        "ci_upper": np.clip(refute[0][k], 0.5, None),
                        "significant": (refute[0][k] < self._epsilon)
                        | (refute[0][k] > (1 - self._epsilon)),
                    },
                    index=refute[0][k].index,
                )

        # evaluation data
        if type(refute[1]) != dict:
            summary_tables_all_treatments_eval = None
        else:
            summary_tables_all_treatments_eval = {k: {} for k in refute[1].keys()}
            for k in refute[1].keys():
                summary_tables_all_treatments_eval[k][
                    self._model.config["outcomes"][-1]
                ] = pd.DataFrame(
                    {
                        "point_estimate": np.full(refute[1][k].shape, np.nan),
                        "mean": np.full(refute[1][k].shape, np.nan),
                        "ci_lower": np.clip(refute[1][k], None, 0.5),
                        "ci_upper": np.clip(refute[1][k], 0.5, None),
                        "significant": (refute[1][k] < self._epsilon)
                        | (refute[1][k] > (1 - self._epsilon)),
                    },
                    index=refute[1][k].index,
                )

        return summary_tables_all_treatments, summary_tables_all_treatments_eval


def _refute_once(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    T: pd.Series,
    model,
    X_eval: pd.DataFrame = pd.DataFrame(),
    Y_eval: pd.DataFrame = pd.DataFrame(),
    T_eval: Union[pd.DataFrame, pd.Series] = pd.DataFrame(),
) -> Tuple[Optional[Dict[str, pd.Series]], Tuple[Dict[str, pd.Series]]]:
    # account for models that fit base learners fold-wise
    propensities_concat = dict.fromkeys(model.t_models.keys())
    propensities_concat_eval = dict.fromkeys(model.t_models.keys())

    for x, t, concat in zip(
        [X, X_eval], [T, T_eval], [propensities_concat, propensities_concat_eval]
    ):
        fold_numbers = pd.Series(np.ones(len(t)), index=t.index)
        for t_level in model.t_models.keys():
            concat[t_level] = pd.Series(
                model.t_models[t_level].predict_proba(x)[..., -1],
                index=fold_numbers.index,
            )
    return propensities_concat, propensities_concat_eval


def perform_propensity_thresholding_test(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    T: pd.Series,
    epsilon: float,
    model,
    show_progress_bar: bool = True,
    X_eval: pd.DataFrame = pd.DataFrame(),
    Y_eval: pd.DataFrame = pd.DataFrame(),
    T_eval: Union[pd.DataFrame, pd.Series] = pd.DataFrame(),
) -> Tuple[Optional[Dict[str, pd.Series]], Tuple[Dict[str, pd.Series]]]:
    # works only if causal model has at least one t_model
    assert (
        len(model.t_models[list(model.t_models.keys())[-1]]) > 0
    ), "PropensityThresholdingTest works only for models with at least one t_model, {} has none.".format(
        model.__class__.__name__
    )
    # works only for categorical treatments at the moment
    for t_model in model.t_models.values():
        if type(t_model) == list:
            t_model = t_model[0]
        assert hasattr(
            t_model, "predict_proba"
        ), "PropensityThresholdingTest works only with categorical treatment at the moment."

    # Run test (no iterations, but output bar to make this comparable to other refutations tests)
    sample_estimates = None
    for _ in tqdm(
        range(1),
        colour=CausalRefuter.PROGRESS_BAR_COLOR,
        disable=not show_progress_bar,
        desc="Performing propensity thresholding test (epsilon={}): ".format(epsilon),
    ):
        sample_estimates = _refute_once(
            X=X, Y=Y, T=T, X_eval=X_eval, Y_eval=Y_eval, T_eval=T_eval, model=model
        )
    if sample_estimates == None:
        raise RuntimeError("PropensityThresholdingTest returned None.")
    return sample_estimates
