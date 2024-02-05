from functools import wraps
import numpy as np
import pandas as pd
import pickle
import logging
import shap
import os
import datetime
from copy import copy
from shap import maskers
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from src.utils.inference.bootstrapping import BootstrapInference
from src.utils import (
    TimeSeriesBootstrapInference,
    StaticBootstrapInference,
    VariationalDropoutInference,
    nonparametric_significance_test,
    aggregate_shap_values,
    checkpoint,
    create_windowed_dataframe,
)
from src.utils.refutation import *
from src.utils.plotting import plot_refutation_results, plot_shap, plot_cate

from .encoders import RepresentationEncoder

# Functionalities shared by the causal models are implemented in parent class CausalModel


def inference(m: Callable[..., None]) -> Callable:
    """Decorator that fits inference object (needed to quantify model uncertainty) before calling the wrapped fit function.

    Parameters
    ----------
    m : Callable[..., None]
        Fit function of a CausalModel.

    Returns
    -------
    Callable
        Included fitting of inference object.
    """

    @wraps(m)
    def call(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        T: pd.Series,
        inference: Optional[str] = None,
        inference_config: Dict = {
            "n_jobs": -1,
            "n_samples": 100,
        },
        **kwargs,
    ) -> None:
        if inference is not None:
            # first call inference fit method
            logging.info(f"[fit_causal_model] Using {inference} for inference")
            inference_obj = self._get_inference(inference, inference_config)
            inference_obj.fit(X=X, Y=Y, T=T, **kwargs)
            self._inference_object = inference_obj

        # finally call main fit method (has to be in this order to circumvent an index error that occurs otherwise)
        m(self, X=X, Y=Y, T=T, **kwargs)
        self.post_fit()

    return call


class CausalModel:
    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs,
    ) -> None:
        self.config = config
        if "val_split" in config.keys():
            self.val_split = config["val_split"]
        else:
            self.val_split = True

        # some elements are initialized now but will be assigned later
        self._inference_object: Optional[BootstrapInference] = None
        self.T_levels: List[str]
        self.save_to: Optional[str] = None
        self.deactivate_plotting: bool = False
        self.fitted = False
        self.encoder: RepresentationEncoder

    def _get_inference(
        self, inference: str, inference_config: Dict
    ) -> Optional[Union[BootstrapInference, VariationalDropoutInference]]:
        """Initializes inference object according to given configuration.

        Parameters
        ----------
        inference : str
            Name of requested inference object [static_bootstrap, time_series_bootstrap, variational_dropout]
        inference_config : Dict
            Dict with inference type-specific configuration.

        Returns
        -------
        Optional[Union[BootstrapInference, VariationalDropoutInference]]
            Inference object.
        """
        if self._inference_object is not None:
            return self._inference_object
        elif inference == "time_series_bootstrap":
            return TimeSeriesBootstrapInference(
                self,
                inference_config["n_samples"],
                inference_config["n_jobs"],
                block_size=inference_config["block_size"]
                if "block_size" in inference_config.keys()
                else 28,
                exclude=inference_config["exclude"]
                if "exclude" in inference_config.keys()
                else [],
            )
        elif inference == "static_bootstrap":
            return StaticBootstrapInference(
                self,
                inference_config["n_samples"],
                inference_config["n_jobs"],
            )
        elif inference == "variational_dropout":
            return VariationalDropoutInference(
                wrapped=self, n_samples=inference_config["n_samples"]
            )
        else:
            return

    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        T: pd.Series,
        X_val: pd.DataFrame = pd.DataFrame(),
        Y_val: pd.DataFrame = pd.DataFrame(),
        T_val: pd.Series = pd.Series(dtype=object),
        **kwargs,
    ) -> None:
        """
        This method should fit causal model with validation set
        """
        raise NotImplementedError("Abstract parent class")

    def post_fit(self) -> None:
        """
        After fitting: Saves model (if self.save_to is set) and set self.fitted
        """
        self.fitted = True
        if self.save_to is not None:
            # save balanced representations
            if self.encoder.save_br_callback is not None and self.save_to is not None:
                with open(f"{self.save_to}/br", "wb") as handle:
                    pickle.dump(self.encoder.save_br_callback.br_dict, handle)

            if self.config["pickle_model"] and not self.config["pickle_inference"]:
                # do not pickle inference object because this may need a lot of memory
                model_without_inference = copy(self)
                model_without_inference._inference_object = None
                with open("{}/fitted_model.pkl".format(self.save_to), "wb") as outp:
                    pickle.dump(model_without_inference, outp)
            elif self.config["pickle_model"]:
                with open("{}/fitted_model.pkl".format(self.save_to), "wb") as outp:
                    pickle.dump(self, outp)
            else:
                return

    def cate(self, X: pd.DataFrame, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        A method that calculates CATE values with the fitted models
        """
        raise NotImplementedError("Abstract parent class")

    @staticmethod
    def format_model_inputs(
        *args, **kwargs
    ) -> Tuple[Union[pd.Series, pd.DataFrame], ...]:
        """Takes input data and formats it according to model-specific requirements.

        Returns
        -------
        Tuple[Union[pd.Series, pd.DataFrame], ...]
            Tuple of formatted data (X, Y, T)
        """
        windowed_data = create_windowed_dataframe(*args, **kwargs)
        if "treatment" in kwargs.keys():
            if kwargs["df"] is not None:
                assert (
                    windowed_data[-1].shape[1] == 1
                ), "Only one treatment is supported for this causal model at the moment. You passed {}".format(
                    kwargs["treatment"]
                )
                # make sure that T is a series
                return *windowed_data[:-1], windowed_data[-1].iloc[:, 0]
            else:
                # X and Y are empty data frames, T is an empty series
                return *windowed_data[:-1], pd.Series(dtype=object)
        return windowed_data

    def effect_interval(
        self,
        X: pd.DataFrame,
        T: Union[pd.Series, pd.DataFrame] = pd.DataFrame(),
        alpha: float = 0.01,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """_summary_

        Parameters
        ----------
        X : pd.DataFrame
            Formatted confounders / effect modifiers
        T : Union[pd.Series, pd.DataFrame], optional
            Formatted treatment, by default pd.DataFrame()
        alpha : float, optional
            Significance level to test if estimate < 0, by default 0.01

        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Dictionary with point and inverval estimates per outcome
        """
        summary_dict = {}
        if self._inference_object is not None:
            cate_dist = self._inference_object.get_cate_distribution(
                X, levels=self.T_levels[1:], T=T
            )
            point_estimates = self.cate(X=X, T=T)

            for k in cate_dist.keys():
                mean = cate_dist[k].mean(axis=-1)
                ci_lower = np.quantile(cate_dist[k], q=alpha / 2, axis=-1)
                ci_upper = np.quantile(cate_dist[k], q=1 - alpha / 2, axis=-1)

                pvalue = nonparametric_significance_test(
                    point_estimates[k].values,
                    cate_dist[k],
                    shift_to_value=0,
                    tailed="left",
                )

                summary_tables = {
                    o: pd.DataFrame(
                        {
                            "point_estimate": point_estimates[k].iloc[:, i],
                            "mean": mean[..., i],
                            "ci_lower": ci_lower[..., i],
                            "ci_upper": ci_upper[..., i],
                            "pvalue": pvalue[..., i],
                            "significant": pvalue[..., i] < alpha / 2,
                        },
                        index=point_estimates[k].index,
                    )
                    for i, o in enumerate(self.config["outcomes"])
                }
                summary_dict[k] = summary_tables

        return summary_dict

    @checkpoint(
        step_name="evaluate_causal_model",
        message="Evaluating CATE predictions and creating plots...",
        subdir="cate",
    )
    def eval_and_plot_cate(
        self,
        X: pd.DataFrame = pd.DataFrame(),
        X_test: pd.DataFrame = pd.DataFrame(),
        T: Union[pd.Series, pd.DataFrame] = pd.Series(dtype=object),
        T_test: Union[pd.Series, pd.DataFrame] = pd.Series(dtype=object),
    ) -> Optional[Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
        if self._inference_object is None:
            logging.warning(
                "No inference object available in {}. Skipping CATE evaluation.".format(
                    self.__class__.__name__
                )
            )
            return

        effect_interval = {}
        for x, t, train_or_test in zip(
            [X, X_test], [T, T_test], ["01_training", "02_evaluation"]
        ):
            if x.empty:
                continue
            if not os.path.isdir(f"{self.save_to}/{train_or_test}"):
                os.makedirs(f"{self.save_to}/{train_or_test}")
            # evaluate CATE predictions
            effect_interval[train_or_test] = self.effect_interval(X=x, T=t)
            if self.save_to is not None:
                for i, treatment in enumerate(effect_interval[train_or_test].keys()):
                    # save to CSV
                    effect_interval[train_or_test][treatment][
                        self.config["outcomes"][-1]
                    ].to_csv(
                        "{}/cate_{}.csv".format(
                            f"{self.save_to}/{train_or_test}", treatment
                        )
                    )
                    if type(t) == pd.DataFrame:
                        t = t.iloc[:, i]

                    if not self.deactivate_plotting:
                        # plot per region
                        for region in x.index.get_level_values("geo_id").unique():
                            plot_cate(
                                effect_intervals=effect_interval[train_or_test],
                                treatment=treatment,
                                T=t,
                                outcome=self.config["outcomes"][-1],
                                region=region,
                                save_to=f"{self.save_to}/{train_or_test}",
                            )
        return effect_interval

    @checkpoint(step_name="refutation", message="Performing refutation tests...")
    def perform_refutation_tests(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        T: Union[pd.DataFrame, pd.Series],
        X_eval: pd.DataFrame,
        Y_eval: pd.DataFrame,
        T_eval: Union[pd.DataFrame, pd.Series],
        refutation_tests: Dict[str, Any],
        causal_model_class: str,
        causal_model_config: Dict[str, Any],
        alpha: float = 0.05,
        validation_start: Optional[str] = None,
    ) -> Tuple[
        List[Dict[str, Dict[str, pd.DataFrame]]],
        Optional[List[Dict[str, Dict[str, pd.DataFrame]]]],
    ]:
        """Peforms specified refutation tests.

        Parameters
        ----------
        X : pd.DataFrame
            Formatted confounders / effect modifiers to fit the model
        Y : pd.DataFrame
            Formatted outcome to fit the model
        T : Union[pd.DataFrame, pd.Series]
            Formatted treatment(s) to fit the model
        X_eval : pd.DataFrame
            Formatted confounders / effect modifiers for evaluation
        Y_eval : pd.DataFrame
            Formatted outcome for evaluation
        T_eval : Union[pd.DataFrame, pd.Series]
            Formatted treatment(s) for evaluation
        refutation_tests : Dict[str, Any]
            Dictionary with requested refutation tests and correspondig configurations
        causal_model_class : str
            Class name of the causal model
        causal_model_config : Dict[str, Any]
            Configuration to initialize new causal model instances
        alpha : float, optional
            Significance level for refutation tests, by default 0.05
        validation_start : Optional[str], optional
            Start of validation period, by default None (no validation)

        Returns
        -------
        Tuple[ List[Dict[str, Dict[str, pd.DataFrame]]], Optional[List[Dict[str, Dict[str, pd.DataFrame]]]]]
            Lists with refutation results (fitting and refutation periods)
        """
        estimate = self.cate(X=X, T=T)
        if not X_eval.empty:
            estimate_eval = self.cate(X=X_eval, T=T_eval)
        else:
            estimate_eval = None

        effect_intervals_list, effect_intervals_list_eval = [], []
        for ref_k, ref_v in refutation_tests.items():
            try:
                ref_obj = eval(ref_k)(
                    X=X,
                    Y=Y,
                    T=T,
                    X_eval=X_eval,
                    Y_eval=Y_eval,
                    T_eval=T_eval,
                    estimates=estimate,
                    estimates_eval=estimate_eval,
                    causal_model_class=causal_model_class,
                    causal_model_config=causal_model_config,
                    model=self,
                    **ref_v,
                )
                effect_intervals, effect_intervals_eval = ref_obj.refute_estimate(
                    show_progress_bar=True,
                    alpha=alpha,
                    validation_start=validation_start,
                )
                effect_intervals_list.append(effect_intervals)
                effect_intervals_list_eval.append(effect_intervals_eval)
                if self.save_to is not None:
                    # save refutation results for training data
                    if effect_intervals is not None:
                        if not os.path.isdir(f"{self.save_to}/01_training"):
                            os.makedirs(f"{self.save_to}/01_training")
                        for treatment in effect_intervals_list[0].keys():
                            # save to CSV
                            effect_intervals[treatment][
                                self.config["outcomes"][-1]
                            ].to_csv(
                                "{}/{}_{}.csv".format(
                                    f"{self.save_to}/01_training", ref_k, treatment
                                )
                            )
                            if not self.deactivate_plotting:
                                # plot per region
                                for region in X.index.get_level_values(
                                    "geo_id"
                                ).unique():
                                    plot_refutation_results(
                                        effect_intervals,
                                        ref_k,
                                        treatment=treatment,
                                        outcome=self.config["outcomes"][-1],
                                        region=region,
                                        save_to=f"{self.save_to}/01_training",
                                    )
                    # save refutation results for evaluation data
                    if effect_intervals_eval is not None:
                        if not os.path.isdir(f"{self.save_to}/02_evaluation"):
                            os.makedirs(f"{self.save_to}/02_evaluation")
                        for treatment in effect_intervals_list_eval[0].keys():
                            # save to CSV
                            effect_intervals_eval[treatment][
                                self.config["outcomes"][-1]
                            ].to_csv(
                                "{}/{}_{}.csv".format(
                                    f"{self.save_to}/02_evaluation", ref_k, treatment
                                )
                            )
                            if not self.deactivate_plotting:
                                # plot per region
                                for region in X.index.get_level_values(
                                    "geo_id"
                                ).unique():
                                    plot_refutation_results(
                                        effect_intervals_eval,
                                        ref_k,
                                        treatment=treatment,
                                        outcome=self.config["outcomes"][-1],
                                        region=region,
                                        save_to=f"{self.save_to}/02_evaluation",
                                    )
            except Exception as e:
                logging.warning("Refutation test {} failed: {}".format(ref_k, e))

        return effect_intervals_list, effect_intervals_list_eval

    @checkpoint(step_name="shap_analysis", message="Performing SHAP analysis...")
    def shap_values(
        self,
        X: pd.DataFrame,
        X_eval: pd.DataFrame,
        T: Union[pd.DataFrame, pd.Series],
        T_eval: Union[pd.DataFrame, pd.Series],
        interaction_variable: Optional[str] = None,
        defined_periods: Optional[List[List[str]]] = None,
        background_samples: int = 100,
        subsample_frac: Optional[float] = 0.5,
        **kwargs,
    ) -> List[Dict]:
        """Performs SHAP analysis with fitted model.

        Parameters
        ----------
        X : pd.DataFrame
            Formatted confounders / effect modifiers (fitting period)
        X_eval : pd.DataFrame
            Formatted confounders / effect modifiers (evaluation period)
        T : Union[pd.DataFrame, pd.Series]
            Formatted treatment(s) (fitting period)
        T_eval : Union[pd.DataFrame, pd.Series]
            Formatted treatment(s) (evaluation period)
        interaction_variable : Optional[str], optional
            If given, dependence plots will be made with interaction, by default None
        defined_periods : Optional[List[List[str]]], optional
            If given, SHAP analysis will be performed for defined_periods only, by default None (SHAP analysis for all data)
        background_samples : int, optional
           Number of observations sampled for background, by default 100
        subsample_frac : Optional[float], optional
            If given, data will be subsampled to speed up computation, by default 0.5

        Returns
        -------
        List[Dict]
            List of dictionaries, each of which contains SHAP results for one treatment level (aggregated per variable over time steps)
        """
        output = []
        if "on_folds" in kwargs.keys():
            logging.warning(
                f"on_folds is no valid option for SHAP analysis of {self.__class__.__name__} and will thus be ignored."
            )
        # concatenate training and evaluation data
        X_all = pd.concat([X, X_eval])
        # T_all = pd.concat([T, T_eval])

        if subsample_frac is not None:
            X_all = X_all.sample(frac=subsample_frac, random_state=42)
            # T_all = T_all.sample(frac=subsample_frac, random_state=42)

        if defined_periods is not None:
            # if given: Transform defined_periods to datetime
            defined_periods_datetime = [
                [
                    pd.to_datetime(
                        datetime.datetime.strptime(period[0] + "-1", "%Y-%W-%w")
                    ),
                    pd.to_datetime(
                        datetime.datetime.strptime(period[1] + "-1", "%Y-%W-%w")
                    ),
                ]
                for period in defined_periods
            ]
        else:
            # else: Run on all data
            defined_periods_datetime = [
                [
                    X_all.index.get_level_values("time_id").min(),
                    X_all.index.get_level_values("time_id").max(),
                ]
            ]

        for period in defined_periods_datetime:
            if not os.path.isdir(
                f"{self.save_to}/{period[0].date()}_{period[1].date()}"
            ):
                os.makedirs(f"{self.save_to}/{period[0].date()}_{period[1].date()}")

            X_subset = X_all[
                (X_all.index.get_level_values("time_id") > period[0])
                & (X_all.index.get_level_values("time_id") <= period[1])
            ]

            aggregated_shap_values_per_t = dict.fromkeys(self.T_levels[1:])
            for t in self.T_levels[1:]:
                background = maskers.Independent(
                    X_subset, max_samples=background_samples
                )

                explainer = shap.Explainer(
                    (lambda features: self.cate(X=features)[t].iloc[:, -1]),
                    background,
                )
                shap_values = explainer(X_subset)

                aggregated_shap_values = aggregate_shap_values(shap_values)
                aggregated_shap_values_per_t[t] = aggregated_shap_values

                if self.save_to is not None and not self.deactivate_plotting:
                    plot_shap(
                        shap_values=aggregated_shap_values,
                        features=list(X.columns.get_level_values("predictor").unique()),
                        treatment=t,
                        interaction_variable=interaction_variable,
                        save_to=f"{self.save_to}/{period[0].date()}_{period[1].date()}",
                    )

            with open(
                f"{self.save_to}/{period[0].date()}_{period[1].date()}/shap_values.pkl",
                "wb",
            ) as outp:
                pickle.dump(aggregated_shap_values_per_t, outp)
            output.append(aggregated_shap_values_per_t)

        return output
