import numpy as np
import pandas as pd
from src.utils import nonparametric_significance_test
from typing import Dict, Union, Any, Optional, List, Tuple


class CausalRefuter:
    # Default value for the number of simulations to be conducted
    DEFAULT_NUM_SIMULATIONS = 100
    PROGRESS_BAR_COLOR = "green"

    def __init__(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        T: Union[pd.Series, pd.DataFrame],
        estimates: Dict[str, Any],
        causal_model_class: str,
        causal_model_config: Dict[str, Any],
        X_eval: pd.DataFrame = pd.DataFrame(),
        Y_eval: pd.DataFrame = pd.DataFrame(),
        T_eval: Union[pd.DataFrame, pd.Series] = pd.DataFrame(),
        estimates_eval: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self._X = X
        self._Y = Y
        self._T = T
        self._X_eval = X_eval
        self._Y_eval = Y_eval
        self._T_eval = T_eval
        self._estimate = estimates
        self._estimate_eval = estimates_eval
        self._causal_model_class = causal_model_class
        self._causal_model_config = causal_model_config
        self._random_seed = None

        # joblib params for parallel processing
        self._n_jobs = kwargs.pop("n_jobs", -1)
        self._verbose = kwargs.pop("verbose", 0)

        if "random_seed" in kwargs:
            self._random_seed = kwargs["random_seed"]
            np.random.seed(self._random_seed)

    def _summmary_tables_all_treatments(
        self,
        alpha: float,
        refute: List[Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]],
        refute_index: int,
        estimate: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Creates tables with summaries on refutation results for each treatment level

        Parameters
        ----------
        alpha : float
            Significance level for refutation test
        refute : List[Tuple[Dict[str, pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]]
            Dictionary with refutation results per treatment level, for fitting and optionally for evaluation data
        refute_index : int
            Specifies which element is used from refute (typically fitting or evaluation)
        estimate : Optional[Dict[str, Any]]
            Point estimate of the original model

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary of summary tables for each refutation test
        """
        if type(refute[0][refute_index]) != dict or estimate is None:
            return None

        summary_tables_all_treatments = {k: {} for k in refute[0][refute_index].keys()}
        ref_distributions_lists = {k: [] for k in refute[0][refute_index].keys()}
        ref_distributions_stacked = dict.fromkeys(refute[0][refute_index].keys())

        for k in refute[0][refute_index].keys():
            for sim in refute:
                all_regions = []
                for region in (
                    sim[refute_index][k].index.get_level_values("geo_id").unique()
                ):
                    # results must be padded because the shifting causes different shapes (first rows are missing)
                    current_ref_results = (
                        sim[refute_index][k]
                        .reindex(estimate[k].index, fill_value=np.nan)
                        .loc[region]
                    )
                    pad_width = (
                        estimate[k].loc[region].shape[0] - current_ref_results.shape[0]
                    )
                    all_regions.append(
                        np.pad(
                            current_ref_results,
                            pad_width=((pad_width, 0), (0, 0)),
                            mode="constant",
                            constant_values=(np.nan,),
                        )
                    )
                all_regions = np.concatenate(all_regions)
                ref_distributions_lists[k].append(all_regions)

            # stack outputs of all simulations
            ref_distributions_stacked[k] = np.moveaxis(
                np.stack(ref_distributions_lists[k]), [0, 1, 2], [2, 0, 1]
            )

            # perform two-tailed significance_test; if distribution is differs from estimate in any direction, mark as significant
            summary_table = self._summary_table(
                estimate[k],
                ref_distributions_stacked[k],
                alpha=alpha,
                tailed="two-tailed",
            )
            # NaNs occur in the beginning of the data because look back refutation does not cover it sufficiently -> automatically pass the test
            for _, one_table in summary_table.items():
                one_table.loc[one_table["mean"].isna(), "significant"] = False
            summary_tables_all_treatments[k] = summary_table

        return summary_tables_all_treatments

    def _summary_table(
        self,
        point_estimates: pd.DataFrame,
        cate_dist: np.ndarray,
        alpha: float = 0.01,
        tailed: str = "two-tailed",
    ) -> Dict[str, pd.DataFrame]:
        """Creates summary table for a given refutation test distribution

        Parameters
        ----------
        point_estimates : pd.DataFrame
            Point estimate to test significance (of refutation test) for
        cate_dist : np.ndarray
            _description_
        alpha : float, optional
            Significance level for refutation, by default 0.01
        tailed : str, optional
            "left", "right", "two-tailed", by default "two-tailed"

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of summary tables for each refutation test
        """
        mean = np.mean(cate_dist, axis=-1)
        ci_lower = np.quantile(cate_dist, q=alpha / 2, axis=-1)
        ci_upper = np.quantile(cate_dist, q=1 - alpha / 2, axis=-1)

        pvalue = nonparametric_significance_test(
            point_estimates.values, cate_dist, shift_to_value=None, tailed=tailed
        )
        if tailed == "two-tailed":
            significant = pvalue < alpha / 2
        else:
            significant = pvalue < alpha

        summary_tables = {
            o: pd.DataFrame(
                {
                    "point_estimate": point_estimates.iloc[..., i],
                    "mean": mean[..., i],
                    "ci_lower": ci_lower[..., i],
                    "ci_upper": ci_upper[..., i],
                    "pvalue": pvalue[..., i],
                    "significant": significant[..., i],
                }
            ).loc[point_estimates.index]
            for i, o in enumerate(self._causal_model_config["outcomes"])
        }

        return summary_tables
