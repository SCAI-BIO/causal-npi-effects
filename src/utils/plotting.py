import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    mean_absolute_percentage_error,
)
import shap
from shap.plots.colors import red_blue
import seaborn as sns
import logging


def plot_fit(
    Y_true: pd.DataFrame,
    Y_pred_dict: Dict[str, pd.DataFrame],
    T_filter: pd.DataFrame,
    metrics: List[str],
    models: List[str],
    time_step: int,
    save_to: str,
    counterfactual: bool = False,
) -> None:
    """Plots how well outcome predictions align with outcome.

    Parameters
    ----------
    Y_true : pd.DataFrame
        Ground truth outcome
    Y_pred_dict : Dict[str, pd.DataFrame]
       Dictionary with outcome predictions for each treatment level
    T_filter : pd.DataFrame
        Filter per treatment level
    metrics : List[str]
        Metrics to include in outputs
    models : List[str]
        Names of the given models
    time_step : int
        Time step in the outcome for which plots are created
    save_to : str
        Output directory
    counterfactual : bool, optional
        Whether model made predictions on treatment condition it was not trained on, by default False
    """
    colours = ["steelblue", "darkseagreen", "indianred", "gold", "darkorchid"]
    outcome_name = "".join(map(str, Y_true.columns[time_step]))

    if counterfactual:
        T_filter = ~T_filter

    _, ax = plt.subplots(figsize=(9, 9), dpi=300)
    metrics_list = {k: [] for k in metrics}
    for i, (k, Y_pred) in enumerate(Y_pred_dict.items()):
        if Y_pred is None:
            continue
        # expand dims if only one column
        if len(Y_pred.shape) == 1:
            Y_pred = np.expand_dims(Y_pred, -1)

        plt.scatter(
            Y_true.loc[T_filter[k]].iloc[..., time_step],
            Y_pred.iloc[..., time_step],
            c=colours[i % len(colours)],
            alpha=0.2,
        )
        # plot line with slope 1
        x = np.linspace(
            np.min(
                [
                    np.min(Y_pred.iloc[:, time_step]),
                    np.min(Y_true.iloc[:, time_step].values),
                    0,
                ]
            ),
            np.max(
                [
                    np.max(Y_pred.iloc[:, time_step]),
                    np.max(Y_true.iloc[:, time_step].values),
                ]
            ),
            len(Y_pred),
        )
        ax.plot(x, x, c="gray", linestyle="--")
        ax.set_xlabel("{} (observed)".format(outcome_name))
        ax.set_ylabel("{} (predicted)".format(outcome_name))
        ax.set_title(
            "Prediction of {} {}".format(
                "observed" if not counterfactual else "counterfactual",
                outcome_name,
            )
        )

        if len(models) > 1:
            info_box = "{}".format(models[i])
        else:
            info_box = "{}".format(models[0])
        for metric in metrics:
            if metric == "mse":
                mse = mean_squared_error(
                    Y_true.loc[T_filter[k]].iloc[..., time_step],
                    Y_pred.iloc[:, time_step],
                )
                metrics_list["mse"] += [mse]
                info_box += "\n{}={:.4f}".format(metric, mse)
            elif metric == "r2":
                r2 = r2_score(
                    Y_true.loc[T_filter[k]].iloc[:, time_step],
                    Y_pred.iloc[:, time_step],
                )
                metrics_list["r2"] += [r2]
                info_box += "\n{}={:.4f}".format(metric, r2)
            elif metric == "accuracy":
                info_box += "\n{}={:.4f}".format(
                    metric,
                    accuracy_score(
                        Y_true.loc[T_filter[k]].iloc[:, time_step],
                        Y_pred.iloc[:, time_step],
                    ),
                )
            elif metric == "mape":
                info_box += "\n{}={:.4f}".format(
                    metric,
                    mean_absolute_percentage_error(
                        Y_true.loc[T_filter[k]].iloc[:, time_step],
                        Y_pred.iloc[:, time_step],
                    ),
                )

        if len(Y_pred_dict) <= 3:
            props = dict(
                boxstyle="round", facecolor=colours[i % len(colours)], alpha=0.5
            )
            ax.text(
                0.05,
                0.97 - (i * len(metrics)) * 0.08,
                info_box,
                fontsize=14,
                va="top",
                transform=ax.transAxes,
                bbox=props,
            )

    plt.savefig(
        "{}/fit_{}_{}.png".format(
            save_to,
            "f" if not counterfactual else "cf",
            outcome_name,
        )
    )
    plt.close()


def plot_cate(
    effect_intervals: Dict,
    treatment: str,
    T: pd.Series,
    outcome: str,
    region: str,
    save_to: str,
) -> None:
    """Plots CATE estimates (point and interval). Color in red if significantly below zero.

    Parameters
    ----------
    effect_intervals : Dict
        Dictionary with interval estimates
    treatment : str
        Treatment name
    T : pd.Series
        Formatted treatment
    outcome : str
        Outcome name
    region : str
        Region for which CATE is plotted
    save_to : str
        Output directory
    """
    significant_point = (
        effect_intervals[treatment][outcome].loc[region]["point_estimate"].copy()
    )
    significant_ci_lower = (
        effect_intervals[treatment][outcome].loc[region]["ci_lower"].copy()
    )
    significant_ci_upper = (
        effect_intervals[treatment][outcome].loc[region]["ci_upper"].copy()
    )
    significant_point[
        ~effect_intervals[treatment][outcome].loc[region]["significant"]
    ] = np.nan
    significant_ci_lower[
        ~effect_intervals[treatment][outcome].loc[region]["significant"]
    ] = np.nan
    significant_ci_upper[
        ~effect_intervals[treatment][outcome].loc[region]["significant"]
    ] = np.nan

    dates = effect_intervals[treatment][outcome].loc[region].index.to_series()
    T = T.loc[region].loc[dates]
    _, ((ax1, ax2)) = plt.subplots(
        2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [1, 0.25]}
    )
    ax1.plot(
        dates,
        effect_intervals[treatment][outcome].loc[region]["point_estimate"],
        c="black",
    )
    ax1.fill_between(
        dates,
        effect_intervals[treatment][outcome].loc[region]["ci_lower"],
        effect_intervals[treatment][outcome].loc[region]["ci_upper"],
        color="lightsteelblue",
        alpha=0.4,
    )

    ax1.fill_between(
        dates, significant_ci_lower, significant_ci_upper, color="lightcoral", alpha=0.8
    )
    ax1.set_title("CATE estimates for {} in {}".format(treatment, region))
    ax1.set_xlabel("Date")
    ax1.set_ylabel("CATE estimate")
    ax1.margins(x=0)

    # AXIS 2: Real NPIs
    if len(T.unique()) < 4:
        sns.heatmap(
            np.asarray(T).reshape(1, -1),
            cmap=sns.color_palette("Greys"),
            cbar=False,
            ax=ax2,
            xticklabels=False,
            yticklabels=False,
        )
    else:
        ax2.plot(T, color="black")
        ax2.margins(x=0)
        x_axis = ax2.axes.get_xaxis()
        x_axis.set_visible(False)

    plt.savefig("{}/cate_{}_{}.png".format(save_to, treatment, region))
    plt.close()


def plot_residuals_over_time(
    target: pd.DataFrame,
    residuals: pd.DataFrame,
    region: str,
    save_to: str,
    fold_numbers: Optional[pd.Series] = None,
) -> None:
    """
    A method to plot residuals from first stage of double/debiased methods.
    TODO: EXCLUDE for published version (applicable only to DML)
    """
    colours = ["steelblue", "darkseagreen", "indianred", "gold", "darkorchid"]
    for t in target.columns:
        fig = plt.figure(figsize=(16, 9))
        t_str = "".join(map(str, t))
        plt.plot(target.index, target[t], color="black")
        plt.title("Residuals from the prediction of {} in {}".format(t_str, region))
        if fold_numbers is not None:
            for i in range(fold_numbers.max() + 1):
                plt.bar(
                    residuals[fold_numbers == i].index,
                    residuals[fold_numbers == i][t],
                    width=0.01,
                    ec=colours[i % len(colours)],
                )
        else:
            plt.bar(residuals.index, residuals[t], width=0.01, ec=colours[0])
        plt.savefig(
            "{}/residuals_{}_{}.png".format(save_to, "".join(map(str, t)), region)
        )
        plt.close()


def plot_window_predictions_over_time(
    target: pd.Series,
    predictions: List[pd.DataFrame],
    region: str,
    level: int,
    save_to: str,
    predictions_counterfactual: Optional[List[pd.DataFrame]] = None,
) -> None:
    """
    A method to plot windowed predictions over time.
    TODO: EXCLUDE in published version (applicable only to causal estimation through time)
    """
    colours = ["steelblue", "darkseagreen", "indianred", "gold", "darkorchid"]

    plt.figure(figsize=(26, 7))
    plt.plot(target.loc[region].index, target.loc[region], c="black", linestyle="-")

    # observed
    for i, pred in enumerate(predictions):
        if pred.empty:
            continue
        fold_pred = pred.loc[region]

        for t in range(fold_pred.shape[0]):
            if fold_pred.shape[1] > 1:
                # window predictions
                plt.plot(
                    [
                        fold_pred.iloc[t].name + pd.DateOffset(i)
                        for i in range(1, fold_pred.shape[1] + 1)
                    ],
                    fold_pred.iloc[t],
                    c=colours[i % len(colours)],
                    alpha=1.0,
                )
            else:
                # categorical
                plt.scatter(
                    fold_pred.iloc[t].name,
                    fold_pred.iloc[t],
                    c=colours[i % len(colours)],
                    alpha=1.0,
                )

    # counterfactual
    if predictions_counterfactual is not None:
        for i, pred in enumerate(predictions_counterfactual):
            if pred.empty:
                continue
            fold_pred = pred.loc[region]

            for t in range(fold_pred.shape[0]):
                if fold_pred.shape[1] > 1:
                    plt.plot(
                        [
                            fold_pred.iloc[t].name + pd.DateOffset(i)
                            for i in range(1, fold_pred.shape[1] + 1)
                        ],
                        fold_pred.iloc[t],
                        c=colours[i % len(colours)],
                        alpha=0.3,
                        linestyle="--",
                    )

    plt.savefig(
        "{}/window_predictions_level{}_{}.png".format(save_to, str(level), region)
    )
    plt.close()


def plot_refutation_results(
    effect_intervals: Dict,
    ref_test: str,
    treatment: str,
    outcome: str,
    region: str,
    save_to: str,
) -> None:
    """Plots test distributions from refutation tests

    Parameters
    ----------
    effect_intervals : Dict
        Dictionary with interval estimates from refutation test
    ref_test : str
        Name of refutation test
    treatment : str
        Name of treatment
    outcome : str
        Name of outcome
    region : str
        Region for which refutation results are plotted
    save_to : str
        Output directory
    """
    # colours to use for the different refutation tests
    colours = {
        "PlaceboTreatmentRefuter": ("wheat", "darkgoldenrod"),
        "RandomCommonCauseRefuter": ("mediumorchid", "indigo"),
        "AddedCommonCauseRefuter": ("cornflowerblue", "midnightblue"),
        "DataSubsetRefuter": ("lightseagreen", "teal"),
        "LookBackRefuter": ("lightcoral", "red"),
        "PropensityThresholdingTest": ("peru", "saddlebrown"),
    }
    default_colours = ("mediumorchid", "indigo")

    fig = plt.figure(figsize=(12, 4), dpi=300)
    significant_point = (
        effect_intervals[treatment][outcome].loc[region]["point_estimate"].copy()
    )
    significant_ci_lower = (
        effect_intervals[treatment][outcome].loc[region]["ci_lower"].copy()
    )
    significant_ci_upper = (
        effect_intervals[treatment][outcome].loc[region]["ci_upper"].copy()
    )
    significant_point[
        ~effect_intervals[treatment][outcome].loc[region]["significant"]
    ] = np.nan
    significant_ci_lower[
        ~effect_intervals[treatment][outcome].loc[region]["significant"]
    ] = np.nan
    significant_ci_upper[
        ~effect_intervals[treatment][outcome].loc[region]["significant"]
    ] = np.nan

    plt.plot(
        effect_intervals[treatment][outcome].loc[region]["point_estimate"],
        c="gray",
        alpha=0.5,
    )
    plt.fill_between(
        effect_intervals[treatment][outcome].loc[region]["point_estimate"].index,
        effect_intervals[treatment][outcome].loc[region]["ci_lower"],
        effect_intervals[treatment][outcome].loc[region]["ci_upper"],
        color=colours.get(ref_test, default_colours)[0],
        alpha=0.3,
    )

    plt.fill_between(
        effect_intervals[treatment][outcome].loc[region]["point_estimate"].index,
        significant_ci_lower,
        significant_ci_upper,
        color=colours.get(ref_test, default_colours)[1],
    )

    plt.title("{} for {} in {}".format(ref_test, treatment, region))

    # deactivate ticks and frames
    plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    fig.savefig("{}/{}_{}_{}.png".format(save_to, ref_test, treatment, region))
    plt.close()


def plot_shap(
    shap_values: shap._explanation.Explanation,
    features: List[str],
    save_to: str,
    treatment: str,
    beeswarm_max_display: int = 15,
    interaction_variable: Optional[str] = None,
) -> None:
    """Creates beeswarm and dependence plots for SHAP results.

    Parameters
    ----------
    shap_values : shap._explanation.Explanation
        Explanation object from SHAP analysis
    features : List[str]
        List of features
    save_to : str
        Output directory
    treatment : str
        Name of treatment
    beeswarm_max_display : int, optional
        Number of features shown in beeswarm, by default 15
    interaction_variable : Optional[str], optional
        If given, dependence plots will be made with interaction, by default None
    """
    colors = ["#033047"]

    # beeswarm plot
    shap.plots.beeswarm(shap_values, max_display=beeswarm_max_display, show=False)
    plt.savefig("{}/shap_beeswarm.png".format(save_to), bbox_inches="tight")
    plt.close()

    # assert that interaction variable is in features
    if (
        interaction_variable is not None
        and interaction_variable not in shap_values.feature_names
    ):
        logging.warning(
            f"Given interaction variable {interaction_variable} is not in feature set. Ignoring for SHAP dependence plot."
        )
        interaction_variable = None

    # dependence plots
    for feature in features:
        plt.figure(figsize=(12, 9), dpi=300)
        idx = [
            (f, i)
            for i, f in enumerate(shap_values.feature_names)
            if f.startswith(feature)
        ]

        for ii, (f, i) in enumerate(idx):
            plt.scatter(
                shap_values.data[:, i],
                shap_values.values[:, i],
                c=shap_values.data[
                    :, shap_values.feature_names.index(interaction_variable)
                ]
                if interaction_variable is not None
                else colors[ii],
                cmap=red_blue,
                alpha=0.8,
            )
            if interaction_variable is not None:
                cbar = plt.colorbar()
                cbar.set_label(interaction_variable, rotation=270)
            plt.xlabel(f)
            plt.ylabel("SHAP value for\n{}".format(f))
        # plt.legend([f for f,_ in idx])
        plt.savefig(
            "{}/shap_dependence_plot_{}_{}.png".format(save_to, feature, treatment)
        )
        plt.close()
