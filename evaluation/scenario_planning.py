# %%
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap

# %%
country = "DE"
waves = ["2nd_wave"]
npis = [
    "npi_stay_home",
    "npi_schools",
    "npi_internal_travel",
    "npi_international_travel",
]
data = pd.read_csv("../data/data.csv", parse_dates=["date"])
# keep only country level
data = data[data["geo"] == country]
# set date as index and keep only npis
data = data.set_index("date")

T = data[npis]
r_number = data["r_number"]

out_dir = "../results"


# %%
# Import CATE, outcome predictions and SHAP
cate_dict = {}
pred_dict = {}
shap_dict = {}

for wave in waves:
    for npi in npis:
        cate_dir = glob.glob(
            f"../results/scenario_analysis/{country}/{wave}/*{npi}*/evaluate_causal_model/cate"
        )[0]
        outcome_dir = glob.glob(
            f"../results/scenario_analysis/{country}/{wave}/*{npi}*/evaluate_causal_model/y_models"
        )[0]
        shap_dir = glob.glob(
            f"../results/scenario_analysis/{country}/{wave}/*{npi}*/shap_analysis"
        )[0]
        for dataset in ["training", "evaluation"]:
            try:
                one_df_cate = pd.read_csv(
                    glob.glob(f"{cate_dir}/*{dataset}/cate*.csv")[0],
                    parse_dates=["time_id"],
                )
                one_df_pred = pd.read_csv(
                    glob.glob(f"{outcome_dir}/*{dataset}/pred_f*.csv")[0],
                    parse_dates=["time_id"],
                )
                # only keep country level
                one_df_pred = one_df_pred[one_df_pred["geo_id"] == country]
                one_df_cate = one_df_cate[one_df_cate["geo_id"] == country]
            except:
                continue
            cate_dict[(wave, npi, dataset)] = one_df_cate
            pred_dict[(wave, npi, dataset)] = one_df_pred

        pickled_shap_values = glob.glob(f"{shap_dir}/*/shap_values.pkl")[0]
        with open(pickled_shap_values, "rb") as f:
            shap_dict[(wave, npi)] = pickle.load(f)


# %%
# Plot CATE and real NPI
def plot_cate(cate_df, ax, color):
    significant_point = cate_df["point_estimate"].copy()
    significant_ci_lower = cate_df["ci_lower"].copy()
    significant_ci_upper = cate_df["ci_upper"].copy()
    significant_point[~cate_df["significant"]] = np.nan
    significant_ci_lower[~cate_df["significant"]] = np.nan
    significant_ci_upper[~cate_df["significant"]] = np.nan

    dates = cate_df["time_id"]
    ax.plot(
        dates,
        cate_df["point_estimate"],
        c="black",
    )
    ax.fill_between(
        dates,
        cate_df["ci_lower"],
        cate_df["ci_upper"],
        color=color,
        alpha=0.2,
    )

    ax.fill_between(
        dates, significant_ci_lower, significant_ci_upper, color=color, alpha=0.8
    )

    return (
        min(cate_df["time_id"]),
        max(cate_df["time_id"]),
        min(cate_df["ci_lower"]) - 0.025,
        max(cate_df["ci_upper"]) + 0.025,
    )


# %%
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

for npi in npis:
    _, axes = plt.subplots(
        len(waves) + 1,
        figsize=(14, 2 + len(waves) * 4),
        gridspec_kw={"height_ratios": len(waves) * [1] + [0.25]},
    )
    min_y = -0.26
    max_y = -0.04
    # plot predicted CATE for different waves
    for i, wave in enumerate(waves):
        min_x, _, min_y_train, max_y_train = plot_cate(
            cate_dict[(wave, npi, "training")], axes[i], color="lightgray"
        )
        _, max_x, min_y_eval, max_y_eval = plot_cate(
            cate_dict[(wave, npi, "evaluation")], axes[i], color="darkseagreen"
        )
        min_y = min(min_y, min_y_train, min_y_eval)
        max_y = max(max_y, max_y_train, max_y_eval)
    for i, wave in enumerate(waves):
        axes[i].set_ylabel("CATE estimate")
        axes[i].margins(x=0)
        axes[i].set_xlim(min_x, max_x)
        axes[i].set_ylim(min_y, max_y)
        axes[i].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        train_style = Rectangle(
            (0, 0), width=3, height=1, color="lightgray", label="Training period"
        )
        val_style = Rectangle(
            (0, 0),
            width=3,
            height=1,
            color="darkseagreen",
            label=f'Evaluation period ({" ".join(wave.split("_"))})',
        )
        axes[i].legend(handles=[train_style, val_style])
        if i == 0:
            axes[i].set_title(f"CATE predictions for {npi}")

    # plot real NPI in last subplot
    sns.heatmap(
        np.asarray(T[T.index.to_series().between(min_x, max_x)][npi]).reshape(1, -1),
        cmap=sns.color_palette("Greys"),
        cbar=False,
        ax=axes[-1],
        xticklabels=False,
        yticklabels=False,
    )

    plt.savefig(
        f"{out_dir}/CATE_{country}_{npi}.svg", bbox_inches="tight", transparent=True
    )


# %%
for npi in npis:
    for wave in waves:
        plt.figure(figsize=(4, 4))
        plt.title("SHAP values for evaluation period")
        shap.plots.beeswarm(
            shap_dict[(wave, npi)][f"{npi}_1"],
            max_display=8,
            show=False,
            plot_size=(3, 4),
        )
        plt.savefig(
            f"{out_dir}/SHAP_{country}_{npi}_{wave}.svg",
            bbox_inches="tight",
            transparent=True,
        )

# %%
colors = {
    "npi_stay_home": "#1f77b4",
    "npi_schools": "#ff7f0e",
    "npi_work": "#2ca02c",
    "npi_internal_travel": "#d62728",
    "npi_international_travel": "#9467bd",
}

for wave in waves:
    _, axes = plt.subplots(
        2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2, 1]}, dpi=300
    )
    for npi in npis:
        factual = pred_dict[(wave, npi, "evaluation")].set_index("time_id")
        factual_counterfactual = pd.DataFrame(
            np.zeros((len(factual), 2)), columns=[1, 0], index=factual.index
        )
        treatment = T[npi].loc[factual.index]

        factual_counterfactual.loc[treatment == 1, 1] = factual[treatment == 1].iloc[
            :, -1
        ]
        factual_counterfactual.loc[treatment == 0, 0] = factual[treatment == 0].iloc[
            :, -1
        ]

        factual_counterfactual = factual_counterfactual.merge(
            cate_dict[(wave, npi, "evaluation")].set_index("time_id")["point_estimate"],
            how="left",
            left_index=True,
            right_index=True,
        )

        factual_counterfactual.loc[treatment == 1, 0] = (
            factual_counterfactual.loc[treatment == 1, 1]
            + factual_counterfactual.loc[treatment == 1, "point_estimate"]
        )
        factual_counterfactual.loc[treatment == 0, 1] = (
            factual_counterfactual.loc[treatment == 0, 0]
            - factual_counterfactual.loc[treatment == 0, "point_estimate"]
        )

        factual_counterfactual["relative_effect"] = -(
            (factual_counterfactual[1] - factual_counterfactual[0])
            / factual_counterfactual[1]
            * 100
        )
        factual_counterfactual = factual_counterfactual.merge(
            r_number, how="left", left_index=True, right_index=True
        )

        axes[0].plot(factual_counterfactual["relative_effect"], c=colors[npi])
        axes[0].set_ylabel("Relative reduction in %")
        axes[0].legend(npis)
        axes[0].set_title(f"Relative $R_t$ reduction in {' '.join(wave.split('_'))}")

        axes[1].plot(factual_counterfactual["r_number"], c="black")
        axes[1].set_title(f"$R_t$ in {' '.join(wave.split('_'))}")

        plt.savefig(f"{out_dir}/S3_Fig.png", dpi=300)
