# %%
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# %%
scopes = [
    "nuts0",
    "nuts0_nuts1",
    "swedish_strategy/nuts0",
    "swedish_strategy/nuts0_nuts1",
]
model = "CRNLearner"
outcome = "r_number7"
training_or_evaluation = "training"
base_dir = "../results"
base_dirs = [f"{base_dir}/{scope}" for scope in scopes]
# by population size
country_order = {
    "DE": "Germany",
    "FR": "France",
    "ES": "Spain",
    "PL": "Poland",
    "NL": "Netherlands",
    "BE": "Belgium",
    "SE": "Sweden",
}
npi_order = [
    "npi_stay_home",
    "npi_schools",
    "npi_work",
    "npi_internal_travel",
    "npi_international_travel",
    "npi_stay_home_r",
    "npi_schools_r",
    "npi_masks_r",
]

titles = ["CATE estimates of CRN (NUTS 0)", "CATE estimates of CRN (NUTS 0 + 1)"]

colors = sns.color_palette(n_colors=5)
MARGIN = 0.25

# %%
import re

df_list = []
for scope in scopes:
    country_dirs = glob.glob(f"{base_dir}/{scope}/*")
    for country_dir in country_dirs:
        npi_dirs = glob.glob(f"{country_dir}/{model}*{outcome}")
        for npi_dir in npi_dirs:
            npi = re.search(r"npi\w+(?=_r)", npi_dir).group()
            try:
                one_df = pd.read_csv(
                    glob.glob(
                        f"{npi_dir}/evaluate_causal_model/cate/*{training_or_evaluation}/cate*.csv"
                    )[0]
                )
            except:
                continue
            one_df["npi"] = npi
            one_df["scope"] = scope.split("/")[-1]
            df_list.append(one_df)
cate_df = pd.concat(df_list)

# %%
# aggregate to country_level
# cate_df["country"] = cate_df["geo_id"].str[:2]

# keep only country level
cate_df = cate_df[cate_df["geo_id"].str.len() == 2]
cate_df["country"] = cate_df["geo_id"]

# aggregate mean and quantiles
cate_df = (
    cate_df.groupby(["country", "npi", "scope"])
    .aggregate(
        mean_cate=("point_estimate", lambda x: x.mean()),
        lower_cate_point=("point_estimate", lambda x: x.quantile(0.05)),
        upper_cate_point=("point_estimate", lambda x: x.quantile(0.95)),
        lower_cate_ci=("ci_lower", lambda x: x.quantile(0.05)),
        upper_cate_ci=("ci_upper", lambda x: x.quantile(0.95)),
    )
    .reset_index()
)

# sort according to given country list
cate_df = cate_df.sort_values(by="scope").sort_values(
    by="country", key=lambda x: x.map(list(country_order.keys()).index)
)
cate_df


# %%
# Include refutation results to grey out too unstable results
ref_results_file = f"{base_dir}/refutation_results.csv"
ref_results = pd.read_csv(ref_results_file)
tests = ["DataSubsetRefuter", "RandomCommonCauseRefuter", "PlaceboTreatmentRefuter"]
threshold = 0.5  # mark if 50% or more observations fail one of the three tests


def collect_ref_results(x, scope):
    # read number of observations below 0 (according to inference)
    cate_below_0 = ref_results[
        (ref_results["scope"] == scope)
        & (ref_results["model"] == model)
        & (ref_results["outcome"] == outcome)
        & (ref_results["country"] == x["country"])
        & (ref_results["treatment"] == x["npi"])
    ]["cate_below_0"].item()
    # read number of passed for each refutation test
    ref_counts_passed = ref_results[
        (ref_results["scope"] == scope)
        & (ref_results["model"] == model)
        & (ref_results["outcome"] == outcome)
        & (ref_results["country"] == x["country"])
        & (ref_results["treatment"] == x["npi"])
    ][tests]
    # compute ratio of passed / cate below 0
    ref_counts_passed_ratio = ref_counts_passed / cate_below_0
    # apply threshold condition: Ratio below threshold or NaN (no significant CATEs found)
    threshold_condition = (
        ref_counts_passed_ratio < threshold
    ) + ref_counts_passed_ratio.isna()
    # return True if condition fulfilled
    return threshold_condition.any().any()


# grey out when condition fulfilled
for scope in scopes:
    cate_df.loc[cate_df["scope"] == scope, "grey_out"] = cate_df.apply(
        collect_ref_results, axis=1, scope=scope.split("/")[-1]
    )

# %%
offsets = {
    "npi_stay_home": 0.05,
    "npi_schools": 0.04,
    "npi_work": 0.03,
    "npi_internal_travel": 0.02,
    "npi_international_travel": 0.01,
    "npi_stay_home_r": 0.05,
    "npi_internal_travel_r": 0.01,
    "npi_masks_r": 0,
}
colors = {
    "npi_stay_home": "#1f77b4",
    "npi_schools": "#ff7f0e",
    "npi_work": "#2ca02c",
    "npi_internal_travel": "#d62728",
    "npi_international_travel": "#9467bd",
    "npi_stay_home_r": "#1f77b4",
    "npi_internal_travel_r": "#d62728",
    "npi_masks_r": "#8c564b",
}
markers = {
    "npi_stay_home": "o",
    "npi_schools": "o",
    "npi_work": "o",
    "npi_internal_travel": "o",
    "npi_international_travel": "o",
    "npi_stay_home_r": "s",
    "npi_internal_travel_r": "s",
    "npi_masks_r": "s",
}


def plot_ranges(df, ax, lbl="", _margin=0):
    y = np.array([offsets[npi] for npi in df["npi"].unique()])
    facecolors = np.array([colors[npi] for npi in df["npi"].unique()])
    markers_list = np.array([markers[npi] for npi in df["npi"].unique()])
    alphas = [0.1 if g else 0.9 for g in df["grey_out"]]
    y = y[: df.shape[0]]
    x = df["mean_cate"].reset_index(drop=True)
    xerr_ci = df[["lower_cate_ci", "upper_cate_ci"]].values.T
    xerr_point = df[["lower_cate_point", "upper_cate_point"]].values.T

    for i in range(len(y)):
        ax.plot(
            xerr_ci[..., i],
            [y[i] - _margin, y[i] - _margin],
            color="black",
            linestyle="--",
            alpha=alphas[i],
        )
        ax.plot(
            xerr_point[..., i],
            [y[i] - _margin, y[i] - _margin],
            color="black",
            alpha=alphas[i],
        )
    for i in range(len(y)):
        ax.scatter(
            x[i],
            y[i] - _margin,
            marker=markers_list[i],
            s=140,
            facecolors=facecolors[i],
            edgecolors="k",
            alpha=alphas[i],
            label=lbl,
        )


# %%
fig, axes = plt.subplots(
    1,
    len(cate_df["scope"].unique()),
    figsize=(len(cate_df["scope"].unique()) * 10, 14),
    dpi=150,
)

for scope, ax, title in zip(cate_df["scope"].unique(), axes, titles):
    for i, country in enumerate(cate_df["country"].unique()):
        _margin = MARGIN * i
        cate_df_subset = cate_df[
            (cate_df["country"] == country) & (cate_df["scope"] == scope)
        ]
        plot_ranges(cate_df_subset, ax, country, _margin)

    ax.axvline(0, color="k", ls="--", alpha=0.6)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("CATE estimate", fontsize=14)
    ax.set_yticks(
        [-MARGIN * i + 0.02 for i in range(len(cate_df["country"].unique()))],
        country_order.values(),
        fontsize=14,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            markersize=12,
            markeredgecolor="black",
            color="w",
            marker="o",
            markerfacecolor=colors["npi_stay_home"],
            label="Stay-at-home orders",
        ),
        Line2D(
            [0],
            [0],
            markersize=12,
            markeredgecolor="black",
            color="w",
            marker="o",
            markerfacecolor=colors["npi_schools"],
            label="School closures",
        ),
        Line2D(
            [0],
            [0],
            markersize=12,
            markeredgecolor="black",
            color="w",
            marker="o",
            markerfacecolor=colors["npi_work"],
            label="Workplace closures",
        ),
        Line2D(
            [0],
            [0],
            markersize=12,
            markeredgecolor="black",
            color="w",
            marker="o",
            markerfacecolor=colors["npi_internal_travel"],
            label="Internal travel restrictions",
        ),
        Line2D(
            [0],
            [0],
            markersize=12,
            markeredgecolor="black",
            color="w",
            marker="o",
            markerfacecolor=colors["npi_international_travel"],
            label="Border closure",
        ),
        Line2D(
            [0],
            [0],
            markersize=12,
            markeredgecolor="black",
            color="w",
            marker="s",
            markerfacecolor=colors["npi_stay_home_r"],
            label="Stay-at-home\n recommendations",
        ),
        Line2D(
            [0],
            [0],
            markersize=12,
            markeredgecolor="black",
            color="w",
            marker="s",
            markerfacecolor=colors["npi_internal_travel_r"],
            label="Recommended internal\n movement restrictions",
        ),
        Line2D(
            [0],
            [0],
            markersize=12,
            markeredgecolor="black",
            color="w",
            marker="s",
            markerfacecolor=colors["npi_masks_r"],
            label="Recommended mask\n wearing",
        ),
    ]
    ax.set_xlim(left=-0.25, right=0.15)
    if ax == axes[-1]:
        ax.legend(handles=legend_elements, loc="lower right")

plt.savefig(f"{base_dir}/Fig2.tif", bbox_inches="tight")
