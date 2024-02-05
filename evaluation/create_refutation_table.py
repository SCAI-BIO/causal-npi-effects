# %%

import re
import glob
import pandas as pd
import numpy as np

result_directories = [
    "nuts0",
    "nuts0_nuts1",
    "swedish_strategy/nuts0",
    "swedish_strategy/nuts0_nuts1",
    # "scenario_analysis/DE"
]

base_dir = "../results"
base_data = "../data/data.csv"

exp_list = []
for dir in result_directories:
    exp_list.extend(glob.glob(f"{base_dir}/{dir}/*/*"))

experiments = [(exp, "_".join(exp.split("/")[-3:])) for exp in exp_list]

training_or_evaluation = "02_evaluation"  # "01_training"

ref_tests = [
    "DataSubsetRefuter",
    "RandomCommonCauseRefuter",
    "PropensityThresholdingTest",
    "PlaceboTreatmentRefuter",
]


# extract treatments
pattern = r"Learner_(.*?)_r_number"
treatments = [re.search(pattern, exp[0]).group(1).strip("_") for exp in experiments]
treatment_levels = len(treatments) * [1]

cate = {
    k[1]: pd.read_csv(
        "{}/evaluate_causal_model/cate/{}/cate_{}.csv".format(
            k[0], training_or_evaluation, "_".join([t, str(tl)])
        )
    ).set_index(["geo_id", "time_id"])
    for k, t, tl in zip(experiments, treatments, treatment_levels)
}
refutation = {
    k1[1]: {
        k2.split("/")[-1]
        .split("_")[0]: pd.read_csv(k2)
        .set_index(["geo_id", "time_id"])
        for k2 in [
            "{}/refutation/{}/{}_{}.csv".format(
                k1[0], training_or_evaluation, ref, "_".join([t, str(tl)])
            )
            for ref in ref_tests
        ]
    }
    for k1, t, tl in zip(experiments, treatments, treatment_levels)
}

rows = []

for exp in experiments:
    ref_sign = dict.fromkeys(["num_obs"] + list(refutation[exp[1]].keys()))
    cate_sign = cate[exp[1]]["significant"]
    ref_sign["num_obs"] = [len(cate[exp[1]])]
    for k2 in refutation[exp[1]].keys():
        # in general, test is failed if a) point estimate is outside test distribution or b) CATE is significant in the first place, but test distribution covers zero
        if k2 not in ["PlaceboTreatmentRefuter", "PropensityThresholdingTest"]:
            sum_nonsig = np.sum(
                (
                    (~refutation[exp[1]][k2][cate_sign]["significant"])
                    & (refutation[exp[1]][k2][cate_sign]["ci_upper"] < 0)
                )
                | (~refutation[exp[1]][k2][~cate_sign]["significant"])
            )
        else:
            # for placebo treatment refuter and propensity thresholding test: Count only how many are not significant overall
            sum_nonsig = ref_sign[k2] = np.sum(~refutation[exp[1]][k2]["significant"])
        ref_sign[k2] = [sum_nonsig]

    if exp[1].startswith("nuts0_nuts1"):
        scope = "nuts0_nuts1"
    else:
        scope = "nuts0"

    split = exp[1].replace(scope, "").split("_")
    treatment = re.search(pattern, exp[0]).group(1).strip("_")

    new_data = {
        "scope": [scope],
        "country": [split[1]],
        "model": [split[2]],
        "treatment": [treatment],
        "outcome": ["_".join(split[-2:])],
        "location": f"{exp[0]}/refutation/{training_or_evaluation}",
    }
    new_data.update(ref_sign)
    rows.append(pd.DataFrame(new_data))


output_df = pd.concat(rows)
output_df.to_csv(
    f"{base_dir}/refutation_results_{training_or_evaluation.split('_')[-1]}.csv",
    index=False,
)
