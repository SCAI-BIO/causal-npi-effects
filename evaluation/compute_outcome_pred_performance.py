# %%
import pandas as pd
import numpy as np
import datetime
import glob
import pickle
import re
import tqdm
import yaml
from sklearn.metrics import mean_absolute_percentage_error

import sys

sys.path.append("..")


base_dir = "../results"
base_data = "../data/data.csv"
base_dirs = [base_dir, f"{base_dir}/swedish_strategy"]
base_data = "../data/data.csv"
df = pd.read_csv(base_data, parse_dates=["date"])

models = ["CRNLearner", "DragonLearner"]
levels = ["nuts0", "nuts0_nuts1"]
# countries = ["DE", "FR", "ES", "PL", "NL", "BE", "SE"]
outcomes = ["r_number7", "r_number14"]
datasets = ["01_training", "02_evaluation"]


for model in models:
    for level in levels:
        # for country in countries:
        for outcome in outcomes:
            for dataset in datasets:
                print(f"{model}, {level}, {outcome}, {dataset}")
                # print(f"{model}, {level}, {country}, {outcome}, {dataset}") # per country
                dirs = []
                experiments = []
                # create list with directories
                for d in base_dirs:
                    dirs.extend(glob.glob(f"{d}/{level}/*/{model}*{outcome}"))
                    # dirs.extend(glob.glob(f"{d}/{level}/{country}/{model}*{outcome}"))  # per country

                # make a list of dictionaries
                for d in dirs:
                    # open config for prediction period, outcome_lag, window_size
                    with open(
                        glob.glob(f"{d}/.logs/*/.hydra/config.yaml")[0], "r"
                    ) as f:
                        config = yaml.safe_load(f)
                    if dataset == "01_training":
                        pred_period = config["data"]["period_fit"]
                    else:
                        pred_period = config["data"]["period_eval"]
                    outcome_var = config["data"]["outcome"]
                    outcome_lag = config["data"]["outcome_lag"]
                    window_size = config["model"]["causal_model"]["model_config"][
                        "window_size"
                    ]
                    time_predictors = config["data"]["time_predictors"]
                    static_real_predictors = config["data"]["static_real_predictors"]

                    # load model
                    with open(f"{d}/fit_causal_model/fitted_model.pkl", "rb") as f:
                        model_obj = pickle.load(f)

                    new_dict = {
                        "country": d.split("/")[-2],
                        "npi": re.findall("Learner_(.*)_r_number", d)[0],
                        "time_predictors": time_predictors,
                        "static_real_predictors": static_real_predictors,
                        "pred_period": pred_period,
                        "window_size": window_size,
                        "outcome": outcome_var,
                        "outcome_lag": outcome_lag,
                        "model": model_obj,
                    }

                    experiments.append(new_dict)

                # load data
                for exp in experiments:
                    pred_period = [
                        pd.to_datetime(datetime.datetime.strptime(x + "-1", "%Y-%W-%w"))
                        for x in exp["pred_period"]
                    ]
                    data = df[df["country_id"] == exp["country"]]
                    # only at country level
                    data = data[data["country_id"] == data["geo"]]
                    data = (
                        data[data["date"].between(pred_period[0], pred_period[1])]
                        .sort_values(by=["geo", "date"])
                        .reset_index()
                    )

                    metadata = {
                        "geo_id": "geo",
                        "time_id": "date",
                        "time_predictors": exp["time_predictors"],
                        "static_real_predictors": exp["static_real_predictors"],
                        "outcome": exp["outcome"],
                        "outcome_lag": exp["outcome_lag"],
                        "treatment": exp["npi"],
                        "window_size": exp["window_size"],
                    }

                    exp["X"], exp["Y"], exp["T"] = exp["model"].format_model_inputs(
                        df=data, **metadata
                    )

                # make predictions
                pred_list = []
                ground_truth_list = []
                for exp in tqdm.tqdm(experiments):
                    predictions = exp[
                        "model"
                    ]._inference_object.get_outcome_distribution(
                        X=exp["X"], T=exp["T"]
                    )[
                        f"{exp['npi']}_1"
                    ][
                        :, -1, :
                    ]
                    pred_list.append(predictions)
                    ground_truth_list.append(exp["Y"].values[:, -1])

                pred = np.vstack(pred_list)
                gt = np.hstack(ground_truth_list)
                gt = np.repeat(gt[..., np.newaxis], 100, axis=-1)

                # compute MAPE
                mape = (
                    mean_absolute_percentage_error(
                        y_true=gt, y_pred=pred, multioutput="raw_values"
                    )
                    * 100
                )

                np.set_printoptions(suppress=True)
                print(f"MAPE: {np.mean(mape): .7f} (+/- {np.std(mape): .7f})")
