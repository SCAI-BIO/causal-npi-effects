import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from config.config import PipelineConfig
import datetime
import logging
import pickle
import os
from src.causal_models import *

cs = ConfigStore.instance()
cs.store(name="pipeline_config", node=PipelineConfig)

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="./config", config_name="config.yaml", version_base=None)
def main(cfg: PipelineConfig) -> None:
    """Pipeline to fit causal models on time series data. Uses config passed on by hydra.
    cfg.general.substeps specifies which of the following are executed:
    1.) fit_causal_model: Fits the selected causal estimator with inference to quantify model uncertainty.
    2.) evaluate_causal_model: Creates plots (if not disabled via cfg.general.deactivate_plotting) and output tables (CATE and outcome predictions).
    3.) refutation: Runs specified refutation tests and creates plots (if not disabled via cfg.general.deactivate_plotting) and output tables with refutation results.
    4.) shap_analysis: Performs SHAP analysis for given periods.

    Parameters
    ----------
    cfg : PipelineConfig
        Config object passed on by hydra.
    """
    # Import data
    df = pd.read_csv(cfg.general.input_data, parse_dates=[cfg.data.time_id])
    df = df[df["country_id"].isin(cfg.data.countries)]
    if cfg.data.regions:
        df = df[df[cfg.data.geo_id].isin(cfg.data.regions)]
    # format train test split
    if len(cfg.data.period_fit) == 2:
        period_fit = [
            pd.to_datetime(datetime.datetime.strptime(x + "-1", "%Y-%W-%w"))
            for x in cfg.data.period_fit
        ]
        df_fit = df[df[cfg.data.time_id].between(period_fit[0], period_fit[1])]
    else:
        logging.warning(
            "No valid or non-empty fitting period given ({}).".format(
                cfg.data.period_fit
            )
        )
        df_fit = None
    if len(cfg.data.period_eval) == 2:
        period_eval = [
            pd.to_datetime(datetime.datetime.strptime(x + "-1", "%Y-%W-%w"))
            for x in cfg.data.period_eval
        ]
        df_eval = df[df[cfg.data.time_id].between(period_eval[0], period_eval[1])]
    else:
        logging.warning(
            "No valid or non-empty evaluation period given ({}).".format(
                cfg.data.period_eval
            )
        )
        df_eval = None

    # create datasets
    fit_inputs = eval(cfg.model.causal_model.model_class).format_model_inputs(
        df=df_fit, **cfg.data
    )
    eval_inputs = eval(cfg.model.causal_model.model_class).format_model_inputs(
        df=df_eval, **cfg.data
    )

    # some configs can only be set at runtime
    causal_model_config = OmegaConf.to_container(
        cfg.model.causal_model.model_config, resolve=True
    )
    causal_model_config.update(
        {
            "outcomes": [
                cfg.data.outcome + str(i + 1) for i in range(cfg.data.outcome_lag)
            ],
            "treatment_levels": len(
                df[
                    cfg.data.treatment
                    if type(cfg.data.treatment) == str
                    else cfg.data.treatment[0]
                ].unique(),
            ),
            "input_size": len(cfg.data.time_predictors)
            + len(cfg.data.static_real_predictors),
            "output_size": cfg.data.outcome_lag,
        }
    )

    # Initialize causal model
    model = eval(cfg.model.causal_model.model_class)(config=causal_model_config)

    # try to load model if it has already been fitted; else, fit model
    if os.path.isfile(
        "{}/fit_causal_model/fitted_model.pkl".format(cfg.general.out_dir)
    ):
        with open(
            "{}/fit_causal_model/fitted_model.pkl".format(cfg.general.out_dir), "rb"
        ) as fitted_model_file:
            model = pickle.load(fitted_model_file)
            model.fitted = True
        logging.info(
            "Found fitted model in out_dir. Loading this model instead of fitting a new one."
        )
    else:
        # Fit the model
        model.fit(
            *fit_inputs,
            inference=cfg.model.inference.inference,
            inference_config=cfg.model.inference.inference_config,
            save_to=cfg.general.out_dir,
            substeps=cfg.general.substeps,
            validation_start=cfg.data.validation_start,
        )

    # Model evaluation and plotting
    model.eval_and_plot_cate(
        X=fit_inputs[0],
        X_test=eval_inputs[0],
        T=fit_inputs[2],
        T_test=eval_inputs[2],
        save_to=cfg.general.out_dir,
        deactivate_plotting=cfg.general.deactivate_plotting,
        substeps=cfg.general.substeps,
    )
    model.eval_and_plot_y_models(
        *fit_inputs,
        *eval_inputs,
        save_to=cfg.general.out_dir,
        deactivate_plotting=cfg.general.deactivate_plotting,
        substeps=cfg.general.substeps,
    )

    # delete inference object to clear memory (can become quite large for bootstrapping)
    del model._inference_object
    model._inference_object = None

    refutation_tests = OmegaConf.to_container(cfg.refutation, resolve=True)
    # Refutation tests
    model.perform_refutation_tests(
        *fit_inputs,
        *eval_inputs,
        refutation_tests=refutation_tests,
        causal_model_class=cfg.model.causal_model.model_class,
        causal_model_config=causal_model_config,
        validation_start=cfg.data.validation_start,
        save_to=cfg.general.out_dir,
        deactivate_plotting=cfg.general.deactivate_plotting,
        substeps=cfg.general.substeps,
    )

    # SHAP analyis
    model.shap_values(
        X=fit_inputs[0],
        X_eval=eval_inputs[0],
        T=fit_inputs[2],
        T_eval=eval_inputs[2],
        defined_periods=cfg.shap.defined_periods,
        interaction_variable=cfg.shap.interaction_variable,
        subsample_frac=cfg.shap.subsample_frac,
        save_to=cfg.general.out_dir,
        deactivate_plotting=cfg.general.deactivate_plotting,
        substeps=cfg.general.substeps,
    )


if __name__ == "__main__":
    main()
