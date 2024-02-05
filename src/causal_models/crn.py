import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Union
from .causal_model import CausalModel, inference
from .encoders import (
    CRNEncoder,
)
from src.utils import checkpoint, validation_split, create_windowed_dataframe
from src.utils.plotting import plot_fit
import torch
import random
import os

# CRN-based model inherits basic functionalities from CausalModel and uses CRNEncoder for representation learning


class CRNLearner(CausalModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = kwargs.pop("config")
        self.encoder = CRNEncoder(
            **config,
            dim_treatments=len(config["treatment"])
            if type(config["treatment"]) == list
            else 1,
        )

        if type(self.config["treatment"]) == list:
            self.T_levels = ["0"] + [f"{t}_1" for t in self.config["treatment"]]
        else:
            self.T_levels = ["0", f"{self.config['treatment']}_{1}"]
        # CRNLearner uses its outcome heads as y_models and its treatment head as t_model
        self.t_models = dict.fromkeys(
            self.T_levels[1:], self.encoder.model.treatment_head
        )
        # same model for all treatments
        self.y_models = dict.fromkeys(
            self.T_levels[1:], self.encoder.model.outcome_head
        )

    @staticmethod
    def format_model_inputs(*args, **kwargs) -> Tuple[pd.DataFrame, ...]:
        """
        CRN (unlike Dragonnet learner) also works with multiple treatments
        Next to the defined predictors, it uses past Ts as predictor (include_treatment_history=True)
        """
        windowed_data = create_windowed_dataframe(
            *args, **kwargs, include_treatment_history=True
        )
        return windowed_data

    @checkpoint(step_name="fit_causal_model", message="Fitting CRNLearner...")
    @validation_split
    @inference
    def fit(
        self, T: pd.Series, T_val: pd.Series = pd.Series(dtype=object), **kwargs
    ) -> None:
        self.encoder.fit(T=T, T_val=T_val, **kwargs)

    def cate(
        self,
        X: pd.DataFrame,
        T: Optional[Union[pd.DataFrame, pd.Series]] = None,
        variational_inference: bool = False,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        # TODO: adapt shap analysis so that it can also be performed for multitreatment setting??
        assert self.fitted, "{} object must be fitted before calling cate()!".format(
            self.__class__.__name__
        )
        if T is None:
            assert (
                len(self.T_levels) == 2
            ), "For multitreatment CRN, T must be explicitly passed to .cate()"
            T = pd.DataFrame(np.ones(shape=X.shape[0]), columns=[self.T_levels[1]])

        cate_estimates = {}
        if type(T) == pd.DataFrame:
            if T.shape[1] > 1:
                self.T_levels = ["0"] + [f"{t}_1" for t in T.columns]
            T.columns = self.T_levels[1:]
        for t in self.T_levels[1:]:
            # copy T and replace column t with all 1 or all 0 for CATE prediction, keeping all other treatment dimensions as they were
            T_1, T_0 = T.copy(), T.copy()
            T_1[t] = 1
            T_0[t] = 0
            # seed must be fixed to ensure that intervals obtained with variational dropout are correct
            seed = random.randint(0, 1000000)
            torch.manual_seed(seed)
            prediction_T_1 = self.encoder.model.outcome_head.predict(
                X, T_1, variational_inference=variational_inference
            )
            torch.manual_seed(seed)
            prediction_T_0 = self.encoder.model.outcome_head.predict(
                X, T_0, variational_inference=variational_inference
            )
            cate_estimates[t] = pd.DataFrame(
                prediction_T_1 - prediction_T_0,
                index=X.index,
            )
        return cate_estimates

    @checkpoint(
        step_name="evaluate_causal_model",
        message="Evaluating outcome predictions and creating plots...",
        subdir="y_models",
    )
    def eval_and_plot_y_models(
        self,
        X: pd.DataFrame = pd.DataFrame(),
        Y: pd.DataFrame = pd.DataFrame(),
        T: pd.Series = pd.Series(dtype=object),
        X_eval: pd.DataFrame = pd.DataFrame(),
        Y_eval: pd.DataFrame = pd.DataFrame(),
        T_eval: pd.Series = pd.Series(dtype=object),
    ) -> Optional[Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict]]]:
        Y_pred_train = self.encoder.model.outcome_head.predict(X, T)
        if not X_eval.empty:
            Y_pred_eval = self.encoder.model.outcome_head.predict(X_eval, T_eval)
        else:
            Y_pred_eval = pd.DataFrame()
        if self.save_to is not None:
            for Y_pred, Y_true, T_filter, dataset in zip(
                [Y_pred_train, Y_pred_eval],
                [Y, Y_eval],
                [
                    pd.DataFrame(
                        np.ones(T.shape[0]),
                        columns=["OutcomeHead"],
                        index=T.index,
                        dtype=bool,
                    ),
                    pd.DataFrame(
                        np.ones(T_eval.shape[0]),
                        columns=["OutcomeHead"],
                        index=T_eval.index,
                        dtype=bool,
                    ),
                ],
                ["01_training", "02_evaluation"],
            ):
                if Y_true.empty:
                    continue
                if not os.path.isdir(f"{self.save_to}/{dataset}"):
                    os.makedirs(f"{self.save_to}/{dataset}")
                # save to csv
                Y_pred.to_csv(
                    f"{self.save_to}/{dataset}/pred_f_{self.config['treatment']}.csv"
                )
                if not self.deactivate_plotting:
                    for time_step in range(Y.shape[1]):
                        # observed
                        plot_fit(
                            Y_true=Y_true,
                            Y_pred_dict={"OutcomeHead": Y_pred},
                            T_filter=T_filter,
                            metrics=["mape", "mse"],
                            models=["OutcomeHead"],
                            time_step=time_step,
                            counterfactual=False,
                            save_to=f"{self.save_to}/{dataset}",
                        )
        return ({self.T_levels[-1]: Y_pred_train}, {}), (
            {self.T_levels[-1]: Y_pred_eval},
            {},
        )
