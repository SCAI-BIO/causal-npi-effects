import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from .causal_model import CausalModel, inference
from .encoders import (
    DragonnetEncoder,
)
from src.utils import checkpoint, validation_split, create_windowed_dataframe
from src.utils.plotting import plot_fit
import torch
import random
import os

# Dragonnet-based model inherits basic functionalities from CausalModel and uses DragonnetEncoder for representation learning


class DragonLearner(CausalModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = DragonnetEncoder(
            **kwargs.pop("config"),
            dim_treatments=1,
        )
        # dragon learner has two treatment levels
        self.T_levels = [f"{self.config['treatment']}_{i}" for i in [0, 1]]
        # DragonLearner uses its outcome heads as y_models and its treatment head as t_model
        self.t_models = {
            f"{self.config['treatment']}_1": self.encoder.model.treatment_head
        }
        self.y_models = {
            f"{self.config['treatment']}_0": self.encoder.model.outcome_head0,
            f"{self.config['treatment']}_1": self.encoder.model.outcome_head1,
        }

    @staticmethod
    def format_model_inputs(*args, **kwargs) -> Tuple[pd.DataFrame, ...]:
        windowed_data = create_windowed_dataframe(*args, **kwargs)
        return windowed_data

    @checkpoint(step_name="fit_causal_model", message="Fitting DragonLearner...")
    @validation_split
    @inference
    def fit(self, **kwargs) -> None:
        self.encoder.fit(**kwargs)

    def cate(
        self, X: pd.DataFrame, variational_inference: bool = False, **kwargs
    ) -> Dict[str, Optional[pd.DataFrame]]:
        assert self.fitted, "{} object must be fitted before calling cate()!".format(
            self.__class__.__name__
        )

        cate_estimates = dict.fromkeys(self.T_levels[1:])
        for t in self.T_levels[1:]:
            # seed must be fixed to ensure that intervals obtained with variational dropout are correct
            seed = random.randint(0, 1000000)
            torch.manual_seed(seed)
            prediction_T_1 = self.y_models[t].predict(
                X, variational_inference=variational_inference
            )
            torch.manual_seed(seed)
            prediction_T_0 = self.y_models[self.T_levels[0]].predict(
                X, variational_inference=variational_inference
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
        eval_counterfactual: bool = True,
    ) -> Optional[Tuple[Tuple[Dict, Dict], Tuple[Dict, Dict]]]:
        if len(self.y_models) == 0:
            logging.warning(
                "Skipping evaluation of y_models because {} does not have any.".format(
                    self.__class__.__name__
                )
            )
            return

        # evaluate y_models
        Y_pred_train, Y_pred_eval = dict.fromkeys(self.T_levels), dict.fromkeys(
            self.T_levels
        )
        # counterfactual
        Y_pred_train_c, Y_pred_eval_c = dict.fromkeys(self.T_levels), dict.fromkeys(
            self.T_levels
        )
        y_models_names = []
        T_dummies_all = pd.DataFrame(
            np.zeros((T.shape[0], len(self.T_levels)), dtype=bool),
            index=T.index,
            columns=self.T_levels,
        )
        T_dummies = pd.get_dummies(T.squeeze(), prefix=self.config["treatment"]).astype(
            bool
        )
        T_dummies_all.loc[:, T_dummies.columns] = T_dummies
        T_eval_dummies_all = pd.DataFrame(
            np.zeros((T_eval.shape[0], len(self.T_levels)), dtype=bool),
            index=T_eval.index,
            columns=self.T_levels,
        )
        if not T_eval.empty:
            T_eval_dummies = pd.get_dummies(
                T_eval.squeeze(), prefix=self.config["treatment"]
            ).astype(bool)
            T_eval_dummies_all.loc[:, T_eval_dummies.columns] = T_eval_dummies

        for i, (k, y_model) in enumerate(self.y_models.items()):
            y_models_names.append("{} ({})".format(y_model.__class__.__name__, str(i)))
            if not X.empty:
                X_subset = X.loc[T_dummies_all.loc[:, k]]
                X_subset_c = X.loc[~T_dummies_all.loc[:, k]]
                if not X_subset.empty:
                    Y_pred_train[k] = pd.DataFrame(
                        y_model.predict(X=X_subset),
                        index=X_subset.index,
                    )
                if not X_subset_c.empty:
                    Y_pred_train_c[k] = pd.DataFrame(
                        y_model.predict(X=X_subset_c),
                        index=X_subset_c.index,
                    )
            if not X_eval.empty:
                X_eval_subset = X_eval.loc[T_eval_dummies_all.loc[:, k]]
                X_eval_subset_c = X_eval.loc[~T_eval_dummies_all.loc[:, k]]
                if not X_eval_subset.empty:
                    Y_pred_eval[k] = pd.DataFrame(
                        y_model.predict(X=X_eval_subset),
                        index=X_eval_subset.index,
                    )
                if not X_eval_subset_c.empty:
                    Y_pred_eval_c[k] = pd.DataFrame(
                        y_model.predict(
                            X=X_eval_subset_c,
                        ),
                        X_eval_subset_c.index,
                    )

        if self.save_to is not None:
            for Y_pred, Y_true, T_filter, dataset in zip(
                [[Y_pred_train, Y_pred_train_c], [Y_pred_eval, Y_pred_eval_c]],
                [Y, Y_eval],
                [T_dummies_all, T_eval_dummies_all],
                ["01_training", "02_evaluation"],
            ):
                for time_step in range(Y.shape[1]):
                    if Y_true.empty:
                        continue
                    if not os.path.isdir(f"{self.save_to}/{dataset}"):
                        os.makedirs(f"{self.save_to}/{dataset}")
                    # save to csv
                    for k, v in Y_pred[0].items():
                        if pd.DataFrame(v).empty:
                            continue
                        pd.DataFrame(v).to_csv(
                            f"{self.save_to}/{dataset}/pred_f_{k}.csv"
                        )
                    if eval_counterfactual:
                        for k, v in Y_pred[1].items():
                            if pd.DataFrame(v).empty:
                                continue
                            pd.DataFrame(v).to_csv(
                                f"{self.save_to}/{dataset}/pred_cf_{k}.csv"
                            )
                    if not self.deactivate_plotting:
                        # observed
                        plot_fit(
                            Y_true=Y_true,
                            Y_pred_dict=Y_pred[0],
                            T_filter=T_filter,
                            metrics=["mape", "mse"],
                            models=y_models_names,
                            time_step=time_step,
                            counterfactual=False,
                            save_to=f"{self.save_to}/{dataset}",
                        )
                        # counterfactual
                        if eval_counterfactual:
                            plot_fit(
                                Y_true=Y_true,
                                Y_pred_dict=Y_pred[1],
                                T_filter=T_filter,
                                metrics=["mape", "mse"],
                                models=y_models_names,
                                time_step=time_step,
                                counterfactual=True,
                                save_to=f"{self.save_to}/{dataset}",
                            )
        return (Y_pred_train, Y_pred_train_c), (Y_pred_eval, Y_pred_eval_c)
