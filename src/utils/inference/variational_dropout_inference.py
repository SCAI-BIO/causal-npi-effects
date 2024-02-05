import pandas as pd
import numpy as np
from typing import Union, List, Dict


class VariationalDropoutInference:
    """
    A class that encapsulates variational dropout inference.
    """

    def __init__(self, wrapped, n_samples: int = 100, **kwargs) -> None:
        self.n_samples = n_samples
        self.wrapped = wrapped
        self.config = wrapped.config

    def fit(*args, **kwargs):
        # This inference object does not need to be fitted because inference is done by dropout within one model.
        pass

    def get_outcome_distribution(
        self, X: pd.DataFrame, T: Union[pd.DataFrame, pd.Series]
    ) -> Dict[str, np.ndarray]:
        """Uses variational dropout to make self.n_samples outcome predicions for given X and T

        Parameters
        ----------
        X : pd.DataFrame
            Formatted confounders / effect modifiers
        T : Union[pd.DataFrame, pd.Series]
            Formatted treatment(s)

        Returns
        -------
        Dict[str, np.ndarray]
            self.n_samples outcome predictions under MC dropout for each outcome
        """
        Y_pred_dict = {k: [] for k in self.wrapped.y_models.keys()}
        for k, y_model in self.wrapped.y_models.items():
            for _ in range(self.n_samples):
                Y_pred_dict[k].append(
                    y_model.predict(X=X, T=T, variational_inference=True)
                )
        return {k: np.stack(Y_pred_dict[k], -1) for k in Y_pred_dict.keys()}

    def get_cate_distribution(
        self,
        X: pd.DataFrame,
        levels: List[str],
        T: Union[pd.Series, pd.DataFrame] = pd.DataFrame(),
    ) -> Dict[str, np.ndarray]:
        """Uses variational dropout to make self.n_samples CATE predicions for given X and T

        Parameters
        ----------
        X : pd.DataFrame
            Formatted confounders / effect modifiers
        levels : List[str]
            List of treatment levels
        T : Union[pd.Series, pd.DataFrame], optional
            Formatted treatment(s), by default pd.DataFrame()

        Returns
        -------
        Dict[str, np.ndarray]
            self.n_samples CATE predictions under MC dropout for each outcome
        """
        cate_dict = {k: [] for k in levels}
        for _ in range(self.n_samples):
            cate = self.wrapped.cate(X=X, T=T, variational_inference=True)
            for k in cate.keys():
                cate_dict[k].append(cate[k])
        return {k: np.stack(cate_dict[k], -1) for k in cate_dict.keys()}
