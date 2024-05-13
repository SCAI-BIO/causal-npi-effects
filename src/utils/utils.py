from typing import Optional, List, Callable, Union, Any, Tuple
import numpy as np
import pandas as pd
from shap._explanation import Explanation
import logging
import os
from functools import wraps
import datetime


def nonparametric_significance_test(
    point_estimate: np.ndarray,
    distribution: np.ndarray,
    shift_to_value: Optional[int] = None,
    tailed: str = "two-tailed",
) -> np.ndarray:
    """Performs non-parameteric significance test of the given point estimate relative to the distribution.

    Parameters
    ----------
    point_estimate : np.ndarray
        Point estimate to test significance for.
    distribution : np.ndarray
        Given distribution (e.g., from inference object or refutation test)
    shift_to_value : Optional[int], optional
        Value that test distribution is shifted to.
    tailed : str, optional
        "left", "right" or "two-tailed", by default "two-tailed"

    Returns
    -------
    np.ndarray
       p-values

    Raises
    ------
    ValueError
        Raises error if invalid value given for tailed argument.
    """
    if shift_to_value is not None:
        means = distribution.mean(axis=-1)
        means = np.repeat(means[..., np.newaxis], distribution.shape[-1], axis=-1)
        distribution = distribution - means + shift_to_value

    point_estimate = np.repeat(
        point_estimate[..., np.newaxis], distribution.shape[-1], axis=-1
    )

    if tailed == "two-tailed":
        pvalues_left = (point_estimate > distribution).sum(
            axis=-1
        ) / distribution.shape[-1]
        pvalues_right = (point_estimate < distribution).sum(
            axis=-1
        ) / distribution.shape[-1]
        pvalues = np.minimum(pvalues_left, pvalues_right) * 2
    elif tailed == "left":
        pvalues = (point_estimate > distribution).sum(axis=-1) / distribution.shape[-1]
    elif tailed == "right":
        pvalues = (point_estimate < distribution).sum(axis=-1) / distribution.shape[-1]
    else:
        raise ValueError(
            "Invalid value passed for tailed in nonparameteric_significance_test \
                       (only 'two-tailed', 'left' or 'right' are valid.)"
        )
    return pvalues


def aggregate_shap_values(shap_values: Explanation) -> Explanation:
    """Creates a new SHAP Explanation object by aggregating over time steps
    Parameters:
            shap_values : Explanation
                Raw SHAP Explanation.
    Returns
    -------
    Explanation
        Aggregated SHAP Explanation.
    """
    # read out indices of the features
    feature_indices = {}
    for i, feature in enumerate(shap_values.feature_names):
        if feature[0] not in feature_indices.keys():
            feature_indices[feature[0]] = [i]
        else:
            feature_indices[feature[0]].append(i)

    # aggregate values and data
    values = []
    data = []
    feature_names = []
    for new_feature in feature_indices.keys():
        feature_names.append(new_feature)
        values.append(shap_values.values[:, feature_indices[new_feature]].sum(axis=1))
        data.append(shap_values.data[:, feature_indices[new_feature]].mean(axis=1))
    values = np.stack(values, axis=1)
    data = np.stack(data, axis=1)

    # initialize the new SHAP explainer
    new_shap_values = Explanation(values=values, data=data, feature_names=feature_names)

    return new_shap_values


def checkpoint(
    step_name: str, subdir: Optional[str] = None, message: Optional[str] = None
) -> Callable:
    """A decorator for logging, which also checks if the decorated step should be executed at all."""

    def decorator(func: Callable) -> Callable:
        def wrapper(
            obj,
            *args,
            skip_checkpoint: bool = False,
            substeps: Optional[List[str]] = None,
            save_to: str = ".",
            deactivate_plotting: bool = False,
            fold: Optional[int] = None,
            **kwargs,
        ) -> Any:
            if skip_checkpoint:
                pass
            elif substeps is not None and step_name not in substeps:
                logging.warning(
                    " [{}] Skipped because missing in the --substeps argument".format(
                        step_name
                    )
                )
                return
            else:
                if message is not None:
                    if fold is not None:
                        logging.info(
                            " [{}] [Fold {}] {}".format(step_name, fold, message)
                        )
                    else:
                        logging.info(" [{}] {}".format(step_name, message))
                if save_to is not None:
                    if subdir is None:
                        obj.save_to = "{}/{}".format(save_to, step_name)
                    else:
                        obj.save_to = "{}/{}/{}".format(save_to, step_name, subdir)
                    if not os.path.exists(obj.save_to):
                        os.makedirs(obj.save_to)
                # plots can be disabled
                obj.deactivate_plotting = deactivate_plotting
            try:
                return func(obj, *args, **kwargs)
            except Exception as e:
                logging.warning(
                    "{} terminated with the following exception: {}".format(
                        func.__name__, e
                    )
                )

        return wrapper

    return decorator


def validation_split(m: Callable[..., None]) -> Callable:
    """
    A decorator to perform one static train-val split
    (does not do anything if validation data already provided or no validation_start is given)
    """

    @wraps(m)
    def call(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        T: Union[pd.DataFrame, pd.Series],
        X_val: pd.DataFrame = pd.DataFrame(),
        Y_val: pd.DataFrame = pd.DataFrame(),
        T_val: Union[pd.DataFrame, pd.Series] = pd.DataFrame(),
        validation_start: Optional[str] = None,
        **kwargs,
    ) -> None:
        if X_val.empty and validation_start is not None:
            validation_start_date = pd.to_datetime(
                datetime.datetime.strptime(validation_start + "-1", "%Y-%W-%w")
            )
            X, X_val = (
                X[X.index.get_level_values("time_id") < validation_start_date],
                X[X.index.get_level_values("time_id") >= validation_start_date],
            )
            Y, Y_val = (
                Y[Y.index.get_level_values("time_id") < validation_start_date],
                Y[Y.index.get_level_values("time_id") >= validation_start_date],
            )
            T, T_val = (
                T[T.index.get_level_values("time_id") < validation_start_date],
                T[T.index.get_level_values("time_id") >= validation_start_date],
            )

        m(self, X=X, Y=Y, T=T, X_val=X_val, Y_val=Y_val, T_val=T_val, **kwargs)

    return call


def windowing(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Filters formatted dataframe with prediction windows to match window_size of model

    Parameters
    ----------
    df : pd.DataFrame
        Formatted dataframe with prediction windows
    window_size : int
        window_size parameter of the model

    Returns
    -------
    pd.DataFrame
        Filtered dataframe

    Raises
    ------
    RuntimeError
        Raised if window_size parameter of model is larger than window size available in the dataframe
    """
    # make sure that the max window size from df actually covers the model-specific window_size parameter
    max_window_size = len(
        df.loc[:, df.columns.get_level_values("step") != "static"]
        .columns.get_level_values("step")
        .unique()
    )
    if max_window_size < window_size:
        raise RuntimeError(
            "Found lower max window size in dataframe input ({}) compared to a model-specific window_size ({}). Please set the window_size parameter to at least {}.".format(
                max_window_size, window_size, window_size
            )
        )
    # only keep window_size entries for each predictor in df
    df = df.groupby(level=0, axis=1, group_keys=False).apply(
        lambda x: x.iloc[:, -window_size:]
    )
    return df


def create_windowed_dataframe(
    df: Optional[pd.DataFrame],
    time_predictors: List[str],
    static_real_predictors: List[str],
    treatment: Optional[Union[str, List[str]]] = None,
    geo_id: str = "geo",
    time_id: str = "date",
    outcome: Optional[str] = "r_number",
    outcome_lag: int = 7,
    window_size: int = 7,
    include_treatment_history: bool = False,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates X, Y and T according to the given parameters.
    Parameters:
        df : Optional[pd.DataFrame]
            A pandas data frame containing all the data.
        time_predictors : List[str]
            A list of variables to be used as time-dependent predictors (will be lagged).
        static_real_predictors : List[str]
            A list of variables to be used as time-independent predictors (will not be lagged).
        geo_id : str
            A string denoting the name of the geo_id column (to be used for multiindex).
        time_id : str
            A string denoting the name of the time_id column (to be used for multiindex).
        outcome : Optional[str]
            A string denoting the outcome variable.
        outcome_lag : int
            Number of steps by which outcome will be lagged. If 0, only use at time step 0. Otherwise, create 1 to outcome_lag.
        treatment: Optional[Union[str, List[str]]]
            String or list of strings denoting the treatment variables.
        window_size : int
            The size of each window.
        include_window_history : bool
            If true, then given treatments will be shifted by one and included as past treatment in X
    Returns:
        X : pd.DataFrame
        Y : pd.DataFrame
        T : pd.DataFrame
    """
    if df is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df.set_index([geo_id, time_id])
    use_vars = list(time_predictors + static_real_predictors)
    if outcome is not None and not outcome in use_vars:
        use_vars += [outcome]
    if treatment is not None:
        if type(treatment) == str:
            use_vars += [treatment]
        else:
            use_vars += treatment
    # skip over past treatments if they have already been created
    df = df[[v for v in use_vars if not v.startswith("past")]]

    # include treatment history
    if treatment is not None and include_treatment_history:
        if type(treatment) == str:
            treatment = [treatment]
        # add past treatment values as predictors
        df[[f"past_{t}" for t in treatment]] = df.groupby(level=geo_id)[
            list(treatment)
        ].shift(1)
        df.dropna(inplace=True)
        if not any(
            set([f"past_{t}" for t in treatment]).intersection(set(time_predictors))
        ):
            time_predictors.extend([f"past_{t}" for t in treatment])

    static_real_predictors_df = df[static_real_predictors]
    groups = df.index.levels[0].unique()

    # compile a data frame that looks back window_size time steps
    X_df, Y_df, T_df = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(
            {},
            index=pd.MultiIndex.from_tuples([], names=["geo_id", "time_id"]),
            dtype=int,
        ),
    )
    for r in groups:
        # keep only time predictors for one region
        time_predictors_r = df[time_predictors].loc[r]
        # create shifts within window
        shifted_X_list = []
        for s in range(1, window_size):
            shifted = time_predictors_r.shift(s)
            shifted.columns = pd.MultiIndex.from_product(
                [shifted.columns, [-s]], names=["predictor", "step"]
            )
            shifted_X_list.append(shifted)
        time_predictors_r.columns = pd.MultiIndex.from_product(
            [time_predictors_r.columns, [0]], names=["predictor", "step"]
        )
        # create repeated representation of static variables
        static_real_predictors_r = static_real_predictors_df.loc[r]

        static_real_predictors_r = pd.DataFrame(
            static_real_predictors_r.values,
            columns=pd.MultiIndex.from_product(
                [list(static_real_predictors_r.columns), ["static"]],
                names=["predictor", "step"],
            ),
            index=time_predictors_r.index,
        )

        # concatenate everything
        X_concat = pd.concat(
            [static_real_predictors_r, time_predictors_r] + shifted_X_list, axis=1
        )
        # drop NA
        X_concat = X_concat.dropna()
        # sort columns
        X_concat = X_concat.sort_index(axis=1, level=[0, 1], ascending=[True, True])
        # put region back into index
        X_concat.index = pd.MultiIndex.from_product(
            [[r], X_concat.index], names=["geo_id", "time_id"]
        )

        # drop outcomes for which there is no prediction window
        # TODO: Maybe create a method for the shifting (is used here an in the random common cause refuter)
        if outcome is not None:
            if outcome_lag > 0:
                shifted_Y_list = []
                for s in range(1, outcome_lag + 1):
                    shifted = df[[outcome]].loc[r].shift(-s)
                    shifted.columns = pd.MultiIndex.from_product(
                        [[outcome], [s]], names=["outcome", "step"]
                    )
                    shifted_Y_list.append(shifted)
                Y_concat = pd.concat(shifted_Y_list, axis=1)
            else:
                Y_concat = df[[outcome]].loc[r]
                Y_concat.columns = pd.MultiIndex.from_product(
                    [[outcome], [0]], names=["outcome", "step"]
                )
            # drop NA
            Y_concat = Y_concat[-X_concat.shape[0] :]
            Y_concat = Y_concat.dropna()
            X_concat = X_concat[: Y_concat.shape[0]]
            Y_concat.index = X_concat.index
            Y_df = pd.concat([Y_df, Y_concat])

        if type(treatment) == str:
            treatment = [treatment]
        treatment_df = df[treatment].loc[r]
        treatment_df = treatment_df[-X_concat.shape[0] :]
        treatment_df.index = X_concat.index
        T_df = pd.concat([T_df, treatment_df])

        X_df = pd.concat([X_df, X_concat])

    # drop windows for which treatment is negative (defined as missing)
    return X_df.loc[T_df.squeeze()>=0], Y_df.loc[T_df.squeeze()>=0], T_df.loc[T_df.squeeze()>=0]
