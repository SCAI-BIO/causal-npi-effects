from dataclasses import dataclass
from typing import List, Optional, Dict

"""
Data classes for typing. 
"""


@dataclass
class General:
    experiment_name: str
    input_data: str
    output_dir_base: str
    out_dir: str
    substeps: List[str]
    deactivate_plotting: bool


@dataclass
class Data:
    geo_id: str
    time_id: str
    treatment: str
    treatment_levels: int
    outcome: str
    outcome_lag: int
    window_size: int
    time_predictors: List[str]
    static_real_predictors: List[str]
    countries: List[str]
    regions: Optional[List[str]]
    period_fit: List[str]
    period_eval: List[str]
    validation_start: Optional[str]


@dataclass
class Inference:
    inference: str
    inference_config: Dict


@dataclass
class CausalModel:
    model_class: str
    model_config: Dict


@dataclass
class BaseLearner:
    model_class: str
    model_config: Dict
    tuned: bool


@dataclass
class Model:
    causal_model: CausalModel
    inference: Inference


@dataclass
class Shap:
    defined_periods: Optional[List[List[str]]]
    interaction_variable: Optional[str]
    subsample_frac: Optional[float]


@dataclass
class PipelineConfig:
    general: General
    data: Data
    model: Model
    refutation: List[str]
    shap: Shap
