# Estimating the causal impact of non-pharmaceutical interventions on COVID-19 spread in seven EU countries via machine learning
Code accompanying a [research paper in Scientific Reports](https://www.nature.com/articles/s41598-025-88433-2) by Jannis Guski, Jonas Botz and Prof. Dr. Holger Fröhlich. If you have any questions regarding the code or paper, please feel free to get in touch (jannis.guski@scai.fraunhofer.de).

## Setup
Navigate to cloned repository and call `mamba env create -f causal-npi-effects.yml` to generate mamba environment, followed by `mamba activate causal-npi-effects` to activate it. 

## Run
Configuration parsing is based on `hydra`, and folder `./config/` provides configuration objects. The input data can be found in `./data/`.

The easiest way to reproduce an experiment is using the `+experiment=<...>` argument. Type `python pipeline.py --help` for an overview of the experiments reported in the paper and additional arguments.
For example, to run the pseudo-prospective scenario planning analysis for `npi_schools` during the second wave in Germany:

```
python pipeline.py +experiment=scenario_planning/2nd_wave/crnlearner_npi_schools_DE
```


If you want to create your own experiment, make sure to fill all `???` in the configuration. Refer to existing experiments for guidance.


`pipeline.py` is the interface of the code base and successively executes a number of predefined substeps.

1. `fit_causal_model`: Fits the selected causal estimator with inference to quantify model uncertainty.

2. `evaluate_causal_model`: Creates plots and output tables (CATE and outcome predictions).

3. `refutation`: Runs specified refutation tests and creates plots and output tables with refutation results.

4. `shap_analysis`: Performs SHAP analysis for given periods.


While each experiment has a predefined list of steps to be run, you can adapt the pipeline on command line call. For example, to fit and evaluate the model, skip refutation but run SHAP analysis afterwards:

```
python pipeline.py +experiment=<...> general.substeps=[fit_causal_model,evaluate_causal_model,shap_analysis]
```


By default, plots will be created and saved for every step in the pipeline. To deactivate plotting, type

```
python pipeline.py +experiment=<...> general.deactivate_plotting=True
```

## Evaluate
Once you have run a number of experiments, you can use the scripts in `./evaluation/` to combine separate results to the final numbers/plots shown in the paper. Just adjust the paths as required.
