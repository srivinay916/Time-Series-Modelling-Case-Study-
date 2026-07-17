Time-Series Modelling Case Study — Forecasting German National Electricity Load
Forecasting Germany's national electricity demand two years ahead, and comparing benchmark,
statistical, machine-learning and deep-learning approaches on the same test window.
Built with: Python · pandas · statsmodels · scikit-learn · TensorFlow/Keras · matplotlib
---
Overview
This project builds an end-to-end weekly load-forecasting pipeline on public German grid data
and asks a deliberately hard question: can any model meaningfully beat a seasonal-naive
benchmark at a two-year horizon? Eight models are trained on 2015–2018 and evaluated on a
held-out 104-week (two-year) test set spanning October 2018 to September 2020 — a window that,
importantly, contains the COVID-19 demand shock of 2020.
The headline finding is that no model meaningfully beats seasonal naive on RMSE at this
horizon, that the model minimising in-sample AIC actually forecasts worst, and that the
dominant source of test error is not model choice but a structural break in the data.
Problem framing
Target: total weekly electricity load for Germany (MWh).
Horizon: 104 weeks (two years), forecast from a single fixed origin.
Seasonal period: `m = 52` (annual cycle in weekly data), derived from the ACF/STL — not
given in the brief.
Working frequencies: raw data is 60-minute; it is aggregated to daily and weekly
for Parts 1–5, and the hourly series is used directly for the LSTM (Part 6).
Data
Source	Data	Notes
Open Power System Data	DE actual load, 60-min	2015-01-01 → Oct 2020; binned to daily/weekly
Open-Meteo Historical Weather API	Berlin daily mean temperature	Used as a national proxy; converted to weekly heating/cooling degree days
Both datasets are downloaded programmatically at run time — no data files need to be committed.
Approach
Four model families are compared, all on the same chronological split and evaluated with MAE,
RMSE, MASE and forecast bias:
Benchmarks — mean, naive, seasonal naive, and drift forecasts.
SARIMA — order selected by exhaustive AIC grid search over `p, q ∈ \[0,6]`, `d ∈ {0,1,2}`,
with seasonal terms and a fixed seasonal difference `D = 1` (588 candidates).
SARIMAX + temperature — a parsimonious `(1,0,1)(0,1,1,52)` model with exogenous heating /
cooling degree days. This is a conditional (explanatory) forecast, since it uses observed
test-period temperature that would not be known at a true forecast origin.
Feature-based regression — Random Forest and Gradient Boosting on cyclic week-of-year
terms, degree days, and seasonal load lags (52 and 104 weeks), forecast recursively to avoid
look-ahead leakage.
LSTM — trained on the hourly series (168-hour input → 24-hour day-ahead output) with a
small hyper-parameter search. Reported separately because it is a day-ahead task, not
horizon-comparable to the models above.
Key results
Accuracy on the 104-week test set (weekly load, MWh), sorted best → worst by RMSE.
Bias > 0 means the model over-forecasts.
Model	MAE	RMSE	MASE	Bias	RMSE vs seasonal naive
Seasonal naive (benchmark)	384,466	502,026	1.694	+293,631	—
SARIMAX (1,0,1)(0,1,1,52) + temp	384,167	509,358	1.693	+322,804	−1.5%
Gradient Boosting	380,829	516,065	1.678	+310,823	−2.8%
Random Forest	401,201	531,812	1.768	+311,760	−5.9%
Mean	636,694	739,546	2.805	+94,280	−47.3%
Drift	646,930	750,371	2.850	+162,353	−49.5%
Naive	648,169	751,929	2.856	+165,402	−49.8%
SARIMA (5,1,6)(1,1,1,52)	767,358	912,000	3.381	+760,673	−81.7%
(LSTM day-ahead, hourly: RMSE 6,321 vs 4,290 MW for an hourly seasonal-naive baseline — a
different, easier task, so excluded from this table.)
Takeaways
AIC ≠ forecast skill. The AIC-optimal SARIMA is the worst forecaster: its high order plus
non-seasonal differencing extrapolates a spurious upward trend over 104 weeks.
Seasonal naive is very hard to beat. Feature-importance analysis shows the 52- and 104-week
load lags carry ~85–90% of the signal — the ML models essentially rediscover the seasonal-naive
rule.
The 2020 structural break dominates. Every model carries a large positive bias because the
test window includes the COVID-19 demand collapse: seasonal-naive mean error rises from +106k
(pre-2020) to +606k (2020). This is an out-of-distribution failure, not a tuning failure.
Temperature adds little at weekly scale, because the annual seasonal term already encodes
the weather-driven cycle — and it isn't known at the forecast origin anyway.
Repository structure
```
Time-Series-Modelling-Case-Study/
├── 24090973\_Time\_Series.ipynb          # main analysis notebook (Parts 1–7)
├── 24090973\_Time\_Series\_Report.docx    # written report with figures \& discussion (Part 8)
├── README.md
└── outputs/                            # created at run time
    ├── figures/                        # EDA, forecast and diagnostic plots
    ├── forecasts/                      # per-model forecast series (CSV)
    └── metrics/                        # consolidated evaluation table
```
> Rename entries above to match your actual committed filenames.
Reproducing the analysis
Requirements: Python 3.10+, plus:
```bash
pip install pandas numpy matplotlib statsmodels scikit-learn requests tensorflow
```
Run: open the notebook and run all cells top to bottom.
```bash
jupyter notebook 24090973\_Time\_Series.ipynb
```
The notebook downloads both datasets automatically, builds the weekly series, fits every model,
and writes plots, forecasts and metrics to `outputs/`. Two cells are computationally heavy — the
588-model SARIMA grid search and the LSTM training — and are best run on a machine with a GPU
(e.g. Google Colab) for the deep-learning part.
Limitations & future work
Evaluation uses a single hold-out; rolling-origin (time-series) cross-validation would give
a more robust estimate and reduce the influence of the single 2020 window.
The COVID break should be modelled explicitly (intervention/dummy variables or a
regime-switching formulation), or the training window extended to include post-shock data.
A production SARIMAX would replace observed temperature with a weather forecast and add
explicit holiday dummies (which, unlike temperature, are known in advance).
Worth benchmarking genuinely multi-step / probabilistic models (Prophet, DeepAR, Temporal
Fusion Transformer) against seasonal naive, ideally with a longer history.
References
Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.
Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis: Forecasting and Control. Holden-Day.
Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of forecast accuracy. IJF, 22(4).
Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8).
Kong, W. et al. (2019). Short-term residential load forecasting based on LSTM. IEEE Trans. Smart Grid, 10(1).
Author
Sumanth Raj · Student number 24090973 · MSc Data Engineering, Northumbria University
Academic case study — data used under the terms of the respective providers.
