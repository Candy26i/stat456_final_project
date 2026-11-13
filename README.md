# STAT 456 Final Project

**Evaluating GLM-Based Local Surrogates for Explainable Machine Learning in Insurance Claim Prediction**

This R project compares Poisson and fixed-dispersion negative-binomial local surrogates with a LIME-style weighted linear baseline. The blackboxes are Poisson gradient boosting and a neural network with an exponential output. Both predict insurance claim frequency while accounting for policy exposure.

## 快速开始

需要 **R 4.3+** 和 **xgboost 3.2.1.1+**。本项目已在 R 4.5.2 / xgboost 3.2.1.1 上运行验证；其他模块使用 base R，不需要 Python、TensorFlow 或整个 CASdatasets 包。

```sh
git clone https://github.com/Candy26i/stat456_final_project.git
cd stat456_final_project
Rscript scripts/install_dependencies.R
Rscript scripts/run_tests.R
Rscript scripts/run_analysis.R --mode quick
```

第一次分析会下载两份公开的 CASdatasets 数据（约 23 MB），缓存到 `data/raw/`。数据固定到同一 Git revision 并检查 MD5；没有用模拟数据替代正式分析。原始数据和训练模型不会提交到 GitHub。

运行结束后，用浏览器打开 `results/quick/report.html`，查看预测性能、局部拟合效果和解释稳定性。输出目录必须为空；重复运行时使用新的 `--output`，避免覆盖已有结果。

```sh
# 小规模流程检查：每份数据 1,800 条、1 个解释对象。
Rscript scripts/run_analysis.R --mode smoke --output results/smoke-new

# 全量实验：全部有效记录、20 个解释对象、10 次重复。
# 神经网络和局部解释均使用 CPU；全量运行可能需要较长时间。
Rscript scripts/run_analysis.R --mode full --output results/full

# 中等规模自定义实验。
Rscript scripts/run_analysis.R --mode quick --max-rows 30000 --anchors 10 --repeats 5 --output results/medium

# 单数据集，以及预先指定的 NB 离散参数敏感性分析。
Rscript scripts/run_analysis.R --mode quick --datasets freMTPL2freq --nb-alpha 0.2 --output results/nb-alpha-02
Rscript scripts/run_analysis.R --help
```

## What is implemented

| Component | Implementation |
| --- | --- |
| Data | `catelematic13` (100,000 × 52) and `freMTPL2freq` (677,991 × 12), from a pinned CASdatasets revision |
| Exposure | Canadian `Duration / 365`; French `Exposure` in years |
| Split | Reproducible 70% training / 15% validation / 15% test; preprocessing learned only on training |
| Boosting | XGBoost `count:poisson` on count/exposure, weighted by exposure; validation early stopping |
| Neural network | One tanh hidden layer, exponential rate output, exposure-aware Poisson loss, analytic gradients and Adam |
| Local neighborhoods | Nearby complete training-policy records, preserving observed feature combinations; disjoint local fitting/evaluation rows |
| Surrogates | Weighted linear ridge, Poisson log link, fixed-alpha NB2 log link; fractional blackbox means are used directly |
| Fidelity | Held-out RMSE, locality-weighted RMSE/R², anchor error, negative linear prediction fraction and effective sample size |
| Stability | Coefficient/effect SD, sign agreement, feature-rank Spearman correlation and top-5 Jaccard across repeated neighborhoods |
| Reproducibility | Fixed seeds, data hashes, split IDs, feature dictionaries, configuration, training history, model objects and session versions |

The linear comparator implements LIME's weighted-ridge idea with the **same empirical neighborhoods, predictors and weights** as the GLMs. It is not a reproduction of the complete canonical LIME perturbation, discretization or feature-selection pipeline. This controlled comparison isolates the surrogate loss/link choice; it should be described as **LIME-style linear**, not a benchmark against the LIME package.

## Results and files

An executed quick run is included under [`results/example/`](results/example/). It uses 10,000 records per dataset, 3 test anchors, 3 repetitions, and neighborhood fitting sample sizes 200 and 500. These are demonstration results, not the full-data final study. The HTML report and CSVs report the actual results without assuming that GLMs outperform linear surrogates.

| Output | Contents |
| --- | --- |
| `report.html`, `fidelity.png` | Browsable results and comparison figure |
| `audit.csv`, `prediction.csv` | Actual sample sizes and test-set predictive metrics, including a constant-rate baseline |
| `summary.csv`, `fidelity.csv` | Per-method means and every individual local fit's fidelity metrics |
| `coefficients.csv`, `effects.csv` | Native coefficients, comparable count-scale effects and local support flags |
| `ranking_stability.csv`, `coefficient_stability.csv` | Ranking consistency and coefficient/effect variation |
| `*_feature_dictionary.csv`, `*_splits.csv` | Encoded terms, categorical references, original feature names and split IDs |
| `*_training.csv`, `*_test_predictions.csv` | Training curves and held-out predictions |
| `models/` | Local RDS models, preprocessing/splits and neighborhood donor IDs; portable XGBoost UBJ files |
| `config.R`, `data_manifest.csv`, `session_info.txt` | Reproduction settings and runtime information |

Source organization:

```text
R/config.R       Experiment sizes and hyperparameters
R/data.R         Download, schema handling, preprocessing and neighborhoods
R/models.R       Exposure-aware boosting and neural network
R/surrogates.R   Weighted linear, Poisson and NB2 optimization
R/evaluation.R   Fidelity, effects and repeated-neighborhood stability
R/reporting.R    CSV, plot and HTML report generation
scripts/         Dependency installation, analysis and test entry points
tests/           Offline mathematical and integration checks
docs/            Statistical definitions and interpretation notes
```

## Interpretation notes

- Surrogates are fitted to **blackbox expected counts**, never to the observed claims in their neighborhoods. Anchor exposure stays fixed across the local fit and evaluation.
- Poisson/NB coefficients are on a log-mean scale. Compare count-scale effects or feature rankings with the linear baseline, rather than comparing native slopes directly.
- NB `alpha` is a fixed working-loss parameter, not a dispersion estimate inferred from deterministic blackbox predictions. No p-values, likelihood-based AIC or causal claims are produced.
- Whole-record donor sampling avoids invalid independent perturbations, but its conclusions concern the local empirical donor distribution. Unsupported categorical contrasts are marked missing.
- The Canadian implementation uses a documented compact set of 20 predictors, including telematics summaries. Observed-period telematics make this a retrospective frequency analysis, not a prospective underwriting forecast.
- Quick/smoke runs establish that the pipeline works. Use full runs, more anchors, alternative bandwidths/alpha values, and repeated training seeds before drawing broad comparative conclusions.

See [`docs/methodology.md`](docs/methodology.md) for equations, feature choices and limitations. Tests check gradients, exposure calibration, reproducibility, recovery of known surrogates, categorical contrasts and separation of local evaluation rows.

## Sources

- [CASdatasets repository](https://github.com/dutangc/CASdatasets), pinned to `227fb56b8734bdb7c0327a41180e01d2ddaeaf26`; [Canadian data documentation](https://dutangc.github.io/CASdatasets/reference/catelematic13.html) and [French data documentation](https://github.com/dutangc/CASdatasets/blob/master/man-md/freMTPL2freq.md).
- So, B., Boucher, J.-P., & Valdez, E. A. (2021). [Synthetic Dataset Generation of Driver Telematics](https://doi.org/10.3390/risks9040058). *Risks*, 9(4), 58.
- [XGBoost Poisson objective documentation](https://xgboost.readthedocs.io/en/stable/parameter.html).
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). [“Why Should I Trust You?”: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938); [LIME weighted-regression implementation](https://github.com/marcotcr/lime/blob/master/lime/lime_base.py).

The proposal motivates this implementation; it does not supply experimental findings. Public datasets retain their upstream terms and citations. No source PDF, unrelated coursework or credentials are included in this repository.
