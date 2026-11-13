# Statistical design

## Prediction target and exposure

For a policy with predictors `x`, observed count `Y` and exposure `e > 0`, both blackboxes estimate an annual frequency `f(x) > 0`. The expected count is `mu(x,e) = e f(x)`.

XGBoost minimizes a Poisson loss for response `Y/e` and observation weight `e`. Up to terms that do not depend on the model, this is the same objective as the count loss `e f(x) - Y log f(x)` with a log-exposure offset. The neural network directly minimizes this count-scale loss with an exponential rate output, a tanh hidden layer and L2 penalties on weights. Its exponential input is clipped to [-20,20] solely as a numerical safeguard, with the corresponding derivative used in optimization. Validation mean count Poisson deviance selects the stopping epoch/iteration; test outcomes are never used to train or stop a model.

Canadian exposure is `Duration / 365`. The source article defines Duration in days. French Exposure is already in years; values above one year are retained, not silently capped. Counts are not capped. IDs, observed claim counts, Canadian `AMT_Claim`, and raw exposure are excluded from risk features.

## Data and encoding

The first run retrieves only the two `.rda` files from the pinned official repository revision. MD5 detects accidental cache changes; HTTPS and the pinned revision identify the source. The cache is ignored by Git. The actual downloaded data have 100,000 Canadian records and 677,991 French records; some upstream prose describes inconsistent dimensions, so the code checks actual fields.

Rows with nonfinite/negative/noninteger counts or nonpositive/nonfinite exposures are excluded and counted. Distinct policy IDs with identical discretized fields are retained. French IDs are unique in the pinned data; a repeated ID causes an explicit error rather than a potentially leaking row split. The Canadian synthetic records have no persistent entity ID, so row indices identify records. A random 70/15/15 split assumes independent policy records; it is not a temporal or unseen-driver generalization study.

The French model uses all ten risk predictors. Density is transformed by `log1p`.

The Canadian model uses these 20 predictors:

| Group | Predictors |
| --- | --- |
| Driver/vehicle | Insured.age, Insured.sex, Car.age, Marital, Car.use, Credit.score, Years.noclaims |
| Geography | Region, Territory (categorical) |
| Mileage/use | Annual.miles.drive, Annual.pct.driven, Annualized.miles = Total.miles.driven / exposure, Avgdays.week |
| Driving timing | Pct.drive.wkday, Pct.drive.rush.am, Pct.drive.rush.pm |
| Driving behavior | Accel.06miles, Brake.06miles, Left.turn.intensity08, Right.turn.intensity08 |

Annual.miles.drive, Annualized.miles, and the four driving-behavior variables use `log1p`. The seven daily shares, complementary weekend share, longer-trip fractions, and higher intensity thresholds are omitted to keep explanations compact and avoid exact compositional aliases or many nearly redundant measurements. This is a prespecified feature restriction, not feature selection based on test outcomes. Model conclusions apply to this chosen feature set.

The source paper explicitly allows `Car.age = -2` or `-1` for newer model-year vehicles. These values are retained. Numeric missingness is imputed with training medians; numeric predictors are centered and scaled with training means/SDs. Categorical levels are sorted by training frequency; at most the 20 most frequent levels with at least 20 training observations are kept, and remaining levels share OTHER. The most frequent level is the reference. Unseen levels map to a learned OTHER level, or the reference if no OTHER exists. Dictionaries record every encoded contrast.

Annualized mileage divides a contemporaneous accumulated quantity by coverage, but exposure is not a free predictor. Annual.pct.driven and Avgdays.week remain as provided by the source. These telematics describe the observation period; their use should not be presented as information known before that period.

## Local distribution and fair comparison

Anchors are a reproducible random sample of test policies. Every method and both blackboxes use the same anchors, neighbor identities, fitting sample draws, kernel weights and held-out evaluation rows.

Each original feature contributes one component to a mixed distance. A numeric component is `min(abs(standardized difference)/3, 1)^2`; a categorical component is 0 for a match and 1 otherwise. The final distance is the square root of the average across original features. Multiple dummy columns therefore do not multiply the weight of geography.

The closest `donor_pool_size` training rows form a candidate neighborhood. A fixed subset is reserved for local evaluation; it is excluded from **every** local fit for that anchor. The remaining rows are sampled without replacement, using nested samples for different fit sample sizes. Complete donor records retain their observed feature combinations. No local outcome values are read: targets come from the fitted blackbox.

The bandwidth is `kernel_width * median(candidate distance)`, with a small positive lower bound. Kernel weights are `exp(-0.5 * (distance / bandwidth)^2)`. The candidate pool, bandwidth and evaluation sample remain fixed when sample size changes. Thus the neighborhood-size experiment measures the effect of fitting sample count, not a simultaneous change in locality. Repetitions can share donor rows; they measure stability conditional on this pool, not independent population replications. Evaluation rows were used to train the blackbox, but were never used to fit its local surrogate; the local estimand is fidelity to that fixed blackbox.

## Surrogate losses

Fix an anchor's exposure `e0`. The local target is the unrounded expected count `t_j = e0 f(x_j)`. Let `z_j` be the encoded feature vector centered at the anchor, so the anchor is `z=0`. Define normalized weights `v_j = w_j / sum(w)` and ridge parameter `lambda`.

The linear surrogate is `g(z) = b0 + z b` and minimizes:

```text
0.5 * sum_j v_j * (g(z_j) - t_j)^2 + 0.5 * lambda * sum_k b_k^2
```

The Poisson and NB2 surrogates use `eta_j = log(e0) + b0 + z_j b` and `mu_j = exp(eta_j)`:

```text
Poisson: sum_j v_j * (mu_j - t_j * eta_j)
         + 0.5 * lambda * sum_k b_k^2

NB2:     sum_j v_j * ((t_j + 1/alpha) * log(1 + alpha*mu_j) - t_j*eta_j)
         + 0.5 * lambda * sum_k b_k^2
```

Intercepts are unpenalized. Alpha is fixed at 1 by default, with the NB2 working variance convention `mu + alpha*mu^2`. It can be varied with `--nb-alpha`. These are mean-matching losses on deterministic, possibly fractional targets. They do not assert that those targets are random integer outcomes, and they do not justify likelihood-based hypothesis tests. No rounding, Poisson simulation or NB dispersion fitting is performed.

All methods use the same small prespecified ridge value, normalized weights and feature matrix. The loss scales and link scales differ, so equal numeric ridge is not equal effective regularization; use `--ridge` sensitivity analyses for substantive comparisons. No test-set selection of the best ridge/alpha is performed. Newton optimization uses analytic gradients/Hessians and backtracking; any failed local optimization stops the run with a diagnostic. LIME-style predictions remain untruncated, including any negative expected counts.

## Fidelity, stability and interpretation

The independent local evaluation set gives ordinary RMSE and kernel-weighted RMSE, measured in expected claims at the anchor's exposure. Weighted R² is undefined for an imperfect fit to an exactly constant target and is recorded as missing. An exact constant-target fit is assigned 1 by explicit convention. Anchor absolute error separately measures the fit at the actual test policy. Effective sample size is `(sum w)^2 / sum(w^2)`. Different anchor exposures imply different count scales, so count RMSE should not be read as a rate-normalized comparison across policies.

For numeric standardized variables, a GLM's local count-scale derivative is `g(0) * b_k`; the linear derivative is `b_k`. A derivative for a log-transformed variable is per training SD of `log1p(variable)`, not per raw mile or event. Categorical effects are exact predictions for one category minus the fixed reference category while other coordinates stay at the anchor. They require both categories to occur in the fit neighborhood. Such one-variable contrasts describe the mathematical surrogate and need not correspond to feasible causal interventions.

The importance of one original feature is the root mean square of its supported count-scale effects (one derivative for a numeric feature; multiple level-reference contrasts for a categorical feature). Pairwise Spearman correlations and top-5 Jaccard compare these importances over repeated neighborhoods. A feature must be supported in every repetition, and a categorical group must have the same supported contrast set in each repetition, to enter the ranking comparison. Near-zero coefficients and ties can still make rank/sign metrics unstable. Ridge fits are dense: top-5 display/ranking is not sparse model fitting.

Coefficient stability CSVs give native coefficient mean/SD, count-scale effect mean/SD, number of supported repetitions, and modal-sign agreement. Unsupported contrasts have missing effects, not evidence of a true zero. Overall fidelity averages repetitions within each anchor, then averages anchors equally. Repeated fits are not treated as independent policy-level statistical evidence. The current report provides descriptive means, not confidence intervals or significance tests.

## Reproducibility and limits

`--mode quick` uses a simple random subsample of at most 10,000 policies per dataset before splitting. `--mode full` uses all valid records, 20 anchors, 10 neighborhood repetitions, and fit sizes 250/500/1000. Even the full configuration is one training seed. The runner records configuration, split IDs, donor identities, feature mappings, model objects, data revision/hashes, training curves and `sessionInfo()`.

Offline tests check analytic neural/surrogate derivatives against finite differences, recovery on fractional targets, the NB-to-Poisson limit, exposure calibration, reproducibility, positive GLM predictions, rank-deficient designs, categorical contrasts and disjoint local evaluation pools. A small real-data run validates the complete integration. Broad results require additional anchors, model seeds and prespecified sensitivity runs; a small demonstration cannot establish universal superiority. Explanations are associations within a trained model, not causal effects, and Canadian synthetic-data results are not independent evidence about a real insurer's portfolio.

Sources: [So et al. (2021), including Duration and model-year descriptions](https://doi.org/10.3390/risks9040058), [CASdatasets](https://github.com/dutangc/CASdatasets), [XGBoost objectives](https://xgboost.readthedocs.io/en/stable/parameter.html), and [LIME's weighted linear baseline](https://github.com/marcotcr/lime/blob/master/lime/lime_base.py).
