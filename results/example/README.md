# Executed quick-run example

These files come from an actual run on the pinned public Canadian and French datasets:

```sh
Rscript scripts/run_analysis.R --mode quick --output results/quick
```

The run used 10,000 randomly sampled records per dataset, three test anchors per dataset, three repeated fitting samples, sizes 200 and 500, two blackboxes and three surrogates: **216 converged local fits**. The held-out local evaluation set contains 250 distinct donor rows per anchor. Seeds and all hyperparameters are in `config.R`; runtime versions are in `session_info.txt`.

This directory is a compact selection of the generated outputs. Full per-row predictions, native coefficients/effects, split IDs, feature dictionaries and model objects are generated locally by the command above. Raw public data are downloaded separately and are not committed.

Open `report.html` locally, or inspect the CSVs and `fidelity.png` on GitHub. The quick run is a demonstration: three anchors and one training seed do not support a general claim that one method is superior. In particular, the French neural network has slightly worse test Poisson deviance than the constant-rate baseline in this sample; surrogate fidelity measures how well it is approximated, not whether its predictions are accurate.
