#!/usr/bin/env Rscript
# Base/recommended R handles everything except Poisson tree boosting.
if (getRversion() < "4.3.0") stop("Please install R >= 4.3 (tested with R 4.5.2).")
if (!requireNamespace("xgboost", quietly = TRUE) || packageVersion("xgboost") < "3.2.1.1")
  install.packages("xgboost", repos = "https://cloud.r-project.org")
if (!requireNamespace("xgboost", quietly = TRUE) || packageVersion("xgboost") < "3.2.1.1")
  stop("Installation did not provide xgboost >= 3.2.1.1.")
cat("Ready: R", as.character(getRversion()), "; xgboost", as.character(packageVersion("xgboost")), "\n")
