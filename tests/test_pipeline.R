source("R/config.R"); source("R/data.R"); source("R/surrogates.R"); source("R/evaluation.R")
set.seed(111)
x <- data.frame(age = rnorm(500), category = rep(c("A", "B", "C", "D"), 125), constant = 1)
p <- fit_preprocessor(x, max_levels = 2, min_count = 20)
z <- transform_features(p, x)
other <- transform_features(p, data.frame(age = NA_real_, category = "UNSEEN", constant = 1))
stopifnot(all(is.finite(other)), other[1, "category=__OTHER__"] == 1,
          abs(mean(z[, "age"])) < 1e-12, abs(sd(z[, "age"]) - 1) < 1e-12)
# An extreme held-out point cannot alter the training transformation.
before <- serialize(p, NULL); invisible(transform_features(p, data.frame(age = 1e8, category = "A", constant = 1)))
stopifnot(identical(before, serialize(p, NULL)))
s <- split_policies(500, 456)
stopifnot(length(unique(unlist(s))) == 500, !length(intersect(s$train, s$test)))
cfg <- project_config("smoke"); cfg$donor_pool_size <- 300
nb <- make_neighborhood(z[s$train, ], z[s$test[1], ], p, cfg, 456)
stopifnot(!length(intersect(nb$fit_pool, nb$evaluation)), all(nb$weights > 0))
nb2 <- make_neighborhood(z[s$train, ], z[s$test[1], ], p, cfg, 456)
stopifnot(identical(nb, nb2))
# Known categorical contrast is an exact finite change, not the log slope.
anchor <- z[2, ]; centered <- sweep(z, 2, anchor, "-")
target <- 0.4 * exp(0.2 * centered[, "age"] + 0.6 * centered[, "category=B"])
fit <- fit_surrogate(centered, target, rep(1, nrow(z)), "poisson", exposure = 0.5, ridge = 0)
effects <- explanation_effects(fit, anchor, centered, p)
cat_row <- effects[effects$column == "category=B", ]
expected <- 0.4 * (1 - exp(-0.6))
stopifnot(abs(cat_row$effect - expected) < 1e-7, !effects$supported[effects$column == "constant"])
stopifnot(abs(count_deviance(c(0, 1), c(0.2, 1)) - 0.2) < 1e-12)
# End-to-end bookkeeping: two repeats, two n values, three methods.
mock <- list(x = x, y = rpois(500, 0.1), exposure = rep(0.7, 500),
             id = as.character(seq_len(500)), audit = data.frame(dataset = "synthetic"))
saved_predict <- if (exists("predict_rate")) predict_rate else NULL
predict_rate <- function(model, x) exp(-2 + 0.3 * x[, "age"])
local <- local_experiments(NULL, "known_loglinear", mock, s, z, p, cfg, 444)
if (!is.null(saved_predict)) predict_rate <- saved_predict
stopifnot(nrow(local$fidelity) == 12L, all(local$fidelity$convergence == 0),
          all(local$fidelity$negative_prediction_fraction >= 0))
st <- stability_metrics(local$effects, local$coefficients)
stopifnot(nrow(st$rankings) == 6L, all(st$rankings$mean_top_k_jaccard <= 1, na.rm = TRUE))
source("R/reporting.R")
report_dir <- tempfile("stat456-report-"); dir.create(report_dir)
write_report(report_dir, cfg, mock$audit, predictive_metrics(mock$y, mock$exposure, rep(0.1, 500)),
             summarize_fidelity(local$fidelity), st$rankings)
stopifnot(file.exists(file.path(report_dir, "report.html")), file.exists(file.path(report_dir, "fidelity.png")))
unlink(report_dir, recursive = TRUE)
cat("Pipeline tests passed.\n")
