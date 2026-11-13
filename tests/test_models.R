# Run from the repository root: Rscript tests/test_models.R
source("R/models.R")

assert_close <- function(actual, expected, tolerance = 1e-8) {
  stopifnot(max(abs(actual - expected)) < tolerance)
}

# Check every NN parameter against a central finite-difference derivative,
# including unequal exposures and the L2 penalty.
set.seed(41)
small_x <- matrix(rnorm(18), nrow = 6L)
small_y <- c(0, 1, 0, 2, 0, 1)
small_e <- c(0.1, 0.5, 0.8, 1, 0.25, 0.7)
parameters <- list(W1 = matrix(rnorm(6, sd = 0.2), 3L, 2L),
                   b1 = c(0.1, -0.1), W2 = matrix(c(0.2, -0.3), 2L, 1L), b2 = -0.4)
analytic <- model_nn_loss_gradient(parameters, small_x, small_y, small_e, 0.03)
for (name in names(parameters)) {
  for (index in seq_along(parameters[[name]])) {
    plus <- minus <- parameters
    plus[[name]][index] <- plus[[name]][index] + 1e-6
    minus[[name]][index] <- minus[[name]][index] - 1e-6
    numerical <- (model_nn_loss_gradient(plus, small_x, small_y, small_e, 0.03)$loss -
                    model_nn_loss_gradient(minus, small_x, small_y, small_e, 0.03)$loss) / 2e-6
    assert_close(numerical, analytic$gradient[[name]][index], 1e-6)
  }
}

# An independent validation set has a strong nonlinear frequency signal.
# Exposure varies and is correlated with a risk feature, making omission of
# the offset harmful. Models must beat the exposure-adjusted constant rate.
set.seed(456)
n <- 2200L
x <- cbind(risk = runif(n, -2, 2), age = rnorm(n), noise = rnorm(n))
e <- 0.1 + 0.9 * plogis(x[, "risk"] + rnorm(n))
true_rate <- exp(-0.7 + 1.2 * tanh(x[, "risk"]) + 0.3 * x[, "age"])
y <- rpois(n, e * true_rate)
train <- seq_len(1600L)
valid <- 1601L:n
config <- list(boosting_rounds = 100L, boosting_depth = 2L, learning_rate = 0.08,
               hidden_units = 6L, neural_epochs = 100L, batch_size = 128L,
               neural_learning_rate = 0.015, patience = 15L, l2 = 0.001)
null_rate <- sum(y[train]) / sum(e[train])
null_deviance <- model_count_deviance(y[valid], e[valid] * null_rate)

for (fit in list(fit_boosting, fit_neural)) {
  model <- fit(x[train, ], y[train], e[train], x[valid, ], y[valid], e[valid],
               config, seed = 12L)
  predicted <- predict_rate(model, x[valid, ])
  stopifnot(length(predicted) == length(valid), all(is.finite(predicted)),
            all(predicted > 0), sd(predicted) > 0.1,
            cor(predicted, true_rate[valid]) > 0.7,
            model_count_deviance(y[valid], e[valid] * predicted) < 0.95 * null_deviance)
  # Validation history must describe the selected model's count predictions.
  assert_close(model_count_deviance(y[valid], e[valid] * predicted),
               model$best_validation_loss, 2e-6)
  repeated <- fit(x[train, ], y[train], e[train], x[valid, ], y[valid], e[valid],
                  config, seed = 12L)
  assert_close(predicted, predict_rate(repeated, x[valid, ]), 1e-10)
  # The analysis saves these complete objects for later explanation work.
  # In particular, the xgboost wrapper must survive RDS serialization.
  saved_path <- tempfile(fileext = ".rds")
  saveRDS(model, saved_path)
  restored <- readRDS(saved_path)
  assert_close(predicted, predict_rate(restored, x[valid, ]), 1e-10)
  unlink(saved_path)

  # Exposure is not accepted by predict_rate: for identical features annual
  # rate is identical; expected counts then scale linearly with exposure.
  same_features <- x[rep(valid[1L], 2L), , drop = FALSE]
  same_rate <- predict_rate(model, same_features)
  assert_close(same_rate[1L], same_rate[2L])
  expected_counts <- same_rate * c(0.25, 1)
  assert_close(expected_counts[2L] / expected_counts[1L], 4)

  bad_exposure <- e[train]
  bad_exposure[1L] <- 0
  failed <- try(fit(x[train, ], y[train], bad_exposure, x[valid, ], y[valid],
                    e[valid], config, seed = 12L), silent = TRUE)
  stopifnot(inherits(failed, "try-error"))
  cat(sprintf("%s: deviance %.4f vs constant %.4f; reproducibility passed\n",
              model$class, model$best_validation_loss, null_deviance))
}

# A constant-feature problem isolates exposure handling. Its fitted annual
# rate should recover total claims / total exposure, rather than mean counts.
constant_x <- matrix(0, 40L, 1L, dimnames = list(NULL, "constant"))
constant_e <- rep(c(0.1, 1), 20L)
constant_y <- rep(c(0, 1), 20L)
target_rate <- sum(constant_y) / sum(constant_e)
for (fit in list(fit_boosting, fit_neural)) {
  model <- fit(constant_x, constant_y, constant_e,
               constant_x, constant_y, constant_e, config, seed = 51L)
  assert_close(mean(predict_rate(model, constant_x)), target_rate, 0.03)
}

cat("All model tests passed, including finite-difference gradients and exposure calibration.\n")
