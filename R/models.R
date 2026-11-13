# Two nonlinear claim-frequency models. Both predict an annual claim rate;
# multiply that rate by a policy's Exposure to predict its claim count.
# Exposure is part of the Poisson likelihood, never an input feature.

model_config <- function(config, name, default) {
  value <- config[[name]]
  if (is.null(value)) default else value
}

model_matrix <- function(x) {
  x <- as.matrix(x)
  if (!is.numeric(x) || nrow(x) < 1L || ncol(x) < 1L || any(!is.finite(x))) {
    stop("Predictors must be a nonempty, finite numeric matrix.")
  }
  storage.mode(x) <- "double"
  x
}

model_check_response <- function(x, y, exposure) {
  if (!is.numeric(y) || !is.numeric(exposure) ||
      length(y) != nrow(x) || length(exposure) != nrow(x) ||
      any(!is.finite(y)) || any(y < 0) ||
      any(!is.finite(exposure)) || any(exposure <= 0)) {
    stop("Each row needs a finite nonnegative count and a positive exposure.")
  }
}

model_check_columns <- function(x, reference_names, reference_ncol) {
  if (ncol(x) != reference_ncol ||
      (!is.null(reference_names) && !identical(colnames(x), reference_names))) {
    stop("Prediction columns must match the training columns in the same order.")
  }
}

# Mean Poisson deviance on COUNTS. For y = 0 the logarithmic term is zero.
model_count_deviance <- function(y, mu) {
  mu <- pmax(as.numeric(mu), 1e-12)
  terms <- mu - y
  positive <- y > 0
  terms[positive] <- terms[positive] + y[positive] * log(y[positive] / mu[positive])
  2 * mean(terms)
}

fit_boosting <- function(x, y, exposure, x_valid, y_valid, exposure_valid,
                         config = list(), seed = 456L) {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("Install xgboost before fitting the boosting model.")
  }
  x <- model_matrix(x)
  x_valid <- model_matrix(x_valid)
  model_check_response(x, y, exposure)
  model_check_response(x_valid, y_valid, exposure_valid)
  model_check_columns(x_valid, colnames(x), ncol(x))
  set.seed(seed)

  # With response y / exposure and weight exposure, the Poisson objective
  # equals the count likelihood with log(exposure) as an offset, up to a
  # constant independent of the fitted rate.
  train <- xgboost::xgb.DMatrix(x, label = y / exposure, weight = exposure,
                                nthread = 2L)
  valid <- xgboost::xgb.DMatrix(x_valid, label = y_valid / exposure_valid,
                                weight = exposure_valid, nthread = 2L)
  count_metric <- function(margin, data) {
    e <- xgboost::getinfo(data, "weight")
    count <- xgboost::getinfo(data, "label") * e
    # The R xgb.train custom-metric API supplies untransformed margins.
    mu <- e * exp(pmin(30, pmax(-30, as.numeric(margin))))
    list(metric = "count_deviance", value = model_count_deviance(count, mu))
  }
  booster <- xgboost::xgb.train(
    params = list(
      objective = "count:poisson", disable_default_eval_metric = TRUE,
      base_score = max(sum(y) / sum(exposure), 1e-6),
      max_depth = as.integer(model_config(config, "boosting_depth", 3L)),
      eta = model_config(config, "learning_rate", 0.05),
      min_child_weight = 5, subsample = 0.9, colsample_bytree = 0.9,
      nthread = 2L, seed = as.integer(seed)
    ),
    data = train,
    nrounds = as.integer(model_config(config, "boosting_rounds", 300L)),
    evals = list(train = train, validation = valid),
    custom_metric = count_metric, verbose = 0L,
    callbacks = list(xgboost::xgb.cb.early.stop(
      stopping_rounds = as.integer(model_config(config, "patience", 15L)),
      maximize = FALSE, verbose = FALSE, save_best = TRUE
    ))
  )
  # Recent xgboost versions store training history in an R attribute.
  history <- attr(booster, "evaluation_log")
  if (is.null(history)) history <- booster$evaluation_log
  training <- data.frame(
    epoch = history$iter,
    train_loss = history$train_count_deviance,
    validation_loss = history$validation_count_deviance
  )
  best_epoch <- which.min(training$validation_loss)
  list(class = "boosting", booster = booster, training = training,
       best_epoch = best_epoch, best_validation_loss = min(training$validation_loss),
       loss_metric = "mean_count_poisson_deviance", seed = seed,
       feature_names = colnames(x), n_features = ncol(x))
}

model_nn_forward <- function(parameters, x) {
  hidden <- tanh(sweep(x %*% parameters$W1, 2L, parameters$b1, "+"))
  raw_eta <- as.vector(hidden %*% parameters$W2 + parameters$b2)
  # Clipping only guards against numerical overflow. Its derivative below
  # makes the gradient consistent with this exact forward calculation.
  eta <- pmin(20, pmax(-20, raw_eta))
  list(hidden = hidden, eta = eta, raw_eta = raw_eta, rate = exp(eta))
}

# Mean negative log likelihood (dropping data-only constants) plus an L2
# penalty on weights. Biases are unpenalized. Kept separate for gradient tests.
model_nn_loss_gradient <- function(parameters, x, y, exposure, l2 = 0.001) {
  forward <- model_nn_forward(parameters, x)
  mu <- exposure * forward$rate
  residual <- (mu - y) * (forward$raw_eta > -20 & forward$raw_eta < 20)
  hidden_gradient <- (residual %o% as.vector(parameters$W2)) *
    (1 - forward$hidden^2)
  n <- nrow(x)
  gradient <- list(
    W1 = crossprod(x, hidden_gradient) / n + l2 * parameters$W1,
    b1 = colMeans(hidden_gradient),
    W2 = crossprod(forward$hidden, residual) / n + l2 * parameters$W2,
    b2 = mean(residual)
  )
  loss <- mean(mu - y * forward$eta) +
    l2 / 2 * (sum(parameters$W1^2) + sum(parameters$W2^2))
  list(loss = loss, gradient = gradient)
}

fit_neural <- function(x, y, exposure, x_valid, y_valid, exposure_valid,
                       config = list(), seed = 456L) {
  x <- model_matrix(x)
  x_valid <- model_matrix(x_valid)
  model_check_response(x, y, exposure)
  model_check_response(x_valid, y_valid, exposure_valid)
  model_check_columns(x_valid, colnames(x), ncol(x))
  feature_names <- colnames(x)
  p <- ncol(x)
  center <- colMeans(x)
  # A training-only transform prevents information from validation leaking
  # into fitting; constant columns are centered without dividing by zero.
  scale <- sqrt(colSums(sweep(x, 2L, center, "-")^2) / max(1, nrow(x) - 1L))
  scale[scale < 1e-8] <- 1
  x <- sweep(sweep(x, 2L, center, "-"), 2L, scale, "/")
  x_valid <- sweep(sweep(x_valid, 2L, center, "-"), 2L, scale, "/")

  hidden_units <- as.integer(model_config(config, "hidden_units", 12L))
  epochs <- as.integer(model_config(config, "neural_epochs", 200L))
  batch_size <- as.integer(model_config(config, "batch_size", 256L))
  learning_rate <- model_config(config, "neural_learning_rate", 0.01)
  patience <- as.integer(model_config(config, "patience", 15L))
  l2 <- model_config(config, "l2", 0.001)
  if (hidden_units < 1L || epochs < 1L || batch_size < 1L || patience < 1L ||
      learning_rate <= 0 || l2 < 0) stop("Invalid neural network configuration.")
  set.seed(seed)
  parameters <- list(
    W1 = matrix(rnorm(p * hidden_units, sd = sqrt(2 / (p + hidden_units))),
                nrow = p, ncol = hidden_units),
    b1 = rep(0, hidden_units),
    W2 = matrix(rnorm(hidden_units, sd = 0.05), ncol = 1L),
    b2 = log(max(sum(y) / sum(exposure), 1e-6))
  )
  first_moment <- lapply(parameters, function(value) value * 0)
  second_moment <- first_moment
  step <- 0L
  evaluate <- function(parameters, predictors, count, e) {
    model_count_deviance(count, e * model_nn_forward(parameters, predictors)$rate)
  }
  training <- data.frame(
    epoch = 0L, train_loss = evaluate(parameters, x, y, exposure),
    validation_loss = evaluate(parameters, x_valid, y_valid, exposure_valid)
  )
  best_parameters <- parameters
  best_loss <- training$validation_loss[1L]
  best_epoch <- 0L
  stale_epochs <- 0L

  for (epoch in seq_len(epochs)) {
    order <- sample.int(nrow(x))
    for (start in seq.int(1L, nrow(x), by = batch_size)) {
      rows <- order[start:min(start + batch_size - 1L, nrow(x))]
      gradient <- model_nn_loss_gradient(
        parameters, x[rows, , drop = FALSE], y[rows], exposure[rows], l2
      )$gradient
      # Adam combines a moving average of each gradient with its squared
      # magnitude. Bias correction accounts for the initially zero averages.
      step <- step + 1L
      for (name in names(parameters)) {
        first_moment[[name]] <- 0.9 * first_moment[[name]] + 0.1 * gradient[[name]]
        second_moment[[name]] <- 0.999 * second_moment[[name]] +
          0.001 * gradient[[name]]^2
        corrected_first <- first_moment[[name]] / (1 - 0.9^step)
        corrected_second <- second_moment[[name]] / (1 - 0.999^step)
        parameters[[name]] <- parameters[[name]] -
          learning_rate * corrected_first / (sqrt(corrected_second) + 1e-8)
      }
    }
    train_loss <- evaluate(parameters, x, y, exposure)
    valid_loss <- evaluate(parameters, x_valid, y_valid, exposure_valid)
    if (!is.finite(train_loss) || !is.finite(valid_loss)) {
      stop("Neural optimization became nonfinite; reduce neural_learning_rate.")
    }
    training <- rbind(training, data.frame(
      epoch = epoch, train_loss = train_loss, validation_loss = valid_loss
    ))
    if (valid_loss < best_loss - 1e-8) {
      best_parameters <- parameters
      best_loss <- valid_loss
      best_epoch <- epoch
      stale_epochs <- 0L
    } else {
      stale_epochs <- stale_epochs + 1L
    }
    if (stale_epochs >= patience) break
  }
  list(class = "neural", parameters = best_parameters, center = center,
       scale = scale, training = training, best_epoch = best_epoch,
       best_validation_loss = best_loss, loss_metric = "mean_count_poisson_deviance",
       seed = seed, feature_names = feature_names, n_features = p)
}

predict_rate <- function(model, x) {
  x <- model_matrix(x)
  model_check_columns(x, model$feature_names, model$n_features)
  if (identical(model$class, "boosting")) {
    rate <- as.numeric(stats::predict(model$booster, x))
  } else if (identical(model$class, "neural")) {
    x <- sweep(sweep(x, 2L, model$center, "-"), 2L, model$scale, "/")
    rate <- model_nn_forward(model$parameters, x)$rate
  } else {
    stop("Unknown model class: ", model$class)
  }
  if (any(!is.finite(rate)) || any(rate <= 0)) stop("Invalid predicted claim rate.")
  rate
}
