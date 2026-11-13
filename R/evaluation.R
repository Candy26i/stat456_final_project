count_deviance <- function(y, mu) {
  stopifnot(all(is.finite(mu)), all(mu > 0), all(y >= 0))
  term <- numeric(length(y)); positive <- y > 0
  term[positive] <- y[positive] * log(y[positive] / mu[positive])
  mean(2 * (term - y + mu))
}

predictive_metrics <- function(y, exposure, rate) {
  mu <- exposure * rate
  data.frame(count_rmse = sqrt(mean((y - mu)^2)),
    count_mae = mean(abs(y - mu)), poisson_deviance = count_deviance(y, mu),
    observed_claims = sum(y), predicted_claims = sum(mu),
    observed_annual_frequency = sum(y) / sum(exposure),
    predicted_annual_frequency = sum(mu) / sum(exposure))
}

# Effects have common expected-claim units. Numeric effects are derivatives
# per training SD. Categorical effects are exact level-vs-reference contrasts,
# with all other coordinates held at the anchor. They are descriptive, not
# causal interventions; unsupported contrasts are explicitly marked missing.
explanation_effects <- function(fit, anchor, z_fit, preprocessor) {
  zero <- matrix(0, 1, length(anchor), dimnames = list(NULL, names(anchor)))
  effect <- surrogate_effects(fit, zero[1, ])
  mapping <- preprocessor$mapping
  result <- mapping
  result$coefficient <- unname(fit$coefficients[-1])
  result$effect <- unname(effect)
  result$supported <- vapply(seq_len(ncol(z_fit)), function(j) sd(z_fit[, j]) > 1e-10, logical(1))
  for (name in names(preprocessor$specs)) {
    spec <- preprocessor$specs[[name]]
    if (spec$type != "categorical" || !length(spec$columns)) next
    cols <- spec$columns; reference <- zero
    reference[1, cols] <- -anchor[cols]
    ref_prediction <- predict_surrogate(fit, reference)
    original <- sweep(z_fit[, cols, drop = FALSE], 2, anchor[cols], "+")
    reference_present <- any(rowSums(original) < 0.5)
    for (column in cols) {
      level <- reference; level[1, column] <- level[1, column] + 1
      i <- match(column, result$column)
      result$effect[i] <- predict_surrogate(fit, level) - ref_prediction
      result$supported[i] <- reference_present && any(original[, column] > 0.5)
    }
  }
  result$effect[!result$supported] <- NA_real_
  result
}

group_effects <- function(effects) {
  do.call(rbind, lapply(split(effects, effects$feature), function(d) {
    available <- d$effect[d$supported]
    data.frame(feature = d$feature[1], type = d$type[1],
      importance = if (length(available)) sqrt(mean(available^2)) else NA_real_,
      signed_effect = if (d$type[1] == "numeric" && length(available)) available[1] else NA_real_,
      supported = length(available) > 0, supported_fraction = mean(d$supported))
  }))
}

local_experiments <- function(model, model_name, data, splits, z, preprocessor, config, dataset_seed) {
  set.seed(dataset_seed + 50L)
  anchors <- sample(splits$test, min(config$anchors, length(splits$test)))
  train <- z[splits$train, , drop = FALSE]
  # Cache each blackbox prediction once. Only features enter this calculation.
  rates <- predict_rate(model, z)
  fidelity <- list(); coefficients <- list(); grouped <- list(); neighborhoods <- list()
  serial <- 0L
  for (a in seq_along(anchors)) {
    index <- anchors[a]; anchor <- z[index, ]; exposure <- data$exposure[index]
    nb <- make_neighborhood(train, anchor, preprocessor, config, dataset_seed + 100L * a)
    z_eval <- sweep(train[nb$evaluation, , drop = FALSE], 2, anchor, "-")
    target_eval <- exposure * rates[splits$train[nb$evaluation]]
    target_anchor <- exposure * rates[index]
    neighborhoods[[a]] <- list(anchor_id = data$id[index], fit_pool_ids = data$id[splits$train[nb$fit_pool]],
      evaluation_ids = data$id[splits$train[nb$evaluation]], bandwidth = nb$bandwidth)
    for (replicate in seq_len(config$repeats)) {
      # Nested samples for n comparisons; same RNG draws for every blackbox/method.
      set.seed(dataset_seed + 10000L * a + replicate)
      sample_order <- sample(nb$fit_pool, max(config$neighborhood_sizes))
      for (n in config$neighborhood_sizes) {
        draw <- head(sample_order, n)
        z_fit <- sweep(train[draw, , drop = FALSE], 2, anchor, "-")
        target_fit <- exposure * rates[splits$train[draw]]
        for (method in c("lime_linear", "poisson", "nb2")) {
          fit <- fit_surrogate(z_fit, target_fit, nb$weights[draw], method,
                               exposure = exposure, alpha = config$nb_alpha, ridge = config$ridge)
          if (fit$convergence != 0L) stop("Local optimization failed: ", model_name, "/", method,
                                          "/anchor ", a, ": ", fit$message)
          prediction <- predict_surrogate(fit, z_eval)
          anchor_prediction <- predict_surrogate(fit, matrix(0, 1, ncol(z), dimnames = list(NULL, colnames(z))))
          key <- data.frame(dataset = data$audit$dataset, model = model_name,
            anchor_id = data$id[index], neighborhood_size = n, replicate = replicate, method = method)
          serial <- serial + 1L
          fidelity[[serial]] <- cbind(key, data.frame(
            rmse = sqrt(mean((prediction - target_eval)^2)),
            weighted_rmse = weighted_rmse(target_eval, prediction, nb$weights[nb$evaluation]),
            weighted_r2 = weighted_r2(target_eval, prediction, nb$weights[nb$evaluation]),
            anchor_absolute_error = abs(anchor_prediction - target_anchor),
            anchor_blackbox_count = target_anchor, anchor_surrogate_count = anchor_prediction,
            exposure_years = exposure, negative_prediction_fraction = mean(prediction < 0),
            fit_effective_n = effective_sample_size(nb$weights[draw]),
            evaluation_effective_n = effective_sample_size(nb$weights[nb$evaluation]),
            bandwidth = nb$bandwidth, convergence = fit$convergence,
            intercept = fit$coefficients[1], iterations = fit$iterations))
          effects <- explanation_effects(fit, anchor, z_fit, preprocessor)
          coefficients[[serial]] <- cbind(key[rep(1L, nrow(effects)), ], effects)
          groups <- group_effects(effects)
          grouped[[serial]] <- cbind(key[rep(1L, nrow(groups)), ], groups)
        }
      }
    }
  }
  list(fidelity = do.call(rbind, fidelity), coefficients = do.call(rbind, coefficients),
       effects = do.call(rbind, grouped), neighborhoods = neighborhoods)
}

stability_metrics <- function(effects, coefficients, top_k = 5L) {
  keys <- c("dataset", "model", "anchor_id", "neighborhood_size", "method")
  grouped <- split(effects, interaction(effects[keys], drop = TRUE))
  scores <- lapply(grouped, function(d) {
    reps <- sort(unique(d$replicate))
    # Compare only features supported in every repeated fit for this anchor.
    common <- Reduce(intersect, lapply(reps, function(r) d$feature[d$replicate == r & d$supported]))
    # Require exactly the same supported categorical contrasts across repeats;
    # otherwise a group RMS could change its definition between repetitions.
    cdata <- coefficients
    for (k in keys) cdata <- cdata[as.character(cdata[[k]]) == as.character(d[[k]][1]), , drop = FALSE]
    consistent <- vapply(common, function(f) {
      sets <- lapply(reps, function(r) sort(cdata$column[cdata$feature == f & cdata$replicate == r & cdata$supported]))
      all(vapply(sets, identical, logical(1), sets[[1]]))
    }, logical(1))
    common <- common[consistent]
    correlations <- jaccards <- numeric()
    if (length(common) >= 2L && length(reps) >= 2L) {
      pairs <- combn(reps, 2L)
      for (j in seq_len(ncol(pairs))) {
        left <- d[d$replicate == pairs[1, j], ]; right <- d[d$replicate == pairs[2, j], ]
        u <- left$importance[match(common, left$feature)]; v <- right$importance[match(common, right$feature)]
        correlations <- c(correlations, if (sd(u) > 0 && sd(v) > 0) cor(u, v, method = "spearman") else NA_real_)
        k <- min(top_k, length(common)); t1 <- common[order(-u, common)[seq_len(k)]]
        t2 <- common[order(-v, common)[seq_len(k)]]
        jaccards <- c(jaccards, length(intersect(t1, t2)) / length(union(t1, t2)))
      }
    }
    safe_mean <- function(v) if (any(is.finite(v))) mean(v[is.finite(v)]) else NA_real_
    cbind(d[1, keys], data.frame(repeats = length(reps), common_supported_features = length(common),
      mean_spearman = safe_mean(correlations), mean_top_k_jaccard = safe_mean(jaccards)))
  })
  coefficient_groups <- split(coefficients, interaction(coefficients[c(keys, "column")], drop = TRUE))
  spreads <- lapply(coefficient_groups, function(d) {
    supported <- d$effect[d$supported]
    cbind(d[1, c(keys, "column", "feature", "type", "level", "reference")], data.frame(
      coefficient_mean = mean(d$coefficient), coefficient_sd = sd(d$coefficient),
      effect_mean = if (length(supported)) mean(supported) else NA_real_,
      effect_sd = if (length(supported) >= 2L) sd(supported) else NA_real_,
      supported_repeats = length(supported),
      sign_agreement = if (length(supported)) max(table(sign(supported))) / length(supported) else NA_real_))
  })
  list(rankings = do.call(rbind, scores), coefficients = do.call(rbind, spreads))
}

summarize_fidelity <- function(d) {
  keys <- c("dataset", "model", "neighborhood_size", "method")
  measures <- c("rmse", "weighted_rmse", "anchor_absolute_error", "negative_prediction_fraction")
  per_anchor <- aggregate(d[measures], d[c(keys, "anchor_id")], mean)
  means <- aggregate(per_anchor[measures], per_anchor[keys], mean)
  counts <- aggregate(per_anchor$anchor_id, per_anchor[keys], length); names(counts)[ncol(counts)] <- "n_anchors"
  merge(means, counts, by = keys, sort = FALSE)
}
