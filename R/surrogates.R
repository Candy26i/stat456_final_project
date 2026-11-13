# Local regression surrogates for fractional black-box expected counts.
# All columns of z must use the same standardized feature coordinates at fit,
# prediction, and explanation time. Exposure is a single fixed policy value e0.
# `lime_linear` is a LIME-style weighted ridge comparator, not canonical LIME.

.surrogate_matrix <- function(z, n = NULL) {
  if (is.null(dim(z))) {
    if (!is.null(n) && n > 1L && length(z) == n) {
      z <- matrix(z, ncol = 1L)
    } else {
      z <- matrix(z, nrow = 1L,
                  dimnames = list(NULL, names(z)))
    }
  }
  z <- as.matrix(z)
  if (!is.numeric(z) || length(dim(z)) != 2L || any(!is.finite(z))) {
    stop("z must be a finite numeric matrix.")
  }
  if (!is.null(n) && nrow(z) != n) stop("z and target have different lengths.")
  if (is.null(colnames(z))) colnames(z) <- paste0("x", seq_len(ncol(z)))
  if (anyDuplicated(colnames(z)) || any(colnames(z) == "(Intercept)")) {
    stop("Feature names must be unique and must not be '(Intercept)'.")
  }
  z
}

.surrogate_weights <- function(weights, n) {
  if (!is.numeric(weights) || length(weights) != n ||
      any(!is.finite(weights)) || any(weights < 0) || !any(weights > 0)) {
    stop("weights must be finite, nonnegative, and have positive total weight.")
  }
  # Rescale first to avoid overflow in sum(weights).
  weights <- weights / max(weights)
  weights / sum(weights)
}

.surrogate_inputs <- function(z, target, weights, method, exposure, alpha, ridge) {
  if (!is.numeric(target) || !length(target) || any(!is.finite(target))) {
    stop("target must be a nonempty finite numeric vector.")
  }
  if (any(target < 0)) stop("Expected-count targets must be nonnegative.")
  method <- match.arg(method, c("lime_linear", "poisson", "nb2"))
  scalar <- function(x) is.numeric(x) && length(x) == 1L && is.finite(x)
  if (!scalar(exposure) || exposure <= 0) stop("exposure must be one positive value.")
  if (!scalar(alpha) || alpha <= 0 || !is.finite(1 / alpha)) {
    stop("alpha must be positive with a finite reciprocal.")
  }
  if (!scalar(ridge) || ridge < 0) stop("ridge must be finite and nonnegative.")
  z <- .surrogate_matrix(z, length(target))
  list(x = cbind("(Intercept)" = 1, z), target = as.numeric(target),
       weights = .surrogate_weights(weights, length(target)),
       method = method, exposure = exposure, alpha = alpha, ridge = ridge)
}

.surrogate_softplus <- function(x) {
  pmax(x, 0) + log1p(exp(-abs(x)))
}

# Objective constants independent of coefficients are deliberately omitted.
# Ridge is 0.5 * ridge * sum(slopes^2); the intercept is never penalized.
.surrogate_evaluate <- function(beta, input, derivatives = TRUE) {
  x <- input$x
  # Zero-weight observations should not affect the objective even when their
  # hypothetical fitted counts would overflow.
  keep <- input$weights > 0
  x <- x[keep, , drop = FALSE]
  w <- input$weights[keep]
  y <- input$target[keep]
  eta <- drop(x %*% beta)
  penalty <- 0.5 * input$ridge * sum(beta[-1L]^2)
  if (input$method == "lime_linear") {
    mu <- eta
    loss <- 0.5 * (mu - y)^2
    score <- mu - y
    curvature <- rep(1, length(y))
  } else {
    eta <- eta + log(input$exposure)
    if (any(!is.finite(eta))) {
      return(list(value = Inf, gradient = rep(NA_real_, length(beta)), hessian = NULL))
    }
    if (input$method == "poisson") {
      # An overflowing trial step is rejected by line search; no fitted mean is
      # clipped, so objective and derivatives continue to describe one model.
      if (any(eta > log(.Machine$double.xmax))) {
        return(list(value = Inf, gradient = rep(NA_real_, length(beta)), hessian = NULL))
      }
      mu <- exp(eta)
      loss <- mu - y * eta
      score <- mu - y
      curvature <- mu
    } else {
      a <- input$alpha
      v <- log(a) + eta
      sp <- .surrogate_softplus(v)
      success <- plogis(v)
      failure <- plogis(-v)
      # Stable forms of the NB2 mean-matching loss and its eta derivatives.
      loss <- sp / a + y * (sp - eta)
      score <- success / a - y * failure
      curvature <- (success / a) * failure + y * success * failure
    }
  }
  value <- sum(w * loss) + penalty
  if (!derivatives) return(list(value = value))
  gradient <- drop(crossprod(x, w * score))
  gradient[-1L] <- gradient[-1L] + input$ridge * beta[-1L]
  hessian <- crossprod(x, x * (w * curvature))
  diag(hessian)[-1L] <- diag(hessian)[-1L] + input$ridge
  names(gradient) <- colnames(x)
  list(value = value, gradient = gradient, hessian = hessian)
}

# Public diagnostic helper: useful for checking analytic gradients and Hessians.
surrogate_objective <- function(coefficients, z, target, weights, method,
                                exposure = 1, alpha = 1, ridge = 1e-4,
                                derivatives = TRUE) {
  input <- .surrogate_inputs(z, target, weights, method, exposure, alpha, ridge)
  if (!is.numeric(coefficients) || length(coefficients) != ncol(input$x) ||
      any(!is.finite(coefficients))) stop("Invalid coefficient vector.")
  .surrogate_evaluate(coefficients, input, derivatives)
}

.surrogate_solve <- function(hessian, rhs) {
  factor <- tryCatch(chol(hessian), error = function(e) NULL)
  if (!is.null(factor)) {
    return(drop(backsolve(factor, forwardsolve(t(factor), rhs))))
  }
  # With ridge = 0 a neighborhood may have redundant or constant columns.
  # The pseudoinverse returns a minimum-norm step in its identifiable subspace.
  eig <- eigen(hessian, symmetric = TRUE)
  cutoff <- max(1, max(abs(eig$values))) *
    .Machine$double.eps * nrow(hessian) * 10
  keep <- eig$values > cutoff
  if (!any(keep)) return(rep(0, length(rhs)))
  basis <- eig$vectors[, keep, drop = FALSE]
  drop(basis %*% (drop(crossprod(basis, rhs)) / eig$values[keep]))
}

fit_surrogate <- function(z, target, weights, method, exposure = 1, alpha = 1,
                          ridge = 1e-4, maxit = 150L, tol = 1e-8) {
  input <- .surrogate_inputs(z, target, weights, method, exposure, alpha, ridge)
  if (length(maxit) != 1L || !is.finite(maxit) || maxit < 1 || maxit != floor(maxit)) {
    stop("maxit must be a positive integer.")
  }
  if (length(tol) != 1L || !is.finite(tol) || tol <= 0) stop("tol must be positive.")
  x <- input$x
  w <- input$weights
  y <- input$target
  beta <- setNames(numeric(ncol(x)), colnames(x))
  mean_target <- sum(w * y)
  iterations <- 0L
  convergence <- 0L
  message <- "Converged."
  boundary <- FALSE

  if (input$method == "lime_linear") {
    hessian <- crossprod(x, x * w)
    diag(hessian)[-1L] <- diag(hessian)[-1L] + ridge
    beta[] <- .surrogate_solve(hessian, drop(crossprod(x, w * y)))
    evaluation <- .surrogate_evaluate(beta, input)
    iterations <- 1L
  } else if (mean_target == 0) {
    # All positive-weight targets are zero. The optimum lies at intercept -Inf;
    # represent that boundary explicitly instead of reporting a spurious fit.
    beta[1L] <- -Inf
    boundary <- TRUE
    evaluation <- list(value = 0, gradient = setNames(rep(0, length(beta)), names(beta)))
    message <- "Exact zero-mean boundary solution (no finite log intercept)."
  } else {
    beta[1L] <- log(mean_target) - log(exposure)
    evaluation <- .surrogate_evaluate(beta, input)
    gradient_tolerance <- tol * max(mean_target, 1e-3)
    for (iteration in seq_len(maxit)) {
      if (max(abs(evaluation$gradient)) <= gradient_tolerance) break
      direction <- -.surrogate_solve(evaluation$hessian, evaluation$gradient)
      directional_derivative <- sum(evaluation$gradient * direction)
      if (!all(is.finite(direction)) || !is.finite(directional_derivative) ||
          directional_derivative >= 0) {
        convergence <- 2L
        message <- "Newton step was not a finite descent direction."
        break
      }
      step <- 1
      accepted <- FALSE
      for (backtrack in seq_len(60L)) {
        candidate <- beta + step * direction
        candidate_value <- .surrogate_evaluate(candidate, input, FALSE)$value
        if (is.finite(candidate_value) &&
            candidate_value <= evaluation$value + 1e-4 * step * directional_derivative) {
          accepted <- TRUE
          break
        }
        # Near the optimum, a true Newton improvement may be smaller than the
        # rounding error in the loss sum. Require a strictly smaller gradient
        # before accepting such a numerically indistinguishable objective.
        roundoff <- 10 * .Machine$double.eps * max(1, abs(evaluation$value))
        if (is.finite(candidate_value) && candidate_value <= evaluation$value + roundoff) {
          candidate_gradient <- .surrogate_evaluate(candidate, input)$gradient
          if (all(is.finite(candidate_gradient)) &&
              max(abs(candidate_gradient)) < max(abs(evaluation$gradient))) {
            accepted <- TRUE
            break
          }
        }
        step <- step / 2
      }
      if (!accepted) {
        convergence <- 2L
        message <- "Backtracking failed to find a decreasing finite objective."
        break
      }
      beta <- candidate
      evaluation <- .surrogate_evaluate(beta, input)
      iterations <- iteration
    }
    if (convergence == 0L && max(abs(evaluation$gradient)) > gradient_tolerance) {
      convergence <- 1L
      message <- "Maximum iterations reached before gradient convergence."
    }
  }
  if (!is.finite(evaluation$value) || any(!is.finite(evaluation$gradient))) {
    stop("Surrogate fitting produced a nonfinite objective or gradient.")
  }
  if (convergence != 0L) warning(message, call. = FALSE)
  structure(list(coefficients = beta, method = input$method, exposure = exposure,
                 alpha = if (input$method == "nb2") alpha else NA_real_,
                 ridge = ridge, feature_names = colnames(x)[-1L],
                 convergence = convergence, converged = convergence == 0L,
                 iterations = iterations, gradient_norm = max(abs(evaluation$gradient)),
                 objective = evaluation$value, message = message,
                 boundary_solution = boundary, effective_n = effective_sample_size(weights)),
            class = "count_surrogate")
}

.surrogate_prediction_matrix <- function(object, z) {
  supplied_names <- if (is.null(dim(z))) names(z) else colnames(z)
  z <- .surrogate_matrix(z)
  if (ncol(z) != length(object$feature_names)) stop("Prediction feature count differs from fit.")
  if (!is.null(supplied_names)) {
    if (!setequal(colnames(z), object$feature_names)) stop("Prediction feature names differ from fit.")
    z <- z[, object$feature_names, drop = FALSE]
  } else {
    colnames(z) <- object$feature_names
  }
  z
}

predict_surrogate <- function(object, z) {
  if (!inherits(object, "count_surrogate")) stop("object must be a fitted count surrogate.")
  z <- .surrogate_prediction_matrix(object, z)
  if (isTRUE(object$boundary_solution)) return(rep(0, nrow(z)))
  eta <- object$coefficients[1L] + drop(z %*% object$coefficients[-1L])
  if (object$method == "lime_linear") return(unname(eta))
  eta <- eta + log(object$exposure)
  if (any(!is.finite(eta)) || any(eta > log(.Machine$double.xmax))) {
    stop("Predicted log mean exceeds finite numeric range.")
  }
  unname(exp(eta))
}

# Derivatives with respect to standardized coordinates. For categorical features,
# use exact prediction contrasts rather than interpreting these as discrete jumps.
surrogate_effects <- function(object, z0) {
  z0 <- .surrogate_prediction_matrix(object, z0)
  if (nrow(z0) != 1L) stop("z0 must describe exactly one point.")
  slopes <- object$coefficients[-1L]
  if (object$method == "lime_linear") return(slopes)
  predict_surrogate(object, z0) * slopes
}

.surrogate_metric_inputs <- function(y, prediction, weights) {
  if (!is.numeric(y) || !is.numeric(prediction) || !length(y) ||
      length(y) != length(prediction) || any(!is.finite(y)) || any(!is.finite(prediction))) {
    stop("Metric inputs must be equally sized finite numeric vectors.")
  }
  .surrogate_weights(weights, length(y))
}

weighted_rmse <- function(y, prediction, weights) {
  w <- .surrogate_metric_inputs(y, prediction, weights)
  sqrt(sum(w * (y - prediction)^2))
}

weighted_r2 <- function(y, prediction, weights) {
  w <- .surrogate_metric_inputs(y, prediction, weights)
  error <- sum(w * (y - prediction)^2)
  variation <- sum(w * (y - sum(w * y))^2)
  # R2 has no defined denominator for an exactly constant response. Report 1 for
  # an exact fit (up to roundoff), and NA otherwise; never invent a finite score.
  threshold <- 100 * .Machine$double.eps^2 * max(1, sum(w * y^2))
  if (variation <= threshold) return(if (error <= threshold) 1 else NA_real_)
  1 - error / variation
}

effective_sample_size <- function(weights) {
  w <- .surrogate_weights(weights, length(weights))
  1 / sum(w^2)
}
