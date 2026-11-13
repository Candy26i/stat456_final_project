# Run from the repository root: Rscript tests/test_surrogates.R
source(file.path("R", "surrogates.R"))

assert_close <- function(actual, expected, tolerance = 1e-7) {
  stopifnot(length(actual) == length(expected),
            all(is.finite(actual)), all(is.finite(expected)),
            max(abs(actual - expected)) < tolerance)
}

set.seed(456)
z <- matrix(rnorm(1200), ncol = 3L,
            dimnames = list(NULL, c("age", "density", "vehicle")))
w <- exp(-rowSums(z^2) / 3)
beta <- c(-0.6, 0.25, -0.4, 0.15)
exposure <- 0.7
target <- exposure * exp(beta[1L] + z %*% beta[-1L])
target <- drop(target) # These are fractional expectations, never rounded counts.

# Numerical differentiation checks the actual penalized fractional-target loss.
for (method in c("lime_linear", "poisson", "nb2")) {
  b <- c(0.2, -0.1, 0.3, 0.05)
  objective <- function(b) surrogate_objective(b, z, target, w, method,
                                               exposure, alpha = 0.8, ridge = 0.03)
  exact <- objective(b)
  eps <- 1e-5
  numerical_gradient <- numeric(length(b))
  numerical_hessian <- matrix(NA_real_, length(b), length(b))
  for (j in seq_along(b)) {
    delta <- rep(0, length(b)); delta[j] <- eps
    numerical_gradient[j] <- (objective(b + delta)$value - objective(b - delta)$value) / (2 * eps)
    numerical_hessian[, j] <- (objective(b + delta)$gradient - objective(b - delta)$gradient) / (2 * eps)
  }
  assert_close(exact$gradient, numerical_gradient, 1e-7)
  assert_close(exact$hessian, numerical_hessian, 1e-7)
}

for (method in c("poisson", "nb2")) {
  fit <- fit_surrogate(z, target, w, method, exposure, alpha = 0.8, ridge = 0)
  stopifnot(fit$convergence == 0, isTRUE(fit$converged), !fit$boundary_solution)
  assert_close(fit$coefficients, beta, 1e-6)
  assert_close(predict_surrogate(fit, z), target, 1e-6)
  stopifnot(all(predict_surrogate(fit, z) >= 0))
  assert_close(surrogate_effects(fit, rep(0, 3)), exposure * exp(beta[1L]) * beta[-1L], 1e-6)
  # Exposure appears exactly once as a fixed offset; doubling e0 halves the
  # learned rate intercept when expected-count targets are unchanged.
  double_exposure <- fit_surrogate(z, target, w, method, 2 * exposure,
                                  alpha = 0.8, ridge = 0)
  assert_close(double_exposure$coefficients[1L], beta[1L] - log(2), 1e-6)
  assert_close(predict_surrogate(double_exposure, z), target, 1e-6)
  changed_policy <- fit
  changed_policy$exposure <- 2 * exposure
  assert_close(predict_surrogate(changed_policy, z), 2 * target, 1e-6)
  rescaled_weights <- fit_surrogate(z, target, w * 1e100, method, exposure,
                                    alpha = 0.8, ridge = 0.01)
  reference <- fit_surrogate(z, target, w, method, exposure, alpha = 0.8, ridge = 0.01)
  assert_close(rescaled_weights$coefficients, reference$coefficients, 1e-8)
  # Duplicating observations must not change the effective ridge strength.
  duplicated <- fit_surrogate(rbind(z, z), rep(target, 2), rep(w, 2), method,
                              exposure, alpha = 0.8, ridge = 0.01)
  assert_close(duplicated$coefficients, reference$coefficients, 1e-8)
}

# NB2 converges to the Poisson mean-matching objective as alpha approaches zero.
nonlinear_target <- target * exp(0.2 * z[, 1]^2)
poisson <- fit_surrogate(z, nonlinear_target, w, "poisson", exposure)
nb_small_alpha <- fit_surrogate(z, nonlinear_target, w, "nb2", exposure, alpha = 1e-7)
assert_close(nb_small_alpha$coefficients, poisson$coefficients, 1e-6)

# Identity-link weighted linear regression allows negative predictions.
linear_target <- 10 + z %*% c(0.2, -0.1, 0.3)
linear <- fit_surrogate(z, drop(linear_target), w, "lime_linear", ridge = 0)
assert_close(linear$coefficients, c(10, 0.2, -0.1, 0.3))
assert_close(surrogate_effects(linear, c(0, 0, 0)), c(0.2, -0.1, 0.3))
stopifnot(predict_surrogate(linear, c(-100, 0, 0)) < 0)
linear_scaled <- fit_surrogate(z, drop(linear_target), 20 * w, "lime_linear", ridge = 0)
assert_close(linear_scaled$coefficients, linear$coefficients)

# Redundant / all-zero columns and exact constant responses are legal.
rank_deficient <- cbind(z, zero = 0, duplicate = z[, 1], constant = 1)
for (method in c("lime_linear", "poisson", "nb2")) {
  for (penalty in c(0, 1e-4)) {
    fit <- fit_surrogate(rank_deficient, rep(0.3, nrow(z)), w, method,
                         exposure = 0.4, ridge = penalty)
    stopifnot(fit$convergence == 0)
    assert_close(predict_surrogate(fit, rank_deficient), rep(0.3, nrow(z)), 1e-7)
  }
  zero <- fit_surrogate(z, rep(0, nrow(z)), w, method)
  assert_close(predict_surrogate(zero, z), rep(0, nrow(z)))
  assert_close(surrogate_effects(zero, rep(0, ncol(z))), rep(0, ncol(z)))
}

# Large NB2 trial log means remain numerically stable without clipping.
large <- surrogate_objective(c(1000, 0, 0, 0), z, target, w, "nb2")
stopifnot(is.finite(large$value), all(is.finite(large$gradient)),
          all(is.finite(large$hessian)))
overflow <- surrogate_objective(c(1000, 0, 0, 0), z, target, w, "poisson")
stopifnot(is.infinite(overflow$value))

# Zero-weight observations do not affect fitting.
with_outlier <- fit_surrogate(rbind(z, c(1e4, -1e4, 1e4)), c(target, 1e100),
                              c(w, 0), "poisson", exposure, ridge = 0)
assert_close(with_outlier$coefficients, beta, 1e-6)

# Named prediction columns are reordered safely, never silently mismatched.
assert_close(predict_surrogate(poisson, z[, c(3, 1, 2)]), predict_surrogate(poisson, z))
assert_close(weighted_rmse(c(1, 3), c(1, 1), c(1, 3)), sqrt(3))
assert_close(weighted_r2(c(1, 3), c(1, 3), c(1, 3)), 1)
stopifnot(weighted_r2(rep(2, 3), rep(2, 3), rep(1, 3)) == 1,
          is.na(weighted_r2(rep(2, 3), rep(1, 3), rep(1, 3))))
assert_close(effective_sample_size(c(1, 1, 0)), 2)
assert_close(effective_sample_size(c(1e300, 1e300, 0)), 2)

# A realistic wide neighborhood includes rare one-hot columns, all-zero columns,
# and duplicates; slope ridge must keep the count-model fit identifiable.
set.seed(2026)
wide <- matrix(rnorm(300 * 70), nrow = 300)
categories <- matrix(rbinom(300 * 25, 1, 0.02), nrow = 300)
wide <- cbind(wide, categories, matrix(0, 300, 4), duplicate = wide[, 1])
colnames(wide) <- paste0("feature", seq_len(ncol(wide)))
wide_target <- 0.08 * exp(0.3 * wide[, 1] - 0.2 * wide[, 2] + 0.1 * wide[, 3]^2)
wide_weights <- exp(-rowSums(wide[, 1:5]^2) / 3)
for (method in c("lime_linear", "poisson", "nb2")) {
  fit <- fit_surrogate(wide, wide_target, wide_weights, method,
                       exposure = 0.5, alpha = 2, ridge = 1e-4)
  stopifnot(fit$convergence == 0, all(is.finite(fit$coefficients)))
  stopifnot(weighted_rmse(wide_target, predict_surrogate(fit, wide), wide_weights) < 0.02)
}

# Optimization failure must be visible to callers, not silently marked success.
warning_seen <- FALSE
unfinished <- withCallingHandlers(
  fit_surrogate(z, nonlinear_target, w, "poisson", maxit = 1),
  warning = function(w) { warning_seen <<- TRUE; invokeRestart("muffleWarning") })
stopifnot(warning_seen, unfinished$convergence != 0, !unfinished$converged)

cat("All surrogate tests passed.\n")
