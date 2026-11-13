# Download just the two public CASdatasets objects, pinned to one revision.
# Data stay in the ignored local cache; nothing is silently synthesized.
data_manifest <- function() {
  data.frame(dataset = c("catelematic13", "freMTPL2freq"),
    revision = "227fb56b8734bdb7c0327a41180e01d2ddaeaf26",
    md5 = c("dcecce6f90b4ea8ad2a1a691d8bc931d", "b67f5cf5849b5744b2ce1bc945e4dcc6"),
    stringsAsFactors = FALSE)
}

download_data <- function(dataset, directory = "data/raw") {
  manifest <- data_manifest(); item <- manifest[manifest$dataset == dataset, ]
  if (nrow(item) != 1L) stop("Unknown dataset: ", dataset)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(directory, paste0(dataset, ".rda"))
  if (!file.exists(path)) {
    url <- paste0("https://raw.githubusercontent.com/dutangc/CASdatasets/",
                  item$revision, "/data/", dataset, ".rda")
    temporary <- tempfile(tmpdir = directory, fileext = ".rda")
    on.exit(unlink(temporary), add = TRUE)
    old_timeout <- getOption("timeout"); on.exit(options(timeout = old_timeout), add = TRUE)
    options(timeout = max(300, old_timeout))
    download.file(url, temporary, mode = "wb", quiet = FALSE)
    if (unname(tools::md5sum(temporary)) != item$md5) stop("Downloaded checksum mismatch")
    if (!file.rename(temporary, path)) stop("Cannot move downloaded data to cache")
  }
  if (unname(tools::md5sum(path)) != item$md5)
    stop("Checksum mismatch for ", path, "; remove that cache file and re-download.")
  path
}

load_policy_data <- function(dataset, max_rows = Inf, seed = 456L) {
  env <- new.env(parent = emptyenv()); load(download_data(dataset), envir = env)
  raw <- env[[dataset]]; original_n <- nrow(raw)
  if (dataset == "catelematic13") {
    required <- c("Duration", "NB_Claim", "AMT_Claim", "Car.age")
    stopifnot(all(required %in% names(raw)))
    # Duration is observed days. A year is defined as 365 days throughout.
    y <- raw$NB_Claim; exposure <- raw$Duration / 365
    id <- as.character(seq_len(nrow(raw)))
    x <- raw[setdiff(names(raw), c("Duration", "NB_Claim", "AMT_Claim"))]
    # Predeclared compact feature set avoids exact compositional aliases and
    # redundant intensity thresholds. Geographic Territory is kept categorical.
    features <- c("Insured.age", "Insured.sex", "Car.age", "Marital", "Car.use",
      "Credit.score", "Region", "Annual.miles.drive", "Years.noclaims", "Territory",
      "Annual.pct.driven", "Total.miles.driven", "Pct.drive.wkday",
      "Pct.drive.rush.am", "Pct.drive.rush.pm", "Avgdays.week", "Accel.06miles",
      "Brake.06miles", "Left.turn.intensity08", "Right.turn.intensity08")
    x <- x[features]
    x$Territory <- as.character(x$Territory)
    # The source paper explicitly permits -2/-1 for next-model-year vehicles.
    invalid_age <- sum(x$Car.age < 0, na.rm = TRUE)
    x$Annualized.miles <- x$Total.miles.driven / exposure
    x$Total.miles.driven <- NULL
    log_features <- c("Annual.miles.drive", "Annualized.miles", "Accel.06miles",
                      "Brake.06miles", "Left.turn.intensity08", "Right.turn.intensity08")
  } else if (dataset == "freMTPL2freq") {
    stopifnot(all(c("IDpol", "ClaimNb", "Exposure") %in% names(raw)))
    y <- raw$ClaimNb; exposure <- raw$Exposure; id <- as.character(raw$IDpol)
    if (anyDuplicated(id)) stop("Repeated policy IDs require a grouped split")
    x <- raw[setdiff(names(raw), c("IDpol", "ClaimNb", "Exposure"))]
    log_features <- "Density"; invalid_age <- 0L
  } else stop("Unknown dataset")
  # Different policy IDs can have identical discretized fields. Keep those
  # policies: deleting them based on observed counts would bias frequency.
  valid <- is.finite(y) & y >= 0 & abs(y - round(y)) < 1e-8 &
           is.finite(exposure) & exposure > 0
  retained <- which(valid)
  if (is.finite(max_rows) && length(retained) > max_rows) {
    set.seed(seed); retained <- sort(sample(retained, max_rows))
  }
  x <- x[retained, , drop = FALSE]
  for (name in log_features) {
    if (any(x[[name]] < 0, na.rm = TRUE)) stop("Negative value in log feature ", name)
    x[[name]] <- log1p(x[[name]])
    names(x)[names(x) == name] <- paste0("log1p_", name)
  }
  audit <- data.frame(dataset = dataset, original_rows = original_n,
    invalid_rows_removed = sum(!valid),
    negative_model_year_ages_retained = invalid_age, analyzed_rows = length(retained),
    claim_total = sum(y[retained]), exposure_years = sum(exposure[retained]),
    zero_claim_fraction = mean(y[retained] == 0), predictors = ncol(x))
  list(x = x, y = y[retained], exposure = exposure[retained], id = id[retained], audit = audit)
}

split_policies <- function(n, seed) {
  if (n < 100L) stop("At least 100 valid policies are needed")
  set.seed(seed); order <- sample.int(n)
  a <- floor(0.7 * n); b <- floor(0.85 * n)
  list(train = order[seq_len(a)], validation = order[seq.int(a + 1L, b)],
       test = order[seq.int(b + 1L, n)])
}

# Numeric features use train means/SDs. Categorical contrasts use the most
# common train level as reference; rare levels share OTHER. Unseen levels also
# map to OTHER (or the reference if no OTHER was learned).
fit_preprocessor <- function(x, max_levels = 20L, min_count = 20L) {
  specs <- list(); mapping <- list(); columns <- character()
  for (name in names(x)) {
    v <- x[[name]]
    if (is.numeric(v)) {
      v[!is.finite(v)] <- NA_real_
      med <- if (all(is.na(v))) 0 else median(v, na.rm = TRUE)
      v[is.na(v)] <- med; s <- sd(v); if (!is.finite(s) || s < 1e-10) s <- 1
      specs[[name]] <- list(type = "numeric", median = med, center = mean(v), scale = s,
                            columns = name)
      mapping[[name]] <- data.frame(column = name, feature = name, type = "numeric", level = "", reference = "")
    } else {
      v <- as.character(v); v[is.na(v)] <- "__MISSING__"
      freq <- sort(table(v), decreasing = TRUE)
      keep <- head(names(freq)[freq >= min_count], max_levels)
      if (!length(keep)) keep <- names(freq)[1]
      levels <- c(keep, if (any(!v %in% keep)) "__OTHER__")
      reference <- levels[1]; contrasts <- levels[-1]
      cn <- paste0(name, "=", contrasts)
      if (!length(contrasts)) cn <- character()
      specs[[name]] <- list(type = "categorical", levels = levels, keep = keep,
                           reference = reference, columns = cn)
      mapping[[name]] <- data.frame(column = cn, feature = rep(name, length(cn)),
        type = rep("categorical", length(cn)), level = contrasts, reference = rep(reference, length(cn)))
    }
    columns <- c(columns, specs[[name]]$columns)
  }
  list(specs = specs, columns = columns, mapping = do.call(rbind, mapping))
}

transform_features <- function(preprocessor, x) {
  out <- matrix(0, nrow(x), length(preprocessor$columns),
                dimnames = list(NULL, preprocessor$columns))
  for (name in names(preprocessor$specs)) {
    spec <- preprocessor$specs[[name]]; v <- x[[name]]
    if (spec$type == "numeric") {
      v[!is.finite(v)] <- spec$median
      out[, spec$columns] <- (v - spec$center) / spec$scale
    } else if (length(spec$columns)) {
      v <- as.character(v); v[is.na(v)] <- "__MISSING__"
      fallback <- if ("__OTHER__" %in% spec$levels) "__OTHER__" else spec$reference
      v[!v %in% spec$keep] <- fallback
      for (j in seq_along(spec$columns)) out[, spec$columns[j]] <- as.numeric(v == spec$levels[j + 1L])
    }
  }
  stopifnot(all(is.finite(out))); out
}

# Average per-original-feature distance, so many category dummies do not
# multiply a feature's influence. Numeric differences are capped at 3 SD.
policy_distance <- function(x, anchor, preprocessor) {
  distances <- matrix(0, nrow(x), length(preprocessor$specs))
  j <- 0L
  for (spec in preprocessor$specs) {
    j <- j + 1L; cols <- spec$columns
    if (!length(cols)) next
    delta <- sweep(x[, cols, drop = FALSE], 2, anchor[cols], "-")
    distances[, j] <- if (spec$type == "numeric") pmin(abs(delta[, 1]) / 3, 1)^2 else
      as.numeric(rowSums(abs(delta)) > 0)
  }
  sqrt(rowMeans(distances))
}

make_neighborhood <- function(x_train, anchor, preprocessor, config, seed) {
  distance <- policy_distance(x_train, anchor, preprocessor)
  pool <- head(order(distance), min(config$donor_pool_size, nrow(x_train)))
  needed <- config$evaluation_size + max(config$neighborhood_sizes)
  if (length(pool) < needed) stop("Donor pool must cover disjoint fit/evaluation samples")
  set.seed(seed); evaluation <- sample(pool, config$evaluation_size)
  fit_pool <- setdiff(pool, evaluation)
  # Bandwidth is fixed across repeats and neighborhood SAMPLE sizes.
  bandwidth <- max(stats::median(distance[pool]) * config$kernel_width, 1e-6)
  weights <- exp(-0.5 * (distance / bandwidth)^2)
  list(fit_pool = fit_pool, evaluation = evaluation, weights = weights,
       bandwidth = bandwidth, distance = distance)
}
