#!/usr/bin/env Rscript
file_argument <- grep("^--file=", commandArgs(), value = TRUE)
if (length(file_argument)) setwd(dirname(dirname(normalizePath(sub("^--file=", "", file_argument[1])))))
for (module in c("config", "data", "models", "surrogates", "evaluation", "reporting"))
  source(file.path("R", paste0(module, ".R")))

args <- commandArgs(trailingOnly = TRUE)
if ("--help" %in% args) {
  cat("Usage: Rscript scripts/run_analysis.R [--mode quick|smoke|full]\n",
      "  [--datasets catelematic13,freMTPL2freq] [--max-rows 10000]\n",
      "  [--anchors 3] [--repeats 3] [--seed 456] [--nb-alpha 1]\n",
      "  [--ridge 0.0001] [--kernel-width 0.75] [--output results/quick]\n")
  quit(status = 0)
}
if (length(args) %% 2 != 0L) stop("Options require --name value pairs. Use --help.")
options <- if (length(args)) setNames(as.list(args[seq.int(2, length(args), 2)]), args[seq.int(1, length(args), 2)]) else list()
allowed <- c("--mode", "--datasets", "--max-rows", "--anchors", "--repeats", "--seed", "--nb-alpha", "--ridge", "--kernel-width", "--output")
if (any(!names(options) %in% allowed) || anyDuplicated(names(options))) stop("Unknown or duplicate option")
config <- project_config(if (is.null(options[["--mode"]])) "quick" else options[["--mode"]])
for (name in c("max-rows", "anchors", "repeats", "seed", "nb-alpha", "ridge", "kernel-width")) {
  if (!is.null(options[[paste0("--", name)]])) config[[gsub("-", "_", name)]] <- as.numeric(options[[paste0("--", name)]])
}
if (!is.null(options[["--datasets"]])) config$datasets <- strsplit(options[["--datasets"]], ",", fixed = TRUE)[[1]]
stopifnot(all(config$datasets %in% data_manifest()$dataset), !anyDuplicated(config$datasets),
  length(config$datasets) > 0L, config$max_rows >= 1000, config$anchors >= 1,
  config$repeats >= 2, config$nb_alpha > 0, config$ridge >= 0, config$kernel_width > 0,
  all(is.finite(c(config$anchors, config$repeats, config$seed, config$nb_alpha, config$ridge, config$kernel_width))))
for (name in c("anchors", "repeats", "seed"))
  if (config[[name]] != floor(config[[name]])) stop(name, " must be an integer")
if (!requireNamespace("xgboost", quietly = TRUE) || packageVersion("xgboost") < "3.2.1.1")
  stop("This project requires xgboost >= 3.2.1.1. Run Rscript scripts/install_dependencies.R.")
directory <- options[["--output"]]
if (is.null(directory)) directory <- file.path("results", config$mode)
# Avoid mixing results from two configurations or overwriting a completed run.
if (dir.exists(directory) && length(list.files(directory, all.files = TRUE, no.. = TRUE)))
  stop("Output directory is not empty: ", directory, ". Choose a new --output path.")
dir.create(directory, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(directory, "models"), showWarnings = FALSE)
writeLines(trimws(capture.output(dput(config)), which = "right"), file.path(directory, "config.R"))
write_csv(data_manifest(), file.path(directory, "data_manifest.csv"))
started <- Sys.time()
all_fidelity <- all_coefficients <- all_effects <- all_prediction <- all_audit <- list()
for (dataset in config$datasets) {
  # Stable dataset-specific seeds: running just one dataset produces the same run.
  dataset_seed <- config$seed + match(dataset, data_manifest()$dataset) * 100000L
  message("Loading ", dataset)
  data <- load_policy_data(dataset, config$max_rows, dataset_seed)
  splits <- split_policies(length(data$y), dataset_seed + 1L)
  required_donors <- config$evaluation_size + max(config$neighborhood_sizes)
  if (min(length(splits$train), config$donor_pool_size) < required_donors)
    stop("Insufficient training donors for disjoint local fit/evaluation samples: need ",
         required_donors, ". Increase --max-rows or use --mode smoke.")
  preprocessor <- fit_preprocessor(data$x[splits$train, , drop = FALSE])
  z <- transform_features(preprocessor, data$x)
  saveRDS(list(preprocessor = preprocessor, splits = splits, policy_ids = data$id),
    file.path(directory, "models", paste0(dataset, "_preprocessing.rds")))
  write_csv(preprocessor$mapping, file.path(directory, paste0(dataset, "_feature_dictionary.csv")))
  assignment <- rep("", length(data$y)); for (name in names(splits)) assignment[splits[[name]]] <- name
  write_csv(data.frame(policy_id = data$id, split = assignment), file.path(directory, paste0(dataset, "_splits.csv")))
  all_audit[[dataset]] <- cbind(data$audit, training_rows = length(splits$train),
    validation_rows = length(splits$validation), test_rows = length(splits$test), encoded_features = ncol(z))
  ytest <- data$y[splits$test]; etest <- data$exposure[splits$test]
  constant <- sum(data$y[splits$train]) / sum(data$exposure[splits$train])
  all_prediction[[paste0(dataset, "_constant")]] <- cbind(dataset = dataset, model = "constant_rate",
    predictive_metrics(ytest, etest, rep(max(constant, 1e-12), length(ytest))))
  for (model_name in c("boosting", "neural")) {
    message("Training ", dataset, " / ", model_name)
    fit_function <- if (model_name == "boosting") fit_boosting else fit_neural
    model <- fit_function(z[splits$train, , drop = FALSE], data$y[splits$train], data$exposure[splits$train],
      z[splits$validation, , drop = FALSE], data$y[splits$validation], data$exposure[splits$validation],
      config = config, seed = dataset_seed + 2L)
    name <- paste(dataset, model_name, sep = "_")
    write_csv(model$training, file.path(directory, paste0(name, "_training.csv")))
    saveRDS(model, file.path(directory, "models", paste0(name, ".rds")))
    if (model_name == "boosting") xgboost::xgb.save(model$booster, file.path(directory, "models", paste0(name, ".ubj")))
    rates <- predict_rate(model, z[splits$test, , drop = FALSE])
    all_prediction[[name]] <- cbind(dataset = dataset, model = model_name, predictive_metrics(ytest, etest, rates))
    write_csv(data.frame(policy_id = data$id[splits$test], observed_count = ytest,
      exposure_years = etest, predicted_rate = rates, predicted_count = etest * rates),
      file.path(directory, paste0(name, "_test_predictions.csv")))
    message("Explaining ", name)
    local <- local_experiments(model, model_name, data, splits, z, preprocessor, config, dataset_seed)
    all_fidelity[[name]] <- local$fidelity; all_coefficients[[name]] <- local$coefficients
    all_effects[[name]] <- local$effects
    saveRDS(local$neighborhoods, file.path(directory, "models", paste0(name, "_neighborhoods.rds")))
  }
}
fidelity <- do.call(rbind, all_fidelity); coefficients <- do.call(rbind, all_coefficients)
effects <- do.call(rbind, all_effects); prediction <- do.call(rbind, all_prediction)
audit <- do.call(rbind, all_audit); summary <- summarize_fidelity(fidelity)
message("Summarizing repeated explanations")
stability <- stability_metrics(effects, coefficients, config$top_k)
for (name in c("fidelity", "coefficients", "effects", "prediction", "audit", "summary"))
  write_csv(get(name), file.path(directory, paste0(name, ".csv")))
write_csv(stability$rankings, file.path(directory, "ranking_stability.csv"))
write_csv(stability$coefficients, file.path(directory, "coefficient_stability.csv"))
write_report(directory, config, audit, prediction, summary, stability$rankings)
writeLines(c(trimws(capture.output(sessionInfo()), which = "right"), paste("Elapsed seconds:", round(as.numeric(difftime(Sys.time(), started, units = "secs")), 2))),
           file.path(directory, "session_info.txt"))
message("Completed: ", normalizePath(file.path(directory, "report.html")))
