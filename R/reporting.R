write_csv <- function(x, file) write.csv(x, file, row.names = FALSE, na = "")

html_table <- function(d, digits = 5L) {
  escape <- function(x) {
    x <- gsub("&", "&amp;", as.character(x), fixed = TRUE)
    x <- gsub("<", "&lt;", x, fixed = TRUE)
    gsub(">", "&gt;", x, fixed = TRUE)
  }
  values <- lapply(d, function(v) if (is.numeric(v)) format(round(v, digits), trim = TRUE) else v)
  header <- paste0("<tr>", paste0("<th>", escape(names(d)), "</th>", collapse = ""), "</tr>")
  rows <- vapply(seq_len(nrow(d)), function(i)
    paste0("<tr>", paste0("<td>", escape(vapply(values, function(v) as.character(v[i]), character(1))), "</td>", collapse = ""), "</tr>"), character(1))
  paste0("<div class='table'><table>", header, paste(rows, collapse = "\n"), "</table></div>")
}

write_report <- function(directory, config, audit, prediction, summary, stability) {
  colors <- c(lime_linear = "#a54c16", poisson = "#2864a5", nb2 = "#18806b")
  panels <- unique(summary[c("dataset", "model")])
  png(file.path(directory, "fidelity.png"), width = 1600, height = 1200, res = 150)
  par(mfrow = c(2, 2), mar = c(4.8, 4.8, 3, 1), oma = c(0, 0, 2, 0))
  for (i in seq_len(nrow(panels))) {
    d <- summary[summary$dataset == panels$dataset[i] & summary$model == panels$model[i], ]
    ylim <- c(0, max(d$weighted_rmse) * 1.35); if (diff(ylim) == 0) ylim <- c(0, 1)
    plot(range(d$neighborhood_size), ylim, type = "n", xlab = "Neighborhood fit sample size",
      ylab = "Weighted RMSE (expected claims)", main = paste(panels$dataset[i], panels$model[i]))
    for (method in names(colors)) {
      s <- d[d$method == method, ]; s <- s[order(s$neighborhood_size), ]
      j <- match(method, names(colors))
      lines(s$neighborhood_size, s$weighted_rmse, type = "b", col = colors[method], lwd = 2,
            pch = c(16, 17, 15)[j], lty = j)
    }
    legend("topright", c("LIME-style linear", "Poisson", "NB2"), col = colors,
           lty = 1:3, pch = c(16, 17, 15), bty = "n", cex = 0.75)
  }
  mtext(paste("Local surrogate fidelity -", config$mode, "run"), outer = TRUE, font = 2)
  dev.off()
  stable_summary <- aggregate(stability[c("mean_spearman", "mean_top_k_jaccard")],
    stability[c("dataset", "model", "neighborhood_size", "method")],
    function(x) if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE))
  content <- c("<!doctype html><html lang='en'><meta charset='utf-8'>",
    "<meta name='viewport' content='width=device-width, initial-scale=1'>",
    "<title>STAT 456 - Local GLM surrogate experiment</title>",
    "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#1e293b}h1,h2{line-height:1.2}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf2f7}.table{overflow:auto}img{width:100%}.note{background:#eef5fa;padding:16px;border-left:4px solid #2864a5}</style>",
    "<h1>Evaluating GLM-based local surrogates</h1>",
    paste0("<p>Mode: <strong>", config$mode, "</strong>; seed: ", config$seed,
           "; NB2 alpha: ", config$nb_alpha, "; ridge: ", config$ridge, ".</p>"),
    "<p class='note'>This is an executed experiment, not a claim that any method is universally better. Quick/smoke runs verify the workflow and use subsamples. Full runs are available in the runner. Uncertainty across policies needs more anchors and repeated model-training seeds.</p>",
    "<h2>Data audit</h2>", html_table(audit),
    "<h2>Held-out predictive performance</h2><p>Blackboxes are evaluated against observed counts on the test split. The constant-rate baseline is estimated only from training data.</p>", html_table(prediction),
    "<h2>Local fidelity</h2><p>Surrogates match blackbox expected counts at fixed anchor exposure. Evaluation donor rows are excluded from every local fit. Means first average repeated neighborhoods within each anchor, then average anchors equally. Lower RMSE is better.</p>",
    "<img src='fidelity.png' alt='Held-out local fidelity by dataset, blackbox, surrogate and sample size'>", html_table(summary),
    "<h2>Explanation stability</h2><p>Spearman feature-importance correlation and top-5 Jaccard are averaged across pairs of repeated neighborhoods per anchor, then across anchors. Only consistently supported features/contrasts enter each anchor comparison. Higher values indicate more stable explanations.</p>", html_table(stable_summary),
    "<h2>Interpretation and limits</h2><p>Numeric effects are expected-claim derivatives per training SD; categorical effects are exact level-versus-reference contrasts. Native GLM slopes are on the log-mean scale and are not directly comparable to linear slopes. Detailed CSVs include support flags, coefficient variation, sign agreement, anchor error, negative linear predictions, effective sample size and convergence.</p>",
    "<p>The linear comparator is LIME-style weighted ridge using the same empirical neighborhoods and encoding as the GLMs. It is not the complete canonical LIME algorithm. Nearest-neighbor resampling preserves observed combinations but is a restricted local distribution. Correlated predictors and small counts can limit interpretation. Explanations describe a fitted model, not causal effects. Canadian telematics are contemporaneous coverage-period features, so this is retrospective claim-frequency modeling.</p>",
    "<p>See <code>README.md</code> and <code>docs/methodology.md</code> for sources, equations and reproducibility details.</p></html>")
  writeLines(content, file.path(directory, "report.html"))
}
