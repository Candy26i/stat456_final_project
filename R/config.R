project_config <- function(mode = "quick") {
  stopifnot(mode %in% c("smoke", "quick", "full"))
  config <- list(mode = mode, seed = 456L,
    datasets = c("catelematic13", "freMTPL2freq"),
    max_rows = 10000L, anchors = 3L, repeats = 3L,
    neighborhood_sizes = c(200L, 500L), evaluation_size = 250L,
    donor_pool_size = 1800L, kernel_width = 0.75,
    nb_alpha = 1, ridge = 1e-4, top_k = 5L,
    boosting_rounds = 100L, boosting_depth = 3L, learning_rate = 0.05,
    hidden_units = 12L, neural_epochs = 70L, batch_size = 256L,
    neural_learning_rate = 0.005, patience = 15L, l2 = 0.001)
  if (mode == "smoke") {
    config$max_rows <- 1800L; config$anchors <- 1L; config$repeats <- 2L
    config$neighborhood_sizes <- c(80L, 150L)
    config$evaluation_size <- 80L; config$donor_pool_size <- 600L
    config$boosting_rounds <- 12L; config$neural_epochs <- 8L
  }
  if (mode == "full") {
    config$max_rows <- Inf; config$anchors <- 20L; config$repeats <- 10L
    config$neighborhood_sizes <- c(250L, 500L, 1000L)
    config$evaluation_size <- 500L; config$donor_pool_size <- 4000L
    config$boosting_rounds <- 500L; config$neural_epochs <- 200L
  }
  config
}
