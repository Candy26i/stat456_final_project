list(mode = "quick", seed = 456L, datasets = c("catelematic13",
"freMTPL2freq"), max_rows = 10000L, anchors = 3L, repeats = 3L,
    neighborhood_sizes = c(200L, 500L), evaluation_size = 250L,
    donor_pool_size = 1800L, kernel_width = 0.75, nb_alpha = 1,
    ridge = 1e-04, top_k = 5L, boosting_rounds = 100L, boosting_depth = 3L,
    learning_rate = 0.05, hidden_units = 12L, neural_epochs = 70L,
    batch_size = 256L, neural_learning_rate = 0.005, patience = 15L,
    l2 = 0.001)
