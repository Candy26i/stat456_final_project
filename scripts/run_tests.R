#!/usr/bin/env Rscript
file_argument <- grep("^--file=", commandArgs(), value = TRUE)
if (length(file_argument)) setwd(dirname(dirname(normalizePath(sub("^--file=", "", file_argument[1])))))
for (test in sort(list.files("tests", pattern = "^test_.*\\.R$", full.names = TRUE))) {
  cat("Running", test, "\n")
  status <- system2(file.path(R.home("bin"), "Rscript"), shQuote(test))
  if (status != 0L) stop("Test failed: ", test)
}
cat("All tests passed.\n")
