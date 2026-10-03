# Independent execution of selected functions from the authors' external R
# source. That GPL source is downloaded separately and is not vendored here.
# Usage: Rscript check_tail_reference.R author.R input.csv output.csv

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Expected author source, input CSV, output CSV")
author <- new.env(parent = globalenv())
sys.source(args[1], envir = author)
input <- read.csv(args[2])
answer <- lapply(split(input, input$case), function(rows) {
  x <- rows$value
  phi <- rows$known_phi[1]
  ell <- rows$bandwidth[1]
  centered <- x - mean(x)
  fitted <- sum(centered[-1] * centered[-length(x)]) / sum(centered^2)
  raw <- author$lrv.np(x, p = 0, q = 1, bw = ell)
  ratio <- function(p, finite = FALSE) {
    denominator <- author$arma11.M(q = 1, l = ell, ar = p, ma = 0, sd = 1)
    numerator <- if (finite) {
      author$arma11.M(q = 1, l = length(x), ar = p, ma = 0, sd = 1)
    } else {
      author$arma11.lrv(ar = p, ma = 0, sd = 1)
    }
    numerator / denominator
  }
  data.frame(case = rows$case[1], fitted_phi = fitted, unadjusted = raw,
             known_lrv = raw * ratio(phi), plugin_lrv = raw * ratio(fitted),
             known_finite = raw * ratio(phi, TRUE), plugin_finite = raw * ratio(fitted, TRUE))
})
options(digits = 17)
write.csv(do.call(rbind, answer), args[3], row.names = FALSE)
