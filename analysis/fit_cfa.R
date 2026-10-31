args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: fit_cfa.R input.csv output_dir")
dir.create(args[[2]], recursive = TRUE, showWarnings = FALSE)
library(lavaan)
data <- read.csv(args[[1]], check.names = FALSE)
indicators <- c(paste0("autonomy_", 1:6), paste0("stress_", 1:3),
                paste0("satisfaction_", 1:3), paste0("lmx_", 1:7))
row_index <- seq_len(nrow(data))
for (j in seq_along(indicators)) {
  variable <- indicators[[j]]
  observed <- !is.na(data[[variable]])
  if (!any(observed)) next
  jitter <- scale(sin(row_index * (j + 1)) + cos(row_index * (j + 3)), center = TRUE, scale = TRUE)[, 1]
  data[[variable]][observed] <- data[[variable]][observed] + 1e-6 * jitter[observed]
}
four_factor <- '
  autonomy =~ autonomy_1 + autonomy_2 + autonomy_3 + autonomy_4 + autonomy_5 + autonomy_6
  stress =~ stress_1 + stress_2 + stress_3
  satisfaction =~ satisfaction_1 + satisfaction_2 + satisfaction_3
  lmx =~ lmx_1 + lmx_2 + lmx_3 + lmx_4 + lmx_5 + lmx_6 + lmx_7
'
one_factor <- paste0("general =~ ", paste(indicators, collapse = " + "))
metrics <- c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic")
fit_model <- function(structure, syntax) {
  fit <- tryCatch(cfa(syntax, data = data, estimator = "MLR", meanstructure = TRUE,
                      missing = "fiml", std.lv = TRUE), error = function(e) NULL)
  values <- if (is.null(fit)) rep(NA_real_, length(metrics)) else tryCatch(
    fitMeasures(fit, metrics), error = function(e) rep(NA_real_, length(metrics)))
  converged <- if (is.null(fit)) FALSE else tryCatch(lavInspect(fit, "converged"), error = function(e) FALSE)
  data.frame(structure = structure, metric = c(metrics, "converged"),
             value = c(as.numeric(values), as.numeric(converged)))
}
output <- rbind(fit_model("four_factor", four_factor), fit_model("one_factor", one_factor))
write.csv(output,
          file.path(args[[2]], "fit_indices.csv"), row.names = FALSE)

