args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: fit_sem.R input.csv output_dir")
dir.create(args[[2]], recursive = TRUE, showWarnings = FALSE)
library(lavaan)
data <- read.csv(args[[1]], check.names = FALSE)
variables <- c("remote_work_intensity", paste0("autonomy_", 1:6), paste0("stress_", 1:3),
               paste0("satisfaction_", 1:3), paste0("lmx_", 1:7))
row_index <- seq_len(nrow(data))
for (j in seq_along(variables)) {
  variable <- variables[[j]]
  observed <- !is.na(data[[variable]])
  if (!any(observed)) next
  jitter <- scale(sin(row_index * (j + 1)) + cos(row_index * (j + 3)), center = TRUE, scale = TRUE)[, 1]
  data[[variable]][observed] <- data[[variable]][observed] + 1e-6 * jitter[observed]
}
model <- '
  autonomy =~ autonomy_1 + autonomy_2 + autonomy_3 + autonomy_4 + autonomy_5 + autonomy_6
  stress =~ stress_1 + stress_2 + stress_3
  satisfaction =~ satisfaction_1 + satisfaction_2 + satisfaction_3
  lmx =~ lmx_1 + lmx_2 + lmx_3 + lmx_4 + lmx_5 + lmx_6 + lmx_7
  autonomy ~ a*remote_work_intensity
  stress ~ c1*remote_work_intensity + b1*autonomy
  satisfaction ~ c2*remote_work_intensity + b2*autonomy
  lmx ~ c3*remote_work_intensity + b3*autonomy
  stress ~~ 0*satisfaction
  stress ~~ 0*lmx
  satisfaction ~~ 0*lmx
'
fit <- sem(model, data = data, estimator = "MLR", meanstructure = TRUE, fixed.x = FALSE)
paths <- standardizedSolution(fit, ci = TRUE)
paths <- paths[paths$op == "~" & paths$label %in% c("a", "c1", "c2", "c3", "b1", "b2", "b3"), ]
output <- data.frame(analysis = "sem", path = paths$label, outcome = paths$lhs,
                     predictor = paths$rhs, estimate = paths$est.std,
                     std_error = paths$se, p_value = paths$pvalue,
                     ci95_low = paths$ci.lower, ci95_high = paths$ci.upper)
write.csv(output, file.path(args[[2]], "standardized_paths.csv"), row.names = FALSE)
measures <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr", "aic", "bic"))
write.csv(data.frame(metric = names(measures), value = as.numeric(measures)),
          file.path(args[[2]], "fit_indices.csv"), row.names = FALSE)

