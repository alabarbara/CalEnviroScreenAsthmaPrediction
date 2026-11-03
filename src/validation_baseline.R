# ============================================================
# Training-mean baseline on EDA modeling CSV
# ============================================================
# Expects data/clean_data/eda_training.csv from EDA_notebook.Rmd:
#   engineered features, correlated cols dropped, covariates median-imputed,
#   existing fold_id retained, asthma NOT imputed.
# Drops NA-asthma rows BEFORE CV, then reuses fold_id for 5-fold RMSE.
# ============================================================

# ------------------------------------------------------------
# 1. Load EDA training CSV
# ------------------------------------------------------------

csv_candidates <- c(
  "../data/clean_data/eda_training.csv",
  "data/clean_data/eda_training.csv"
)
csv_path <- csv_candidates[file.exists(csv_candidates)][1]
if (is.na(csv_path)) {
  stop(
    "Could not find eda_training.csv. Knit/run EDA_notebook.Rmd first."
  )
}

development_data <- read.csv(csv_path, stringsAsFactors = FALSE)

stopifnot(
  "asthma" %in% names(development_data),
  "fold_id" %in% names(development_data),
  "census_tract" %in% names(development_data)
)

cat("Loaded:", normalizePath(csv_path), "\n")


# ------------------------------------------------------------
# 2. Prepare asthma; drop missing outcomes BEFORE CV
# ------------------------------------------------------------

asthma_text <- trimws(as.character(development_data$asthma))

missing_marker <- is.na(development_data$asthma) |
  is.na(asthma_text) |
  toupper(asthma_text) %in% c("", "NA", "N/A")

asthma_numeric <- suppressWarnings(as.numeric(asthma_text))
asthma_numeric[missing_marker] <- NA_real_

unexpected <- !missing_marker & is.na(asthma_numeric)
if (any(unexpected)) {
  stop("Unexpected nonnumeric values in the asthma column.")
}

if (any(!is.finite(asthma_numeric[!is.na(asthma_numeric)]))) {
  stop("The asthma outcome contains infinite values.")
}

# Exclude missing outcomes before any fold-based scoring.
# Do not impute asthma; covariates were already imputed in EDA.
eligible <- !is.na(asthma_numeric)

cat(
  "Development observations before exclusions:",
  nrow(development_data), "\n"
)
cat(
  "Excluded for missing asthma outcome:",
  sum(!eligible), "\n"
)

asthma_development <- development_data[eligible, , drop = FALSE]
asthma_development$asthma <- asthma_numeric[eligible]

# Remove legacy binary response if present
asthma_development$y <- NULL

y_asthma <- asthma_development$asthma
fold_id <- asthma_development$fold_id

stopifnot(
  !anyNA(y_asthma),
  !anyNA(fold_id),
  all(fold_id %in% 1:5),
  all(1:5 %in% fold_id)
)

cat(
  "Eligible development observations (post NA-asthma drop):",
  nrow(asthma_development), "\n"
)
print(table(Fold = fold_id))


# ------------------------------------------------------------
# 3. RMSE helper
# ------------------------------------------------------------

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}


# ------------------------------------------------------------
# 4. Training-mean baseline (each fold as validation once)
# ------------------------------------------------------------
# Covariates are unused by this baseline. Val scoring uses only
# observed asthma (already guaranteed by the pre-CV drop).

baseline_prediction <- rep(NA_real_, length(y_asthma))

fold_results_asthma <- data.frame(
  fold = 1:5,
  training_mean = NA_real_,
  validation_n = NA_integer_,
  validation_rmse = NA_real_
)

for (k in 1:5) {
  training <- fold_id != k
  validation <- fold_id == k

  # Fit on training outcomes only (all observed after pre-CV drop).
  mean_k <- mean(y_asthma[training])

  baseline_prediction[validation] <- mean_k
  fold_results_asthma$training_mean[k] <- mean_k
  fold_results_asthma$validation_n[k] <- sum(validation)
  fold_results_asthma$validation_rmse[k] <- rmse(
    y_asthma[validation],
    baseline_prediction[validation]
  )
}

stopifnot(!anyNA(baseline_prediction))

# Pooled CV RMSE over all out-of-fold predictions (not mean of fold RMSEs).
baseline_cv_rmse <- rmse(y_asthma, baseline_prediction)


# ------------------------------------------------------------
# 5. Display
# ------------------------------------------------------------

print(fold_results_asthma)
cat(
  "\nAsthma baseline cross-validated RMSE:",
  round(baseline_cv_rmse, 4),
  "\n"
)


# ------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------

split_dirs <- c("../data/split_data", "data/split_data")
split_dir <- if (dir.exists("../data/split_data")) {
  "../data/split_data"
} else if (dir.exists("data/split_data")) {
  "data/split_data"
} else {
  dir.create("../data/split_data", showWarnings = FALSE, recursive = TRUE)
  "../data/split_data"
}

# Eligible development rows used for CV (NA asthma already removed).
saveRDS(
  asthma_development,
  file.path(split_dir, "asthma_development_data.rds")
)

write.csv(
  fold_results_asthma,
  file.path(split_dir, "asthma_baseline_by_fold.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(
    model = "Training-mean baseline",
    cv_rmse = baseline_cv_rmse,
    development_n = length(y_asthma),
    excluded_missing_asthma = sum(!eligible)
  ),
  file.path(split_dir, "asthma_baseline_summary.csv"),
  row.names = FALSE
)

cat(
  "\nSummary saved to:\n",
  normalizePath(file.path(split_dir, "asthma_baseline_summary.csv")),
  "\n"
)
