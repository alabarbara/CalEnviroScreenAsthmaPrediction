# ============================================================
# 1. Load the existing development data
# ============================================================

# If this relative path fails, use:
# development_data <- readRDS(file.choose())
# Select asthma_development.rds, NOT the holdout file.

development_data <- readRDS(
  "../data/split_data/asthma_development.rds"
)

stopifnot(
  "asthma" %in% names(development_data),
  "fold_id" %in% names(development_data),
  "census_tract" %in% names(development_data)
)


# ============================================================
# 2. Prepare the asthma outcome
# ============================================================

# Convert to numeric and identify missing-value markers.
asthma_text <- trimws(
  as.character(development_data$asthma)
)

missing_marker <- is.na(asthma_text) |
  toupper(asthma_text) %in% c("", "NA", "N/A")

asthma_numeric <- suppressWarnings(
  as.numeric(asthma_text)
)

# Stop if unexpected text could not be converted.
unexpected <- !missing_marker & is.na(asthma_numeric)

if (any(unexpected)) {
  stop("Unexpected nonnumeric values in the asthma column.")
}

asthma_numeric[missing_marker] <- NA_real_

if (any(!is.finite(asthma_numeric[!is.na(asthma_numeric)]))) {
  stop("The asthma outcome contains infinite values.")
}

# Exclude missing outcomes; do not impute them.
eligible <- !is.na(asthma_numeric)

cat(
  "Development observations before exclusions:",
  nrow(development_data), "\n"
)

cat(
  "Excluded for missing asthma outcome:",
  sum(!eligible), "\n"
)

asthma_development <- development_data[
  eligible, , drop = FALSE
]

asthma_development$asthma <- asthma_numeric[eligible]

# Remove the old binary response to avoid using it accidentally.
asthma_development$y <- NULL

y_asthma <- asthma_development$asthma

# Retain the saved fold assignments.
fold_id <- asthma_development$fold_id

stopifnot(
  !anyNA(fold_id),
  all(fold_id %in% 1:5),
  all(1:5 %in% fold_id)
)

cat(
  "Eligible development observations:",
  nrow(asthma_development), "\n"
)

print(table(Fold = fold_id))


# ============================================================
# 3. Define RMSE
# ============================================================

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}


# ============================================================
# 4. Evaluate the training-mean baseline
# ============================================================

baseline_prediction <- rep(
  NA_real_,
  length(y_asthma)
)

fold_results_asthma <- data.frame(
  fold = 1:5,
  training_mean = NA_real_,
  validation_n = NA_integer_,
  validation_rmse = NA_real_
)

for (k in 1:5) {
  
  training <- fold_id != k
  validation <- fold_id == k
  
  # Calculate the mean using training observations ONLY.
  mean_k <- mean(y_asthma[training])
  
  # Assign that mean to every validation observation.
  baseline_prediction[validation] <- mean_k
  
  fold_results_asthma$training_mean[k] <- mean_k
  
  fold_results_asthma$validation_n[k] <- sum(validation)
  
  fold_results_asthma$validation_rmse[k] <- rmse(
    y_asthma[validation],
    baseline_prediction[validation]
  )
}

stopifnot(!anyNA(baseline_prediction))

# Pool squared errors across all out-of-fold predictions,
# then take the square root.
# Do NOT simply average the five fold RMSE values.
baseline_cv_rmse <- rmse(
  y_asthma,
  baseline_prediction
)


# ============================================================
# 5. Display the result for part (c)
# ============================================================

print(fold_results_asthma)

cat(
  "\nAsthma baseline cross-validated RMSE:",
  round(baseline_cv_rmse, 4),
  "\n"
)


# ============================================================
# 6. Save development data and results
# ============================================================

dir.create("../data/split_data", showWarnings = FALSE, recursive = TRUE)

# Save separately; preserve the original development file.
saveRDS(
  asthma_development,
  "../data/split_data/asthma_development_data.rds"
)

write.csv(
  fold_results_asthma,
  "../data/split_data/asthma_baseline_by_fold.csv",
  row.names = FALSE
)

write.csv(
  data.frame(
    model = "Training-mean baseline",
    cv_rmse = baseline_cv_rmse,
    development_n = length(y_asthma)
  ),
  "../data/split_data/asthma_baseline_summary.csv",
  row.names = FALSE
)

cat(
  "\nSummary saved to:\n",
  file.path(
    getwd(),
    "..",
    "data",
    "split_data",
    "asthma_baseline_summary.csv"
  ),
  "\n"
)
