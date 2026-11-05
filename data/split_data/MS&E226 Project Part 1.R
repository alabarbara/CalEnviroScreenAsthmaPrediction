# ============================================================
# CalEnviroScreen: Original data preparation and asthma baseline
#
# Reconstructs the original split and CV folds, checks them
# against saved records, and calculates pooled baseline RMSE.
#
# Base R only. No additional packages required.
# ============================================================


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

# Change this folder if running on another computer.
base_folder <- "/Users/stevensinger/Desktop/Stanford/MS&E226"

raw_path <- file.path(
  base_folder,
  "CalEnviro Data.csv"
)

reference_development_path <- file.path(
  base_folder,
  "asthma_development.rds"
)

reference_split_path <- file.path(
  base_folder,
  "asthma_split_record.rds"
)

output_folder <- file.path(
  base_folder,
  "Asthma_Project_Reconstructed"
)

input_paths <- c(
  raw_path,
  reference_development_path,
  reference_split_path
)

if (!all(file.exists(input_paths))) {
  stop(
    "Input files not found:\n",
    paste(input_paths[!file.exists(input_paths)], collapse = "\n")
  )
}


# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------

to_numeric <- function(x, column_name) {
  text <- trimws(as.character(x))
  
  missing <- is.na(text) |
    toupper(text) %in% c("", "NA", "N/A")
  
  value <- suppressWarnings(as.numeric(text))
  
  if (any(!missing & is.na(value))) {
    stop("Unexpected nonnumeric values in: ", column_name)
  }
  
  value[missing] <- NA_real_
  
  if (any(!is.finite(value[!is.na(value)]))) {
    stop("Infinite values in: ", column_name)
  }
  
  value
}

tract_key <- function(x) {
  value <- to_numeric(x, "census_tract")
  
  if (anyNA(value) || any(value != floor(value))) {
    stop("Missing or invalid census tract identifier.")
  }
  
  sprintf("%.0f", value)
}

rmse <- function(actual, predicted) {
  stopifnot(
    length(actual) == length(predicted),
    length(actual) > 0,
    all(is.finite(actual)),
    all(is.finite(predicted))
  )
  
  sqrt(mean((actual - predicted)^2))
}


# ------------------------------------------------------------
# 3. Import the original CSV
# ------------------------------------------------------------

full_data <- read.csv(
  raw_path,
  na.strings = c("", "NA", "N/A"),
  stringsAsFactors = FALSE,
  check.names = FALSE,
  fileEncoding = "UTF-8-BOM"
)

imported_n <- nrow(full_data)

# Remove completely empty columns.
empty_columns <- vapply(
  full_data,
  function(x) all(is.na(x)),
  logical(1)
)

full_data <- full_data[, !empty_columns, drop = FALSE]

# Remove exact duplicate records before splitting.
before_duplicates <- nrow(full_data)
full_data <- unique(full_data)

duplicates_removed <- before_duplicates - nrow(full_data)

if (!"census_tract" %in% names(full_data)) {
  stop("The raw CSV has no census_tract column.")
}

raw_ids <- tract_key(full_data$census_tract)

# Different records sharing an ID require manual resolution.
if (anyDuplicated(raw_ids) > 0L) {
  stop("Multiple different records share a census tract ID.")
}


# ------------------------------------------------------------
# 4. Retain the original analysis columns
#
# These reproduce the original development file.
# Later removal of education/housing burden and creation of
# region/waste indicators belong in the EDA preparation stage.
# ------------------------------------------------------------

environmental_features <- c(
  "ozone",
  "pm2_5",
  "diesel_pm",
  "drinking_water",
  "pesticides",
  "tox_release",
  "traffic",
  "cleanup_sites",
  "groundwater_threats",
  "haz_waste",
  "imp_water_bodies",
  "solid_waste"
)

socioeconomic_features <- c(
  "education",
  "linguistic_isolation",
  "poverty",
  "unemployment",
  "housing_burden"
)

required_columns <- c(
  "census_tract",
  "asthma",
  environmental_features,
  "california_county",
  socioeconomic_features
)

missing_columns <- setdiff(
  required_columns,
  names(full_data)
)

if (length(missing_columns) > 0L) {
  stop(
    "Required columns not found: ",
    paste(missing_columns, collapse = ", ")
  )
}

numeric_columns <- c(
  "asthma",
  environmental_features,
  socioeconomic_features
)

for (column in numeric_columns) {
  full_data[[column]] <- to_numeric(
    full_data[[column]],
    column
  )
}

# Outcomes must be observed; predictors may remain missing.
missing_outcome_n <- sum(is.na(full_data$asthma))

full_data <- full_data[
  !is.na(full_data$asthma),
  required_columns,
  drop = FALSE
]

# Preserve original row order: do not sort or shuffle here.


# ------------------------------------------------------------
# 5. Reconstruct the original 80/20 split
#
# Explicitly use the standard RNG settings for R >= 3.6.
# Saved-file checks below stop the script if assignments differ.
# ------------------------------------------------------------

if (getRversion() < "3.6.0") {
  stop("Use R version 3.6.0 or later.")
}

RNGkind(
  kind = "Mersenne-Twister",
  normal.kind = "Inversion",
  sample.kind = "Rejection"
)

set.seed(123)

n <- nrow(full_data)
n_holdout <- floor(0.20 * n)

holdout_index <- sample.int(
  n,
  size = n_holdout
)

development_data <- full_data[
  -holdout_index,
  ,
  drop = FALSE
]

# Record only holdout identifiers here; do not evaluate outcomes.
holdout_tracts <- full_data$census_tract[holdout_index]

split_record <- list(
  seed = 123,
  development_tracts = development_data$census_tract,
  holdout_tracts = holdout_tracts
)

# Remove the complete dataset from the active environment.
rm(full_data, raw_ids, holdout_index)


# ------------------------------------------------------------
# 6. Reconstruct the original five CV folds
# ------------------------------------------------------------

set.seed(456)

K <- 5L

development_data$fold_id <- sample(
  rep(
    seq_len(K),
    length.out = nrow(development_data)
  )
)


# ------------------------------------------------------------
# 7. Verify against the original saved split and development file
#
# No output files are written unless all checks pass.
# ------------------------------------------------------------

reference_split <- readRDS(reference_split_path)
reference_development <- readRDS(reference_development_path)

if (!all(
  c("seed", "development_tracts", "holdout_tracts") %in%
  names(reference_split)
)) {
  stop("Saved split record is missing required fields.")
}

if (length(reference_split$seed) != 1L ||
    reference_split$seed != 123) {
  stop("The saved split record does not use seed 123.")
}

# Check exact membership AND row order.
if (!identical(
  tract_key(split_record$development_tracts),
  tract_key(reference_split$development_tracts)
)) {
  stop(
    "Reconstructed development membership/order differs ",
    "from the saved split. No outputs were written."
  )
}

if (!identical(
  tract_key(split_record$holdout_tracts),
  tract_key(reference_split$holdout_tracts)
)) {
  stop(
    "Reconstructed holdout membership/order differs ",
    "from the saved split. No outputs were written."
  )
}

if (!all(
  c("census_tract", "asthma", "fold_id") %in%
  names(reference_development)
)) {
  stop("Saved development file is missing required columns.")
}

reconstructed_ids <- tract_key(development_data$census_tract)
reference_ids <- tract_key(reference_development$census_tract)

if (anyDuplicated(reference_ids) > 0L) {
  stop("Saved development file contains repeated tract IDs.")
}

if (!identical(reconstructed_ids, reference_ids)) {
  stop(
    "Reconstructed development rows/order differ ",
    "from asthma_development.rds."
  )
}

reference_folds <- to_numeric(
  reference_development$fold_id,
  "reference fold_id"
)

if (!identical(
  as.integer(development_data$fold_id),
  as.integer(reference_folds)
)) {
  stop(
    "Reconstructed fold assignments differ from the saved ",
    "development file. No outputs were written."
  )
}

# Check original analysis values, including missingness.
original_columns <- setdiff(
  names(reference_development),
  "fold_id"
)

if (!all(original_columns %in% names(development_data))) {
  stop("Reconstructed data are missing original analysis columns.")
}

for (column in original_columns) {
  actual <- development_data[[column]]
  expected <- reference_development[[column]]
  
  if (column == "census_tract") {
    equal <- identical(
      tract_key(actual),
      tract_key(expected)
    )
  } else if (column %in% numeric_columns) {
    equal <- isTRUE(all.equal(
      to_numeric(actual, column),
      to_numeric(expected, column),
      check.attributes = FALSE
    ))
  } else {
    equal <- identical(
      as.character(actual),
      as.character(expected)
    )
  }
  
  if (!equal) {
    stop(
      "Values differ from the original development file in: ",
      column,
      ". No outputs were written."
    )
  }
}

if (length(intersect(
  tract_key(split_record$development_tracts),
  tract_key(split_record$holdout_tracts)
)) > 0L) {
  stop("Development and holdout memberships overlap.")
}

cat("\nVERIFIED: Original split, folds, and data reproduced.\n")
cat("Imported records:", imported_n, "\n")
cat("Exact duplicates removed:", duplicates_removed, "\n")
cat("Missing asthma outcomes excluded:", missing_outcome_n, "\n")
cat("Eligible records:", n, "\n")
cat("Development observations:", nrow(development_data), "\n")
cat("Holdout observations:", n_holdout, "\n")

print(table(Fold = development_data$fold_id))


# ------------------------------------------------------------
# 8. Calculate the training-mean baseline
#
# Predictor NAs are retained. The baseline uses only outcomes.
# No predictor imputation is needed for this baseline.
# ------------------------------------------------------------

y <- development_data$asthma
fold_id <- development_data$fold_id

baseline_prediction <- rep(
  NA_real_,
  length(y)
)

fold_results <- data.frame(
  fold = seq_len(K),
  training_mean = NA_real_,
  validation_n = NA_integer_,
  validation_rmse = NA_real_
)

for (k in seq_len(K)) {
  training <- fold_id != k
  validation <- fold_id == k
  
  training_mean <- mean(y[training])
  
  baseline_prediction[validation] <- training_mean
  
  fold_results$training_mean[k] <- training_mean
  fold_results$validation_n[k] <- sum(validation)
  
  fold_results$validation_rmse[k] <- rmse(
    y[validation],
    baseline_prediction[validation]
  )
}

baseline_cv_rmse <- rmse(
  y,
  baseline_prediction
)

print(fold_results)

cat(
  "\nBaseline pooled cross-validated RMSE:",
  round(baseline_cv_rmse, 4),
  "\n"
)

# Verify the previously reported baseline.
if (abs(baseline_cv_rmse - 30.418279823448962) > 1e-6) {
  stop(
    "Baseline differs from the verified original result. ",
    "No outputs were written."
  )
}


# ------------------------------------------------------------
# 9. Save verified outputs
#
# The original holdout file is neither read nor overwritten.
# ------------------------------------------------------------

dir.create(
  output_folder,
  recursive = TRUE,
  showWarnings = FALSE
)

saveRDS(
  development_data,
  file.path(
    output_folder,
    "asthma_development.rds"
  )
)

saveRDS(
  split_record,
  file.path(
    output_folder,
    "asthma_split_record.rds"
  )
)

write.csv(
  fold_results,
  file.path(
    output_folder,
    "asthma_baseline_by_fold.csv"
  ),
  row.names = FALSE
)

write.csv(
  data.frame(
    model = "Training-mean baseline",
    development_n = nrow(development_data),
    cv_rmse = baseline_cv_rmse
  ),
  file.path(
    output_folder,
    "asthma_baseline_summary.csv"
  ),
  row.names = FALSE
)

capture.output(
  sessionInfo(),
  file = file.path(
    output_folder,
    "session_info.txt"
  )
)

cat(
  "\nVerified files saved in:\n",
  output_folder,
  "\n"
)

# This script ends after original preparation and baseline.
# Naomi's EDA script performs subsequent feature engineering.
# Predictive models must fit imputation and other learned
# preprocessing within each cross-validation training split.