# MS&E 226, Part 2 — Question 3a
# Run from the repository folder. No extra R packages needed.

dev_path <- "data/clean_data/eda_training.csv"
raw_path <- "data/calenviroscreen.csv"

# Use the saved Part 1 export for development membership and region.
dev <- read.csv(dev_path, colClasses = c(census_tract = "character"))
raw <- read.csv(raw_path, colClasses = c(census_tract = "character"))

# Some exports store tract IDs with a trailing ".0".
tract_id <- function(x) sub("\\.0+$", "", trimws(x))
dev_ids <- tract_id(dev$census_tract)
raw_ids <- tract_id(raw$census_tract)

stopifnot(nrow(dev) == 6428L, !anyDuplicated(dev_ids),
          !anyDuplicated(raw_ids))

idx <- match(dev_ids, raw_ids)
stopifnot(!anyNA(idx))

# Use original values so this analysis preserves missingness.
variables <- c("asthma", "poverty", "unemployment",
               "linguistic_isolation", "ozone", "pm2_5")
analysis <- raw[idx, variables, drop = FALSE]
analysis$region <- factor(
  dev$region,
  levels = c("Bay Area", "Other", "San Joaquin Valley", "SoCal")
)
stopifnot(!anyNA(analysis$region))

# Keep the same formula, reference region, and complete-case rule for 3b.
spec <- asthma ~ poverty + unemployment + linguistic_isolation +
  ozone + pm2_5 + region

fit <- lm(spec, data = analysis, na.action = na.omit)

alpha <- 0.05
coefs <- coef(summary(fit))
ci <- confint(fit, level = 0.95)

results <- data.frame(
  term = rownames(coefs),
  estimate = coefs[, "Estimate"],
  ci_lower = ci[, 1],
  ci_upper = ci[, 2],
  p_value = coefs[, "Pr(>|t|)"],
  significant_at_5_pct = coefs[, "Pr(>|t|)"] < alpha,
  row.names = NULL
)

cat("Development tracts used:", nobs(fit), "of", nrow(dev), "\n")
print(results, digits = 4, row.names = FALSE)

dir.create("part2/results", recursive = TRUE, showWarnings = FALSE)
write.csv(results, "part2/results/inference_development.csv", row.names = FALSE)

# MS&E 226, Part 2 — Question 3b
# Append after 3a: uses raw, raw_ids, dev_ids, variables, spec,
# analysis, alpha, and results from the code above.

# Recover the same holdout used in Question 2.
holdout_idx <- !(raw_ids %in% dev_ids)
stopifnot(sum(holdout_idx) == 1607L)

# Apply the same county-to-region mapping used in Part 1.
region_for_county <- function(county) {
  groups <- list(
    "Bay Area" = c(
      "Alameda", "Contra Costa", "Marin", "Napa", "San Francisco",
      "San Mateo", "Santa Clara", "Solano", "Sonoma"
    ),
    "San Joaquin Valley" = c(
      "San Joaquin", "Stanislaus", "Merced", "Madera",
      "Fresno", "Kings", "Tulare", "Kern"
    ),
    "SoCal" = c(
      "Imperial", "Los Angeles", "Orange", "Riverside",
      "San Bernardino", "San Diego", "San Luis Obispo",
      "Santa Barbara", "Ventura"
    )
  )
  region <- rep("Other", length(county))
  for (group in names(groups)) {
    region[county %in% groups[[group]]] <- group
  }
  region
}

# Verify that the mapping reproduces the regions used in 3a.
stopifnot(all(
  region_for_county(raw$california_county[idx]) ==
    as.character(analysis$region)
))

holdout <- raw[holdout_idx, variables, drop = FALSE]
holdout$region <- factor(
  region_for_county(raw$california_county[holdout_idx]),
  levels = levels(analysis$region)
)

# Refit the same specification from scratch on complete holdout rows.
fit_holdout <- lm(spec, data = holdout, na.action = na.omit)
holdout_coefs <- coef(summary(fit_holdout))
holdout_ci <- confint(fit_holdout, level = 0.95)

# Align by coefficient name before comparing.
terms <- results$term
stopifnot(all(terms %in% rownames(holdout_coefs)))
holdout_p <- holdout_coefs[terms, "Pr(>|t|)"]

comparison <- data.frame(
  term = terms,
  development_estimate = results$estimate,
  development_ci_lower = results$ci_lower,
  development_ci_upper = results$ci_upper,
  development_p_value = results$p_value,
  development_significant = results$significant_at_5_pct,
  holdout_estimate = holdout_coefs[terms, "Estimate"],
  holdout_ci_lower = holdout_ci[terms, 1],
  holdout_ci_upper = holdout_ci[terms, 2],
  holdout_p_value = holdout_p,
  holdout_significant = holdout_p < alpha,
  row.names = NULL
)

cat("\nHoldout tracts used:", nobs(fit_holdout), "of", nrow(holdout), "\n")
print(comparison, digits = 4, row.names = FALSE)

write.csv(
  comparison,
  "part2/results/inference_development_vs_holdout.csv",
  row.names = FALSE
)
