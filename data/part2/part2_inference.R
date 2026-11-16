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
