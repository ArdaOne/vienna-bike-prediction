# ============================================================
# CATBOOST FINAL MODEL
# Vienna Bike Prediction
# Generates: submissions/submission_catboost.csv and submission.csv
# ============================================================

library(data.table)
library(lubridate)
library(catboost)

# ============================================================
# 0. PROJECT PATHS
# ============================================================

# This works whether you run from the repository root or from scripts/
current_dir <- getwd()

if (basename(current_dir) == "scripts") {
  project_dir <- normalizePath(file.path(current_dir, ".."))
} else {
  project_dir <- current_dir
}

data_dir <- file.path(project_dir, "data")
submissions_dir <- file.path(project_dir, "submissions")

if (!dir.exists(submissions_dir)) {
  dir.create(submissions_dir, recursive = TRUE)
}

train_path <- file.path(data_dir, "train.csv")
test_path  <- file.path(data_dir, "test.csv")

submission_path <- file.path(submissions_dir, "submission_catboost.csv")
submission_required_path <- file.path(project_dir, "submission.csv")

cat("Project directory:", project_dir, "\n")
cat("Train path:", train_path, "\n")
cat("Test path:", test_path, "\n")

if (!file.exists(train_path)) {
  stop("train.csv not found. Please place it in the data/ folder.")
}

if (!file.exists(test_path)) {
  stop("test.csv not found. Please place it in the data/ folder.")
}

rmsle_fun <- function(actual, pred) {
  pred <- pmax(pred, 0)
  sqrt(mean((log1p(pred) - log1p(actual))^2))
}

# ============================================================
# 1. READ DATA
# ============================================================

train <- fread(train_path)
test  <- fread(test_path)

train[, datetime := as.POSIXct(datetime, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")]
test[, datetime  := as.POSIXct(datetime,  format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")]

train[, is_train := 1]
test[, is_train := 0]
test[, bikes := NA_integer_]

all_data <- rbindlist(list(train, test), fill = TRUE)

cat("Train rows:", nrow(train), "\n")
cat("Test rows:", nrow(test), "\n")
cat("Train range:", as.character(min(train$datetime)), "to", as.character(max(train$datetime)), "\n")
cat("Test range:", as.character(min(test$datetime)), "to", as.character(max(test$datetime)), "\n")

# ============================================================
# 2. TIME FEATURES
# ============================================================

all_data[, `:=`(
  hour = hour(datetime),
  minute = minute(datetime),
  weekday = wday(datetime),
  day = day(datetime),
  month = month(datetime),
  week = isoweek(datetime)
)]

all_data[, `:=`(
  is_weekend = as.integer(weekday %in% c(1, 7)),
  hour_sin = sin(2 * pi * hour / 24),
  hour_cos = cos(2 * pi * hour / 24),
  weekday_sin = sin(2 * pi * weekday / 7),
  weekday_cos = cos(2 * pi * weekday / 7)
)]

# ============================================================
# 3. STATION-LEVEL HISTORICAL FEATURES
# ============================================================

train_base <- all_data[is_train == 1]

global_mean <- mean(train_base$bikes, na.rm = TRUE)

station_stats <- train_base[, .(
  station_mean_bikes = mean(bikes, na.rm = TRUE),
  station_max_bikes = max(bikes, na.rm = TRUE),
  station_sd_bikes = sd(bikes, na.rm = TRUE)
), by = station_number]

station_weekday_hour <- train_base[, .(
  avg_bikes_station_weekday_hour = mean(bikes, na.rm = TRUE)
), by = .(station_number, weekday, hour)]

all_data <- merge(
  all_data,
  station_stats,
  by = "station_number",
  all.x = TRUE,
  sort = FALSE
)

all_data <- merge(
  all_data,
  station_weekday_hour,
  by = c("station_number", "weekday", "hour"),
  all.x = TRUE,
  sort = FALSE
)

all_data[is.na(station_mean_bikes), station_mean_bikes := global_mean]
all_data[is.na(station_max_bikes), station_max_bikes := global_mean]
all_data[is.na(station_sd_bikes), station_sd_bikes := 0]
all_data[is.na(avg_bikes_station_weekday_hour), avg_bikes_station_weekday_hour := global_mean]

# ============================================================
# 4. MODEL FEATURES
# ============================================================

features <- c(
  "station_number",
  "lat",
  "lng",
  "hour",
  "minute",
  "weekday",
  "day",
  "month",
  "week",
  "is_weekend",
  "hour_sin",
  "hour_cos",
  "weekday_sin",
  "weekday_cos",
  "station_mean_bikes",
  "station_max_bikes",
  "station_sd_bikes",
  "avg_bikes_station_weekday_hour"
)

model_data <- as.data.table(all_data[, ..features])

cat_features <- c(
  "station_number",
  "hour",
  "weekday",
  "month",
  "week",
  "is_weekend"
)

for (col in cat_features) {
  model_data[[col]] <- as.factor(model_data[[col]])
}

for (col in names(model_data)) {
  if (!(col %in% cat_features)) {
    med_val <- median(model_data[[col]], na.rm = TRUE)
    if (is.na(med_val)) med_val <- 0
    model_data[[col]][is.na(model_data[[col]])] <- med_val
  }
}

train_rows <- which(all_data$is_train == 1)
test_rows  <- which(all_data$is_train == 0)

train_data <- as.data.frame(model_data[train_rows])
test_data  <- as.data.frame(model_data[test_rows])

train_final <- all_data[train_rows]

y_train <- log1p(train_final$bikes)

cat_features_idx <- which(names(train_data) %in% cat_features)

cat("\nCatBoost categorical feature names:\n")
print(names(train_data)[cat_features_idx])

cat("\nTrain data:", nrow(train_data), "x", ncol(train_data), "\n")
cat("Test data:", nrow(test_data), "x", ncol(test_data), "\n")

# ============================================================
# 5. VALIDATION
# ============================================================

cutoff_date <- as.POSIXct("2025-02-20 00:00:00", tz = "UTC")

tr_idx <- which(train_final$datetime < cutoff_date)
va_idx <- which(train_final$datetime >= cutoff_date)

cat("\nValidation train rows:", length(tr_idx), "\n")
cat("Validation rows:", length(va_idx), "\n")

train_pool <- catboost.load_pool(
  data = train_data[tr_idx, ],
  label = y_train[tr_idx],
  cat_features = cat_features_idx
)

valid_pool <- catboost.load_pool(
  data = train_data[va_idx, ],
  label = y_train[va_idx],
  cat_features = cat_features_idx
)

params <- list(
  loss_function = "RMSE",
  eval_metric = "RMSE",
  iterations = 2500,
  learning_rate = 0.035,
  depth = 6,
  l2_leaf_reg = 6,
  random_strength = 1,
  bagging_temperature = 0.5,
  od_type = "Iter",
  od_wait = 100,
  random_seed = 123,
  verbose = 100
)

set.seed(123)

valid_model <- catboost.train(
  learn_pool = train_pool,
  test_pool = valid_pool,
  params = params
)

valid_pred_log <- catboost.predict(valid_model, valid_pool)
valid_pred <- pmax(expm1(valid_pred_log), 0)

valid_rmsle <- rmsle_fun(train_final$bikes[va_idx], valid_pred)

cat("\nCatBoost validation RMSLE:", valid_rmsle, "\n")

best_iter <- valid_model$tree_count_

if (is.null(best_iter) || length(best_iter) == 0 || is.na(best_iter)) {
  best_iter <- params$iterations
}

cat("Best/tree count used:", best_iter, "\n")

# ============================================================
# 6. FINAL MODEL
# ============================================================

full_pool <- catboost.load_pool(
  data = train_data,
  label = y_train,
  cat_features = cat_features_idx
)

final_params <- params
final_params$iterations <- as.integer(best_iter)
final_params$od_type <- NULL
final_params$od_wait <- NULL

set.seed(123)

final_model <- catboost.train(
  learn_pool = full_pool,
  params = final_params
)

# ============================================================
# 7. PREDICT TEST
# ============================================================

test_pool <- catboost.load_pool(
  data = test_data,
  cat_features = cat_features_idx
)

test_pred_log <- catboost.predict(final_model, test_pool)
test_pred <- pmax(expm1(test_pred_log), 0)

cat("\nPrediction summary:\n")
print(summary(test_pred))
print(quantile(test_pred, probs = c(0, 0.01, 0.05, 0.5, 0.95, 0.99, 1)))

# ============================================================
# 8. SUBMISSION
# ============================================================

submission <- data.frame(
  id = paste(
    format(test$datetime, "%Y-%m-%d %H:%M:%S", tz = "UTC"),
    test$station_number,
    sep = "_"
  ),
  bikes = as.integer(round(pmax(test_pred, 0)))
)

cat("\nSubmission checks:\n")
cat("Rows:", nrow(submission), "\n")
cat("Expected rows:", nrow(test), "\n")
cat("Duplicate IDs:", sum(duplicated(submission$id)), "\n")
cat("Missing predictions:", sum(is.na(submission$bikes)), "\n")
cat("Negative predictions:", sum(submission$bikes < 0), "\n")

stopifnot(nrow(submission) == nrow(test))
stopifnot(sum(duplicated(submission$id)) == 0)
stopifnot(sum(is.na(submission$bikes)) == 0)
stopifnot(sum(submission$bikes < 0) == 0)

write.csv(submission, submission_path, row.names = FALSE)
write.csv(submission, submission_required_path, row.names = FALSE)

cat("\nSaved submissions to:\n")
cat(submission_path, "\n")
cat(submission_required_path, "\n")