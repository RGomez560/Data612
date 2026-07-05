Project 5: Implementing a Recommender System on Spark
================
Robert Gomez, DrPH, MPH
July 2026

- [Introduction](#introduction)
- [Recommender and Spark Context](#recommender-and-spark-context)
  - [Why Spark Changes the
    Implementation](#why-spark-changes-the-implementation)
  - [Algorithm Used](#algorithm-used)
  - [Comparison to Earlier Project 4](#comparison-to-earlier-project-4)
- [Setup](#setup)
- [Loading and Splitting the Food.com
  Data](#loading-and-splitting-the-foodcom-data)
  - [Utility Matrix Sparsity](#utility-matrix-sparsity)
- [Local Benchmark Models](#local-benchmark-models)
  - [Bias Baseline](#bias-baseline)
  - [Local Matrix Factorization with
    recosystem](#local-matrix-factorization-with-recosystem)
- [Spark Adaptation](#spark-adaptation)
  - [PySpark ALS Implementation](#pyspark-als-implementation)
- [Efficiency and Complexity
  Comparison](#efficiency-and-complexity-comparison)
- [When Spark Would Become
  Necessary](#when-spark-would-become-necessary)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Summary](#summary)
- [Sources](#sources)

# Introduction

In this project, I adapted the Food.com nutrition recipe recommender
from the earlier assignments to a Spark-oriented implementation. Project
4 focused on accuracy and beyond-accuracy tradeoffs for top-10
recommendation lists. Project 5 asked a different engineering question:
what changes when the recommender is moved toward a distributed platform
such as Apache Spark, and when is that added complexity worth it? The
recommender task stayed the same so the comparison was meaningful. The
system recommends recipes to Food.com users based on past user-recipe
ratings. The data form a sparse utility matrix: users are rows, recipes
are columns, and observed ratings are known entries. Most user-recipe
pairs are unknown, which is why a recommender needs to estimate missing
preferences rather than simply look up ratings. For this assignment, I
used the Food.com preprocessed interaction file because it already
includes numeric user and recipe identifiers. I then created a
deterministic 80/20 modeling split from that file and filtered the
validation rows to user-recipe identifiers that were present in
training. That made the comparison fair for Spark ALS, which cannot
score completely new items unless the model has item factors for them.
The project included:

- **A local benchmark:** a bias baseline and a recosystem matrix
  factorization model trained on the preprocessed Food.com training
  split.
- **A Spark adaptation:** PySpark ALS code that uses the same user,
  item, and rating fields in local Spark mode.
- **A comparison framework:** accuracy, runtime, implementation
  complexity, and scalability tradeoffs.
- **A conclusion about scale:** when this recommender would need Spark
  rather than local optimized packages.

This is still a course recommender prototype, not medical advice or
clinical decision support. Food.com ratings measure recipe preference,
not health value, safety, affordability, cultural fit, allergies, or
appropriateness for a person’s clinical goals.

# Recommender and Spark Context

## Why Spark Changes the Implementation

Spark matters when the data or computation no longer works well on one
machine. A recommender built from user-item interactions can become
complicated for several reasons:

- The utility matrix can have millions of users, millions of items, and
  billions of possible user-item pairs.
- Matrix factorization repeatedly updates user and item factor vectors,
  which can require large shuffles and repeated passes over the ratings.
- Production recommenders often need scheduled retraining, candidate
  generation, model evaluation, and batch scoring for many users.
- A single-machine workflow can become fragile when data loading, joins,
  feature engineering, or model training exceed memory.

## Algorithm Used

The Spark implementation uses alternating least squares, or ALS. ALS is
a matrix factorization method. It assumes that a user’s rating for a
recipe can be approximated by the dot product of a user latent-factor
vector and a recipe latent-factor vector, with regularization to reduce
overfitting.

## Comparison to Earlier Project 4

Project 4 used local R tools, especially recosystem, for matrix
factorization and then evaluated rating error and top-10
recommendation-list quality. Its rendered rating prediction results
were:

``` r
# Store the rendered Project 4 metrics so the comparison has a clear reference point.
project4_reference <- tibble::tribble(
  ~Model, ~Factors, ~RMSE, ~MAE,
  "Bias Baseline", NA_real_, 0.560, 0.345,
  "recosystem Matrix Factorization", 40, 0.572, 0.404,
  "recosystem Matrix Factorization", 80, 0.572, 0.402
)

# Print the Project 4 results as a compact comparison table.
project4_reference %>%
  mutate(across(where(is.numeric), ~ format(round(.x, 3), nsmall = 3))) %>%
  knitr::kable(caption = "Project 4 Rendered Rating Prediction Results")
```

| Model                           | Factors | RMSE  | MAE   |
|:--------------------------------|:--------|:------|:------|
| Bias Baseline                   | NA      | 0.560 | 0.345 |
| recosystem Matrix Factorization | 40.000  | 0.572 | 0.404 |
| recosystem Matrix Factorization | 80.000  | 0.572 | 0.402 |

Project 4 Rendered Rating Prediction Results

Those numbers came from the Project 4 custom filtered 80/20 split.
Project 5 uses a deterministic holdout from the Food.com preprocessed
training file so the Spark ALS code can use the same numeric identifiers
expected by MLlib while avoiding a pure cold-start item validation set.
Since the splits are different, I do not treat the raw RMSE values as a
direct apples-to-apples performance claim. The fair comparison in this
report is the same-split Project 5 benchmark, plus the conceptual
comparison with Project 4.

# Setup

``` r
# Format table numbers with commas and without scientific notation.
format_report_number <- function(x, digits = 3) {
  format(round(x, digits),
         big.mark = ",",
         scientific = FALSE,
         trim = TRUE,
         nsmall = digits)
}

# Keep predicted ratings inside the Food.com 1-5 rating range.
clip_rating <- function(x) {
  pmin(5, pmax(1, x))
}

# Calculate root mean squared error.
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

# Calculate mean absolute error.
mae <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}
```

# Loading and Splitting the Food.com Data

The Food.com preprocessed interaction file includes user_id, recipe_id,
rating, and zero-based numeric identifiers u and i. Those numeric
identifiers are useful because both recosystem and Spark ALS expect
integer user and item columns.

``` r
# Define possible data locations so the report can run from the project folder or desktop data folder.
desktop_data_dir <- "~/Desktop/Food.com Data"
project_data_dir <- "data/food_com"

# Find the preprocessed Food.com interaction file.
train_path <- case_when(
  file.exists(file.path(desktop_data_dir, "interactions_train.csv")) ~
    file.path(desktop_data_dir, "interactions_train.csv"),
  file.exists(file.path(project_data_dir, "interactions_train.csv")) ~
    file.path(project_data_dir, "interactions_train.csv"),
  TRUE ~ NA_character_
)

# Stop the report early if the required interaction file is missing.
if (is.na(train_path)) {
  stop("interactions_train.csv is required for this project.")
}

# Load the interaction data and keep the numeric user and recipe IDs used by the recommender packages.
full_interaction_data <- read_csv(train_path,
                                  show_col_types = FALSE,
                                  col_select = c(user_id, recipe_id, date, rating, u, i)) %>%
  mutate(row_id = row_number(),
         user_index = as.integer(u),
         item_index = as.integer(i),
         rating = as.numeric(rating))

# Create a deterministic 80/20 split using row position.
training_data <- full_interaction_data %>%
  filter(row_id %% 5 != 0)

validation_initial <- full_interaction_data %>%
  filter(row_id %% 5 == 0)

# Keep validation rows that Spark ALS can score because the user and item exist in training.
validation_data <- validation_initial %>%
  filter(user_index %in% training_data$user_index,
         item_index %in% training_data$item_index)

# Write the same modeling split to CSV so the PySpark chunk uses identical rows.
spark_split_dir <- "/private/tmp/data612_project5_spark"
dir.create(spark_split_dir, showWarnings = FALSE, recursive = TRUE)

training_data %>%
  transmute(user_index = as.integer(user_index),
            item_index = as.integer(item_index),
            rating = as.numeric(rating)) %>%
  write_csv(file.path(spark_split_dir, "spark_train.csv"))

validation_data %>%
  transmute(user_index = as.integer(user_index),
            item_index = as.integer(item_index),
            rating = as.numeric(rating)) %>%
  write_csv(file.path(spark_split_dir, "spark_validation.csv"))

# Reconfirm numeric index and rating types for the local R models.
training_data <- training_data %>%
  mutate(user_index = as.integer(u),
         item_index = as.integer(i),
         rating = as.numeric(rating))

validation_data <- validation_data %>%
  mutate(user_index = as.integer(u),
         item_index = as.integer(i),
         rating = as.numeric(rating))

display_train_path <- sub(normalizePath("~"), "~", train_path, fixed = TRUE)
training_user_count <- n_distinct(training_data$user_id)
training_recipe_count <- n_distinct(training_data$recipe_id)
full_possible_pairs <- as.numeric(training_user_count) *
  as.numeric(training_recipe_count)

# Print the data split summary.
tibble(Metric = c("Interaction source",
                  "Original interaction rows",
                  "Model training ratings",
                  "Initial holdout ratings",
                  "Known-user/known-item validation ratings",
                  "Training users",
                  "Training recipes",
                  "Full possible user-recipe pairs",
                  "Average training rating"),
       Value = c(display_train_path,
                 format_report_number(nrow(full_interaction_data), 0),
                 format_report_number(nrow(training_data), 0),
                 format_report_number(nrow(validation_initial), 0),
                 format_report_number(nrow(validation_data), 0),
                 format_report_number(training_user_count, 0),
                 format_report_number(training_recipe_count, 0),
                 format_report_number(full_possible_pairs, 0),
                 format_report_number(mean(training_data$rating), 3))) %>%
  kable(caption = "Food.com Modeling Split Summary")
```

| Metric | Value |
|:---|:---|
| Interaction source | ~/Desktop/Food.com Data/interactions_train.csv |
| Original interaction rows | 698,901 |
| Model training ratings | 559,121 |
| Initial holdout ratings | 139,780 |
| Known-user/known-item validation ratings | 123,143 |
| Training users | 24,875 |
| Training recipes | 146,257 |
| Full possible user-recipe pairs | 3,638,142,875 |
| Average training rating | 4.574 |

Food.com Modeling Split Summary

Compared with Project 4, this version uses a larger Food.com interaction
sample. The goal was to make the Spark implementation meaningful while
still keeping the run practical on a single machine.

``` r
# Compare the Project 4 modeling subset with the larger Project 5 Spark-oriented subset.
project_size_comparison <- tibble::tribble(
  ~Metric, ~Project_4, ~Project_5,
  "Users", 7795, n_distinct(training_data$user_id),
  "Recipes", 8000, n_distinct(training_data$recipe_id),
  "Observed/modeling ratings", 253120, nrow(full_interaction_data),
  "Training ratings", 202496, nrow(training_data),
  "Test or validation ratings", 50624, nrow(validation_data)
)

# Add the percent change from Project 4 to Project 5.
project_size_comparison <- project_size_comparison %>%
  mutate(`Project 5 Increase` = (Project_5 - Project_4) / Project_4)

# Print the size comparison with formatted counts.
project_size_comparison %>%
  transmute(Metric,
            `Project 4` = format_report_number(Project_4, 0),
            `Project 5` = format_report_number(Project_5, 0),
            `Project 5 Increase` = str_c(format_report_number(`Project 5 Increase` * 100,
                                                              1),
                                         "%")) %>%
  kable(caption = "Project 4 and Project 5 Modeling Size Comparison")
```

| Metric                     | Project 4 | Project 5 | Project 5 Increase |
|:---------------------------|:----------|:----------|:-------------------|
| Users                      | 7,795     | 24,875    | 219.1%             |
| Recipes                    | 8,000     | 146,257   | 1,728.2%           |
| Observed/modeling ratings  | 253,120   | 698,901   | 176.1%             |
| Training ratings           | 202,496   | 559,121   | 176.1%             |
| Test or validation ratings | 50,624    | 123,143   | 143.3%             |

Project 4 and Project 5 Modeling Size Comparison

## Utility Matrix Sparsity

The modeling split is still very sparse. Missing entries mean unknown
preferences, not dislikes. This matters because a model that treats
every missing user-recipe pair as negative feedback would distort the
task.

``` r
# Calculate the number of possible user-recipe pairs in the training matrix.
possible_pairs <- full_possible_pairs

# Calculate how much of the matrix is observed.
observed_share <- nrow(training_data) / possible_pairs

# Print the sparsity summary.
tibble(Metric = c("Possible user-recipe pairs",
                  "Possible training matrix entries",
                  "Observed training ratings",
                  "Observed share",
                  "Sparsity"),
       Value = c(format_report_number(possible_pairs, 0),
                 format_report_number(possible_pairs, 0),
                 format_report_number(nrow(training_data), 0),
                 format_report_number(observed_share, 6),
                 format_report_number(1 - observed_share, 6))) %>%
  kable(caption = "Training Utility Matrix Sparsity")
```

| Metric                           | Value         |
|:---------------------------------|:--------------|
| Possible user-recipe pairs       | 3,638,142,875 |
| Possible training matrix entries | 3,638,142,875 |
| Observed training ratings        | 559,121       |
| Observed share                   | 0.000154      |
| Sparsity                         | 0.999846      |

Training Utility Matrix Sparsity

# Local Benchmark Models

## Bias Baseline

The bias baseline predicts validation ratings from the training global
average plus user and recipe average-rating effects. This is not
distributed, but it is a useful benchmark because average user and item
behavior can explain a surprising amount of rating variation.

``` r
# Calculate the overall average rating in the training data.
global_mean <- mean(training_data$rating, na.rm = TRUE)

# Estimate each user's average deviation from the global mean.
user_bias <- training_data %>%
  group_by(user_id) %>%
  summarize(user_bias = mean(rating - global_mean, na.rm = TRUE),
            .groups = "drop")

# Estimate each recipe's average deviation from the global mean.
item_bias <- training_data %>%
  group_by(recipe_id) %>%
  summarize(item_bias = mean(rating - global_mean, na.rm = TRUE),
            .groups = "drop")

# Time the baseline prediction step.
baseline_start <- Sys.time()

# Predict validation ratings with the global mean plus user and recipe effects.
baseline_predictions <- validation_data %>%
  left_join(user_bias, by = "user_id") %>%
  left_join(item_bias, by = "recipe_id") %>%
  mutate(user_bias = replace_na(user_bias, 0),
         item_bias = replace_na(item_bias, 0),
         prediction = clip_rating(global_mean + user_bias + item_bias))

baseline_elapsed <- as.numeric(difftime(Sys.time(), baseline_start, units = "secs"))

# Store the baseline accuracy and runtime metrics.
baseline_results <- tibble(Model = "Local bias baseline",
                           Factors = NA_real_,
                           RMSE = rmse(baseline_predictions$rating,
                                       baseline_predictions$prediction),
                           MAE = mae(baseline_predictions$rating,
                                     baseline_predictions$prediction),
                           Runtime_Seconds = baseline_elapsed)

# Print the baseline results.
baseline_results %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 3))) %>%
  kable(caption = "Bias Baseline Validation Results")
```

| Model               | Factors | RMSE  | MAE   | Runtime_Seconds |
|:--------------------|:--------|:------|:------|:----------------|
| Local bias baseline | NA      | 0.993 | 0.558 | 0.028           |

Bias Baseline Validation Results

## Local Matrix Factorization with recosystem

The local matrix factorization benchmark uses recosystem, which is
optimized for sparse explicit-feedback recommendation. I trained two
small factor counts to keep the comparison practical on a laptop. The
purpose was not to exhaustively tune the model; it was to create a local
benchmark against the Spark ALS design.

``` r
# Store the training ratings in the sparse triple format expected by recosystem.
train_reco_data <- data_memory(user_index = training_data$user_index,
                               item_index = training_data$item_index,
                               rating = training_data$rating,
                               index1 = FALSE)

# Store the validation ratings in the same sparse triple format.
validation_reco_data <- data_memory(user_index = validation_data$user_index,
                                    item_index = validation_data$item_index,
                                    rating = validation_data$rating,
                                    index1 = FALSE)

# Train one recosystem model and return its validation metrics.
fit_recosystem_model <- function(factor_count) {
  set.seed(612)
  model <- Reco()

  # Train a regularized matrix factorization model.
  train_time <- system.time(
    model$train(train_reco_data,
                opts = list(dim = factor_count,
                            costp_l2 = 0.1,
                            costq_l2 = 0.1,
                            lrate = 0.05,
                            niter = 10,
                            nthread = 2,
                verbose = FALSE))
  )

  # Predict all validation ratings in one vectorized step.
  predictions <- clip_rating(model$predict(validation_reco_data, out_memory()))

  # Return model accuracy and elapsed training time.
  tibble(Model = "Local recosystem matrix factorization",
         Factors = factor_count,
         RMSE = rmse(validation_data$rating, predictions),
         MAE = mae(validation_data$rating, predictions),
         Runtime_Seconds = unname(train_time["elapsed"]))
}

# Fit two factor sizes for a small local comparison.
recosystem_results <- map_dfr(c(20, 40), fit_recosystem_model)

# Combine the baseline and matrix factorization results.
local_model_results <- bind_rows(baseline_results, recosystem_results)

# Print the local model comparison.
local_model_results %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 3))) %>%
  kable(caption = "Local Validation Results on the Project 5 Split")
```

| Model                                 | Factors | RMSE  | MAE   | Runtime_Seconds |
|:--------------------------------------|:--------|:------|:------|:----------------|
| Local bias baseline                   | NA      | 0.993 | 0.558 | 0.028           |
| Local recosystem matrix factorization | 20.000  | 1.067 | 0.701 | 0.224           |
| Local recosystem matrix factorization | 40.000  | 1.050 | 0.685 | 0.353           |

Local Validation Results on the Project 5 Split

``` r
# Plot RMSE and MAE for the local models.
local_model_results %>%
  mutate(Model_Label = if_else(is.na(Factors),
                               Model,
                               str_c(Model, " (", Factors, " factors)")),
         Model_Label = factor(Model_Label, levels = Model_Label)) %>%
  pivot_longer(cols = c(RMSE, MAE),
               names_to = "Metric",
               values_to = "Error") %>%
  ggplot(aes(x = Model_Label, y = Error, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  labs(title = "Local Food.com Validation Error",
       subtitle = "Lower is better",
       x = NULL,
       y = "Prediction Error",
       fill = "Metric") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
```

![](Project-5--Implementing-a-Recommender-System-on-Spark_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

On this split, the bias baseline was hard to beat. That does not mean
matrix factorization is useless. It means that, for explicit recipe
ratings in this validation slice, average user and recipe tendencies
were strong. A stronger tuning pass would use a separate development
split, search regularization and factor counts more carefully, and
reserve the validation or test split for final reporting.

# Spark Adaptation

## PySpark ALS Implementation

The code below is the Spark version run in local mode. It keeps the data
in Spark DataFrames, casts the identifiers to integer columns, trains
ALS on the training split, and evaluates RMSE and MAE on the validation
split. The key point is that the algorithm is package-supported and
distributed by Spark rather than implemented with manual loops.

``` python
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Start a local Spark session for the ALS model.
spark = (
    SparkSession.builder
    .appName("data612-project5-foodcom-als")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

# Keep the Spark output focused on warnings and errors.
spark.sparkContext.setLogLevel("WARN")

# Use the same split files that were created in the R data preparation chunk.
train_path = "/private/tmp/data612_project5_spark/spark_train.csv"
validation_path = "/private/tmp/data612_project5_spark/spark_validation.csv"

# Load the training ratings as a Spark DataFrame.
train = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(train_path)
    .select(
        F.col("user_index").cast("int").alias("user_index"),
        F.col("item_index").cast("int").alias("item_index"),
        F.col("rating").cast("double").alias("rating"),
    )
    .where(F.col("rating").isNotNull())
    .cache()
)

# Load the validation ratings as a Spark DataFrame.
validation = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(validation_path)
    .select(
        F.col("user_index").cast("int").alias("user_index"),
        F.col("item_index").cast("int").alias("item_index"),
        F.col("rating").cast("double").alias("rating"),
    )
    .where(F.col("rating").isNotNull())
    .cache()
)

# Count the rows used by Spark so the rendered output documents the model input.
spark_train_rows = train.count()
spark_validation_rows = validation.count()

# Define the explicit-feedback ALS model.
als = ALS(
    userCol="user_index",
    itemCol="item_index",
    ratingCol="rating",
    rank=20,
    maxIter=5,
    regParam=0.10,
    coldStartStrategy="drop",
    nonnegative=False,
    implicitPrefs=False,
    seed=612,
)

# Train the Spark ALS model and record elapsed time.
spark_start = time.time()
model = als.fit(train)
spark_runtime_seconds = time.time() - spark_start

# Score the validation ratings.
predictions = model.transform(validation).cache()
spark_prediction_rows = predictions.count()

# Evaluate Spark ALS with RMSE.
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction",
)

spark_rmse = evaluator.evaluate(predictions)

# Evaluate Spark ALS with MAE.
mae_evaluator = RegressionEvaluator(
    metricName="mae",
    labelCol="rating",
    predictionCol="prediction",
)

spark_mae = mae_evaluator.evaluate(predictions)

# Print the Spark settings and validation results.
print({
    "spark_als_rank": 20,
    "spark_als_max_iter": 5,
    "spark_als_reg_param": 0.10,
    "train_rows": spark_train_rows,
    "validation_rows": spark_validation_rows,
    "prediction_rows": spark_prediction_rows,
    "runtime_seconds": round(spark_runtime_seconds, 3),
    "rmse": round(spark_rmse, 6),
    "mae": round(spark_mae, 6),
})
```

    ## {'spark_als_rank': 20, 'spark_als_max_iter': 5, 'spark_als_reg_param': 0.1, 'train_rows': 559121, 'validation_rows': 123143, 'prediction_rows': 123143, 'runtime_seconds': 5.889, 'rmse': 1.448613, 'mae': 1.075714}

``` python

# Stop the Spark session after the model has been evaluated.
spark.stop()
```

The Spark ALS results were added to the same table as the local models:

``` r
# Store the Spark ALS metrics returned by the Python chunk.
spark_results <- tibble(Model = "Spark MLlib ALS",
                        Factors = 20,
                        RMSE = py$spark_rmse,
                        MAE = py$spark_mae,
                        Runtime_Seconds = py$spark_runtime_seconds,
                        Validation_Predictions = py$spark_prediction_rows)

# Add the Spark results to the local model comparison.
local_model_results %>%
  mutate(Validation_Predictions = nrow(validation_data)) %>%
  bind_rows(spark_results) %>%
  mutate(Factors = if_else(is.na(Factors),
                           "NA",
                           format_report_number(Factors, 3)),
         RMSE = if_else(is.na(RMSE),
                        "Not run",
                        format_report_number(RMSE, 3)),
         MAE = if_else(is.na(MAE),
                       "Not run",
                       format_report_number(MAE, 3)),
         Runtime_Seconds = if_else(is.na(Runtime_Seconds),
                                   "Not run",
                                   format_report_number(Runtime_Seconds, 3)),
         Validation_Predictions = format_report_number(Validation_Predictions, 0)) %>%
  kable(caption = "Project 5 Same-Split Local and Spark ALS Results")
```

| Model | Factors | RMSE | MAE | Runtime_Seconds | Validation_Predictions |
|:---|:---|:---|:---|:---|:---|
| Local bias baseline | NA | 0.993 | 0.558 | 0.028 | 123,143 |
| Local recosystem matrix factorization | 20.000 | 1.067 | 0.701 | 0.224 | 123,143 |
| Local recosystem matrix factorization | 40.000 | 1.050 | 0.685 | 0.353 | 123,143 |
| Spark MLlib ALS | 20.000 | 1.449 | 1.076 | 5.889 | 123,143 |

Project 5 Same-Split Local and Spark ALS Results

# Efficiency and Complexity Comparison

Based on the results, the local implementation was simpler and faster to
develop. It ran directly in R, used a local CSV reader, and trained the
local matrix factorization model quickly on this subset. Spark added
dependency management, a Spark session, partitioning choices, and Spark
UI or log monitoring. The extra steps does not mean it is better or
worst but it appears that Spark becomes valuable when the project needs
distributed data loading, distributed joins, repeated model training
over larger rating histories, or batch recommendation generation for
many users. For this specific local Food.com project, the local
optimized tools were enough. The training split had hundreds of
thousands of ratings, which is large but not large enough to require a
cluster on this machine. The first Spark ALS run also showed that Spark
is a scalability tool, not a guarantee of better accuracy. Without
tuning, ALS performed worse than the simple bias baseline on this split.

``` r
# Summarize the practical tradeoffs between the local and Spark implementations.
complexity_comparison <- tibble::tribble(
  ~Dimension, ~Local_R_Approach, ~Spark_ALS_Approach,
  "Setup", "R packages only", "Requires a Spark runtime and PySpark or SparkR/sparklyr",
  "Data structure", "Local data frames and sparse triples", "Distributed Spark DataFrames",
  "Algorithm", "recosystem matrix factorization", "Spark MLlib ALS",
  "Best use", "Laptop-scale experimentation and reporting", "Large-scale training and batch scoring",
  "Main risk", "Memory or runtime limits on larger data", "More operational complexity and harder debugging",
  "Evaluation", "Direct local RMSE and MAE", "RMSE and MAE after Spark prediction DataFrame is materialized"
)

# Print the implementation comparison.
complexity_comparison %>%
  kable(caption = "Local versus Spark Implementation Tradeoffs")
```

| Dimension | Local_R_Approach | Spark_ALS_Approach |
|:---|:---|:---|
| Setup | R packages only | Requires a Spark runtime and PySpark or SparkR/sparklyr |
| Data structure | Local data frames and sparse triples | Distributed Spark DataFrames |
| Algorithm | recosystem matrix factorization | Spark MLlib ALS |
| Best use | Laptop-scale experimentation and reporting | Large-scale training and batch scoring |
| Main risk | Memory or runtime limits on larger data | More operational complexity and harder debugging |
| Evaluation | Direct local RMSE and MAE | RMSE and MAE after Spark prediction DataFrame is materialized |

Local versus Spark Implementation Tradeoffs

# When Spark Would Become Necessary

For this recommender, I would move from local R packages to Spark when
at least one of these conditions is true:

- The interaction table grows from hundreds of thousands of ratings to
  tens or hundreds of millions of interactions.
- The model needs to score recommendations for many users on a schedule,
  not just evaluate a sample in a course report.
- Feature engineering requires large joins across recipes, ingredients,
  reviews, user histories, and event logs.
- The user-item matrix or intermediate factorization artifacts no longer
  fit comfortably in memory.
- The team needs production-style monitoring of distributed jobs,
  partitions, shuffles, and scheduled retraining.

In short, Spark becomes necessary when scale changes the bottleneck from
modeling logic to data movement, memory, and repeated computation.

# Limitations and Next Steps

- **The split differs from Project 4:** Project 4 used a custom filtered
  80/20 split. Project 5 uses a deterministic holdout from Food.com’s
  preprocessed interaction training file. The metrics should be
  interpreted within each split rather than treated as a direct
  winner-take-all comparison.

- **The original validation file was a cold-start item split:** Every
  recipe in the provided validation file was absent from the training
  file, so Spark ALS with coldStartStrategy = “drop” could not score
  those rows. For this assignment, I used a known-user/known-item
  holdout so ALS could be evaluated honestly. A separate cold-start
  project would need item metadata or a hybrid model.

- **Rating prediction is not the whole recommender:** RMSE and MAE
  evaluate held-out rating prediction. They do not prove that the top-10
  list is diverse, novel, feasible, safe, affordable, or useful.

- **Cold start remains unresolved:** ALS and local matrix factorization
  both depend on interaction history. New users and new recipes would
  need content features, onboarding questions, popularity priors, or a
  hybrid model.

- **Health and nutrition constraints are missing:** Recipe ratings are
  not clinical outcomes. A deployable nutrition recommender would need
  safeguards for allergies, medical conditions, sodium or sugar
  constraints, budget, kitchen access, cultural fit, and user control.

# Summary

Project 5 moved the Food.com recipe recommender toward a distributed
Spark implementation. The local benchmark showed that a simple bias
baseline and recosystem matrix factorization can be run efficiently on
the available Food.com split. The Spark adaptation used the appropriate
distributed recommender tool, Spark MLlib ALS, kept the data in Spark
DataFrames, and produced validation predictions in local Spark mode. The
main practical finding was that Spark adds real operational complexity:
it requires Spark configuration, distributed data structures, and
attention to lazy execution. For this data size, local optimized
packages are still the more practical choice, and the untuned Spark ALS
model did not beat the simple bias baseline. Spark becomes necessary
when interaction volume, feature joins, retraining schedules, and batch
scoring needs outgrow a single-machine workflow.

# Sources

Food.com Recipes and Interactions. Kaggle dataset by Shuyang Li.
<https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions>

Apache Spark MLlib Collaborative Filtering Documentation.
<https://spark.apache.org/docs/latest/ml-collaborative-filtering.html>

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization
techniques for recommender systems. *Computer*, 42(8), 30-37.
<https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf>

recosystem: Matrix Factorization for Recommender Systems.
<https://cran.r-project.org/package=recosystem>
