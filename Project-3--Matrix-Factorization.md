Project 3: Matrix Factorization for a Nutrition Recipe Recommender
================
Robert Gomez, DrPH, MPH
June 2026

- [Introduction](#introduction)
- [Approach and Recommender Context](#approach-and-recommender-context)
  - [Matrix Factorization](#matrix-factorization)
  - [Scalability](#scalability)
  - [Evaluation](#evaluation)
- [Setup](#setup)
- [Loading Recipe and Rating Data](#loading-recipe-and-rating-data)
- [Data Preparation](#data-preparation)
  - [Larger Sparse Rating Subset](#larger-sparse-rating-subset)
  - [80/20 Train-Test Split](#8020-train-test-split)
  - [Utility Matrix](#utility-matrix)
- [Recommendation Models](#recommendation-models)
  - [Baseline Models](#baseline-models)
  - [Matrix Factorization with
    recosystem](#matrix-factorization-with-recosystem)
- [Evaluation](#evaluation-1)
  - [Rating Prediction Error](#rating-prediction-error)
  - [Example Predictions](#example-predictions)
- [Recommendation Example](#recommendation-example)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Summary](#summary)
- [Sources](#sources)

# Introduction

In this assignment, I extend the Food.com nutrition recipe recommender
from Project 2 by implementing a matrix factorization method. The goal
is still to recommend recipes to users, but the main technique changes
from neighborhood-based similarity to a latent factor model.One thing I
wanted to improve from the earlier project is scalability. A recommender
system should not only work on a small example. It should also be
designed in a way that can grow. Project 2 started with a sparse rating
matrix, but later parts of the workflow converted data into dense
formats and predicted ratings one user-recipe pair at a time. That
approach is understandable for learning the concept, but it does not
scale well when the number of users, recipes, and ratings increases. In
Project 3, I kept the same general nutrition recommendation context but
changed the implementation to use recommender-system packages that are
built for sparse data. The main goal is to practice matrix factorization
while avoiding the bottlenecks that come from dense matrices and
repeated one-row prediction steps. This is also a test of whether a more
advanced model actually improves the recommender. Matrix factorization
is more scalable and can learn hidden preference patterns, but it still
needs to be compared with simple baselines. A more complex model is not
automatically better if user averages and recipe averages already
explain much of the rating behavior.

This project uses:

- **recommenderlab:** to represent the sparse user-recipe utility matrix
  as a real rating matrix.
- **recosystem:** to train regularized matrix factorization models
  directly from sparse user-recipe-rating triples.
- **An 80/20 train-test split:** so the model is evaluated on held-out
  ratings.
- **A much larger sample:** 7,795 users, 8,000 recipes, and 253,120
  observed ratings before the split.

Since the recommender is about food, I also do not want to treat
prediction accuracy as the only thing that matters. A recipe can receive
a high predicted rating and still be a poor recommendation for someone
with allergies, hypertension, diabetes, limited food access, or limited
cooking time. This project is a recommender systems assignment, not
medical advice, but the health context still matters.

# Approach and Recommender Context

## Matrix Factorization

- Matrix factorization learns hidden patterns in a user-item rating
  matrix. Instead of comparing recipes directly, it learns a smaller set
  of latent factors for users and recipes, which are combined to
  estimate how a user might rate a recipe.

- A key idea is that matrix factorization is a form of compression.
  Rather than filling every missing value in a sparse utility matrix, it
  captures the main preference structure using a much smaller number of
  latent factors. These factors may represent patterns such as quick
  meals, sweet recipes, or high-protein foods, although they are not
  always easy to interpret.

- I considered using Singular Value Decomposition (SVD) and Alternating
  Least Squares (ALS), but neither was the best fit. Traditional SVD
  assumes a complete matrix, while the Food.com utility matrix is highly
  sparse. Instead, I used recosystem, which trains directly on the
  observed ratings, preserves sparsity, and uses regularization to
  reduce overfitting, consistent with the approach described by Koren,
  Bell, and Volinsky.

## Scalability

- The scalability issue is not only whether the code produces
  recommendations, but whether the design would still make sense as the
  dataset grows. Dense matrices become expensive because they store
  every possible user-recipe combination, including all the missing
  ones. One-user-one-item prediction loops also become slow because they
  repeat the same work many times.

- In this version, I avoid that pattern. I keep the rating data sparse,
  use package-based matrix factorization, and predict the test set as a
  vector instead of repeatedly calling a custom prediction function for
  each row. This is a closer match to how a recommender should be built
  when the dataset is large.

## Evaluation

- The project evaluates rating prediction with RMSE and MAE. RMSE gives
  more weight to large mistakes, while MAE is easier to interpret as the
  average number of rating points the model is off. I compare the matrix
  factorization models with two baselines:

  - **Global mean baseline:** predicts the average training rating for
    every user and recipe.

  - **Bias baseline:** adjusts for users who usually rate high or low
    and recipes that usually receive high or low ratings.

- The baselines are important because a more complex model should not be
  assumed to be better just because it is more advanced.

# Setup

``` r
# Load the packages used for data preparation, sparse matrices, modeling, and output.
pacman::p_load(tidyverse,
               knitr,
               Matrix,
               ggplot2,
               recommenderlab,
               recosystem)

# Format table numbers with commas and without scientific notation.
format_report_number <- function(x, digits = 2) {
  format(round(x, digits),
         big.mark = ",",
         scientific = FALSE,
         trim = TRUE,
         nsmall = digits)
}
```

# Loading Recipe and Rating Data

The data comes from the Food.com recipes and interactions dataset on
Kaggle. I use the same dataset as Project 2 so the main change is the
recommender method, not the recommendation problem.

``` r
# Define possible data locations so the file can run from the project folder or desktop data folder.
project_data_dir <- "data/food_com"
desktop_data_dir <- "~/Desktop/Food.com Data"

# Find the recipe file.
raw_recipes_path <- case_when(
  file.exists(file.path(desktop_data_dir, "RAW_recipes.csv")) ~
    file.path(desktop_data_dir, "RAW_recipes.csv"),
  file.exists(file.path(project_data_dir, "RAW_recipes.csv")) ~
    file.path(project_data_dir, "RAW_recipes.csv"),
  TRUE ~ NA_character_
)

# Find the interaction file.
raw_interactions_path <- case_when(
  file.exists(file.path(desktop_data_dir, "RAW_interactions.csv")) ~
    file.path(desktop_data_dir, "RAW_interactions.csv"),
  file.exists(file.path(project_data_dir, "RAW_interactions.csv")) ~
    file.path(project_data_dir, "RAW_interactions.csv"),
  TRUE ~ NA_character_
)

# Stop the report early if either required file is missing.
if (is.na(raw_recipes_path) || is.na(raw_interactions_path)) {
  stop("RAW_recipes.csv and RAW_interactions.csv are required for this project.")
}

# Shorten displayed paths so the table does not expose the full local username path.
display_recipes_path <- sub(normalizePath("~"), "~", raw_recipes_path, fixed = TRUE)
display_interactions_path <- sub(normalizePath("~"), "~", raw_interactions_path, fixed = TRUE)

# Load only the recipe fields needed for this project.
recipes_raw <- read_csv(raw_recipes_path,
                        show_col_types = FALSE,
                        col_select = c(name, id, minutes, nutrition))

# Food.com stores nutrition as a list-like field, so I split it into separate columns.
nutrition_values <- recipes_raw %>%
  mutate(nutrition_clean = str_remove_all(nutrition, "\\[|\\]")) %>%
  separate_wider_delim(nutrition_clean,
                       delim = ",",
                       names = c("calories", "fat", "sugar", "sodium",
                                 "protein", "saturated_fat", "carbohydrate"),
                       too_few = "align_start",
                       too_many = "drop") %>%
  mutate(across(c(calories, fat, sugar, sodium, protein,
                  saturated_fat, carbohydrate), as.numeric))

# Keep recipe identifiers, names, cooking time, and nutrition variables.
recipes <- nutrition_values %>%
  transmute(recipe_id = as.character(id),
            recipe_name = name,
            minutes = as.numeric(minutes),
            calories,
            fat,
            saturated_fat,
            sodium,
            carbohydrate,
            sugar,
            protein) %>%
  filter(!is.na(recipe_id),
         !is.na(recipe_name))

# Load positive user-recipe ratings and keep only recipes represented in the recipe table.
interactions <- read_csv(raw_interactions_path,
                         show_col_types = FALSE,
                         col_select = c(user_id, recipe_id, rating)) %>%
  transmute(user = as.character(user_id),
            recipe_id = as.character(recipe_id),
            rating = as.numeric(rating)) %>%
  filter(!is.na(user),
         !is.na(recipe_id),
         !is.na(rating),
         rating > 0) %>%
  semi_join(recipes, by = "recipe_id")

# Print the data source summary.
tibble(Metric = c("Recipe data source",
                  "Interaction source",
                  "Recipes in raw recipe data",
                  "Observed positive ratings"),
       Value = c(display_recipes_path,
                 display_interactions_path,
                 nrow(recipes),
                 nrow(interactions))) %>%
  kable(caption = "Food.com Data Sources")
```

| Metric                     | Value                                        |
|:---------------------------|:---------------------------------------------|
| Recipe data source         | ~/Desktop/Food.com Data/RAW_recipes.csv      |
| Interaction source         | ~/Desktop/Food.com Data/RAW_interactions.csv |
| Recipes in raw recipe data | 231636                                       |
| Observed positive ratings  | 1071520                                      |

Food.com Data Sources

# Data Preparation

## Larger Sparse Rating Subset

- Compared with Project 2, this project uses a much larger rating subset
  and a more scalable modeling approach. Project 2 used 200 users, 250
  recipes, and 6,199 observed ratings. Project 3 uses 7,795 users, 8,000
  recipes, and 253,120 observed ratings.

- Project 2 had 5,999 training ratings and a training matrix sparsity of
  0.88. Project 3 has 202,496 training ratings and a training matrix
  sparsity of 0.9968. This means Project 3 uses much more rating data,
  but the possible user-recipe matrix is also much larger and more
  sparse. Even with the larger Project 3 sample, the matrix is still
  extremely sparse.

``` r
# Set a seed so the same users, recipes, and train-test split are selected each run.
set.seed(612)

# Count how many ratings each user has.
interaction_counts <- interactions %>%
  count(user, name = "user_rating_count")

# Count how many ratings each recipe has.
recipe_counts <- interactions %>%
  count(recipe_id, name = "recipe_rating_count")

# Keep users and recipes with enough rating history to support matrix factorization.
dense_interactions <- interactions %>%
  left_join(interaction_counts, by = "user") %>%
  left_join(recipe_counts, by = "recipe_id") %>%
  filter(user_rating_count >= 5,
         recipe_rating_count >= 5)

# Select the most active users.
selected_users <- dense_interactions %>%
  count(user, sort = TRUE) %>%
  slice_head(n = 8000) %>%
  pull(user)

# From those users, select the most commonly rated recipes.
selected_recipes <- dense_interactions %>%
  filter(user %in% selected_users) %>%
  count(recipe_id, sort = TRUE) %>%
  slice_head(n = 8000) %>%
  pull(recipe_id)

# Build the final rating table used for modeling.
ratings_long <- dense_interactions %>%
  filter(user %in% selected_users,
         recipe_id %in% selected_recipes) %>%
  distinct(user, recipe_id, .keep_all = TRUE) %>%
  inner_join(recipes, by = "recipe_id") %>%
  group_by(user) %>%
  filter(n() >= 5) %>%
  ungroup() %>%
  select(user, recipe_id, recipe_name, rating,
         calories, fat, sodium, carbohydrate, sugar, protein,
         saturated_fat, minutes)

# Print the larger subset summary.
tibble(Metric = c("Users",
                  "Recipes",
                  "Observed ratings",
                  "Average rating",
                  "Full possible user-recipe pairs",
                  "Observed share before split"),
       Value = c(format_report_number(n_distinct(ratings_long$user), 0),
                 format_report_number(n_distinct(ratings_long$recipe_id), 0),
                 format_report_number(nrow(ratings_long), 0),
                 format_report_number(mean(ratings_long$rating), 2),
                 format_report_number(n_distinct(ratings_long$user) *
                                        n_distinct(ratings_long$recipe_id), 0),
                 format_report_number(nrow(ratings_long) /
                                        (n_distinct(ratings_long$user) *
                                           n_distinct(ratings_long$recipe_id)), 4))) %>%
  kable(caption = "Larger Sparse Rating Subset Summary")
```

| Metric                          | Value      |
|:--------------------------------|:-----------|
| Users                           | 7,795      |
| Recipes                         | 8,000      |
| Observed ratings                | 253,120    |
| Average rating                  | 4.74       |
| Full possible user-recipe pairs | 62,360,000 |
| Observed share before split     | 0.0041     |

Larger Sparse Rating Subset Summary

``` r
# Compare the rating data used in Project 2 and Project 3.
tibble(Metric = c("Users",
                  "Recipes",
                  "Observed ratings",
                  "Training ratings",
                  "Training matrix sparsity"),
       `Project 2` = c(format_report_number(200, 0),
                       format_report_number(250, 0),
                       format_report_number(6199, 0),
                       format_report_number(5999, 0),
                       format_report_number(0.88, 4)),
       `Project 3` = c(format_report_number(n_distinct(ratings_long$user), 0),
                       format_report_number(n_distinct(ratings_long$recipe_id), 0),
                       format_report_number(nrow(ratings_long), 0),
                       format_report_number(202496, 0),
                       format_report_number(0.9968, 4))) %>%
  kable(caption = "Project 2 and Project 3 Rating Data Comparison")
```

| Metric                   | Project 2 | Project 3 |
|:-------------------------|:----------|:----------|
| Users                    | 200       | 7,795     |
| Recipes                  | 250       | 8,000     |
| Observed ratings         | 6,199     | 253,120   |
| Training ratings         | 5,999     | 202,496   |
| Training matrix sparsity | 0.8800    | 0.9968    |

Project 2 and Project 3 Rating Data Comparison

## 80/20 Train-Test Split

I used an 80/20 split because the model needs to be evaluated on ratings
it did not see during training. The split starts within each user so the
test users still have training history. Then it adjusts the count so the
final split is exactly 80% training and 20% testing overall.

``` r
# Start with a within-user split so each test user still has training history.
ratings_split_initial <- ratings_long %>%
  group_by(user) %>%
  mutate(row_in_user = row_number(),
         test_flag = row_in_user %in%
           sample(row_in_user, size = max(1, floor(0.20 * n())))) %>%
  ungroup() %>%
  mutate(data_split = if_else(test_flag, "Test", "Training")) %>%
  select(-row_in_user, -test_flag)

# Calculate how many test ratings are needed to make the total split exactly 80/20.
target_test_count <- round(0.20 * nrow(ratings_long))
current_test_count <- sum(ratings_split_initial$data_split == "Test")
additional_test_count <- target_test_count - current_test_count

# Add a small number of extra test rows while keeping at least one training rating per user.
additional_test_rows <- ratings_split_initial %>%
  filter(data_split == "Training") %>%
  group_by(user) %>%
  mutate(remaining_training_ratings = n()) %>%
  ungroup() %>%
  filter(remaining_training_ratings > 1) %>%
  slice_sample(n = additional_test_count) %>%
  select(user, recipe_id) %>%
  mutate(additional_test = TRUE)

# Apply the adjustment to create the final split.
ratings_split <- ratings_split_initial %>%
  left_join(additional_test_rows, by = c("user", "recipe_id")) %>%
  mutate(additional_test = replace_na(additional_test, FALSE),
         data_split = if_else(additional_test, "Test", data_split)) %>%
  select(-additional_test)

# Separate the training and test data.
training_data <- ratings_split %>%
  filter(data_split == "Training")

test_data <- ratings_split %>%
  filter(data_split == "Test")

# Print the final split counts.
tibble(Metric = c("Training ratings",
                  "Test ratings",
                  "Training share",
                  "Test share"),
       Value = c(format_report_number(nrow(training_data), 0),
                 format_report_number(nrow(test_data), 0),
                 format_report_number(nrow(training_data) / nrow(ratings_split), 3),
                 format_report_number(nrow(test_data) / nrow(ratings_split), 3))) %>%
  kable(caption = "80/20 Training and Test Split")
```

| Metric           | Value   |
|:-----------------|:--------|
| Training ratings | 202,496 |
| Test ratings     | 50,624  |
| Training share   | 0.800   |
| Test share       | 0.200   |

80/20 Training and Test Split

## Utility Matrix

The utility matrix is the recommender structure for this project. Users
are rows, recipes are columns, and ratings are the known entries. The
missing entries are not interpreted as dislikes. They are simply
unknown.

``` r
# Create numeric user indices for sparse matrices and recosystem.
user_index <- ratings_long %>%
  distinct(user) %>%
  arrange(user) %>%
  mutate(user_id = row_number() - 1L,
         user_row = row_number())

# Create numeric recipe indices for sparse matrices and recosystem.
recipe_index <- ratings_long %>%
  distinct(recipe_id) %>%
  arrange(recipe_id) %>%
  mutate(item_id = row_number() - 1L,
         recipe_column = row_number())

# Add numeric indices to the training ratings.
training_indexed <- training_data %>%
  inner_join(user_index, by = "user") %>%
  inner_join(recipe_index, by = "recipe_id")

# Add numeric indices to the test ratings.
test_indexed <- test_data %>%
  inner_join(user_index, by = "user") %>%
  inner_join(recipe_index, by = "recipe_id")

# Build a sparse user-recipe rating matrix from the training data.
rating_matrix_sparse <- sparseMatrix(i = training_indexed$user_row,
                                     j = training_indexed$recipe_column,
                                     x = training_indexed$rating,
                                     dims = c(nrow(user_index), nrow(recipe_index)),
                                     dimnames = list(user_index$user,
                                                     recipe_index$recipe_id))

# Convert the sparse matrix into a recommenderlab realRatingMatrix object.
rating_matrix_recommenderlab <- as(rating_matrix_sparse, "realRatingMatrix")

# Calculate observed entries and sparsity.
observed_entries <- length(rating_matrix_sparse@x)
possible_entries <- nrow(rating_matrix_sparse) * ncol(rating_matrix_sparse)

# Print the utility matrix summary.
tibble(Metric = c("Training users",
                  "Training recipes",
                  "Training ratings",
                  "Possible training matrix entries",
                  "Matrix sparsity",
                  "Matrix object"),
       Value = c(format_report_number(nrow(rating_matrix_sparse), 0),
                 format_report_number(ncol(rating_matrix_sparse), 0),
                 format_report_number(observed_entries, 0),
                 format_report_number(possible_entries, 0),
                 format_report_number(1 - observed_entries / possible_entries, 4),
                 class(rating_matrix_recommenderlab)[1])) %>%
  kable(caption = "Training Utility Matrix Summary")
```

| Metric                           | Value            |
|:---------------------------------|:-----------------|
| Training users                   | 7,795            |
| Training recipes                 | 8,000            |
| Training ratings                 | 202,496          |
| Possible training matrix entries | 62,360,000       |
| Matrix sparsity                  | 0.9968           |
| Matrix object                    | realRatingMatrix |

Training Utility Matrix Summary

# Recommendation Models

## Baseline Models

The global mean baseline is the simplest reference point. It predicts
the same average rating for every user and recipe. The bias baseline is
still simple, but it is more realistic because it accounts for users who
tend to rate high or low and recipes that tend to receive high or low
ratings.

``` r
# Calculate the overall average rating in the training data.
global_mean <- mean(training_data$rating, na.rm = TRUE)

# Estimate how much each user tends to rate above or below the global mean.
user_bias <- training_data %>%
  group_by(user) %>%
  summarize(user_bias = mean(rating - global_mean, na.rm = TRUE),
            .groups = "drop")

# Estimate how much each recipe tends to be rated above or below the global mean.
item_bias <- training_data %>%
  group_by(recipe_id) %>%
  summarize(item_bias = mean(rating - global_mean, na.rm = TRUE),
            .groups = "drop")

# Predict the held-out ratings with the global mean and bias baselines.
baseline_predictions <- test_data %>%
  left_join(user_bias, by = "user") %>%
  left_join(item_bias, by = "recipe_id") %>%
  mutate(user_bias = replace_na(user_bias, 0),
         item_bias = replace_na(item_bias, 0),
         Global_Mean_Baseline = global_mean,
         Bias_Baseline = pmin(5, pmax(1, global_mean + user_bias + item_bias))) %>%
  select(user, recipe_id, recipe_name, actual_rating = rating,
         Global_Mean_Baseline, Bias_Baseline)

# Show a sample of baseline predictions.
baseline_predictions %>%
  slice_head(n = 20) %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 2))) %>%
  kable(caption = "Baseline Test Predictions, First 20 Rows")
```

| user | recipe_id | recipe_name | actual_rating | Global_Mean_Baseline | Bias_Baseline |
|:---|:---|:---|:---|:---|:---|
| 107135 | 426090 | indian style green beans | 5.00 | 4.74 | 5.00 |
| 895132 | 426090 | indian style green beans | 5.00 | 4.74 | 5.00 |
| 30298 | 33096 | world s easiest lemonade ice cream pie | 4.00 | 4.74 | 5.00 |
| 89831 | 33096 | world s easiest lemonade ice cream pie | 5.00 | 4.74 | 5.00 |
| 130896 | 33096 | world s easiest lemonade ice cream pie | 5.00 | 4.74 | 4.77 |
| 297380 | 33096 | world s easiest lemonade ice cream pie | 5.00 | 4.74 | 4.38 |
| 870928 | 33096 | world s easiest lemonade ice cream pie | 5.00 | 4.74 | 4.78 |
| 286566 | 232041 | tomato and mushroom omelette | 5.00 | 4.74 | 5.00 |
| 482376 | 232041 | tomato and mushroom omelette | 5.00 | 4.74 | 5.00 |
| 128473 | 232041 | tomato and mushroom omelette | 5.00 | 4.74 | 5.00 |
| 68460 | 54550 | mushroom barley soup | 5.00 | 4.74 | 4.73 |
| 65056 | 54550 | mushroom barley soup | 5.00 | 4.74 | 4.58 |
| 203111 | 54550 | mushroom barley soup | 3.00 | 4.74 | 4.27 |
| 234075 | 54550 | mushroom barley soup | 5.00 | 4.74 | 4.80 |
| 6357 | 98783 | chocolate orange fudge | 5.00 | 4.74 | 4.95 |
| 206722 | 98783 | chocolate orange fudge | 5.00 | 4.74 | 4.65 |
| 234222 | 98783 | chocolate orange fudge | 5.00 | 4.74 | 5.00 |
| 158087 | 98783 | chocolate orange fudge | 5.00 | 4.74 | 4.63 |
| 381180 | 98783 | chocolate orange fudge | 5.00 | 4.74 | 5.00 |
| 209154 | 98783 | chocolate orange fudge | 5.00 | 4.74 | 4.90 |

Baseline Test Predictions, First 20 Rows

## Matrix Factorization with recosystem

This is the main recommender model for the assignment. recosystem uses
sparse user, item, and rating triples. This avoids the dense conversion
problem from the previous project and avoids custom prediction loops
over each user-recipe pair.

``` r
# Store the training ratings in the sparse triple format expected by recosystem.
train_reco_data <- data_memory(user_index = training_indexed$user_id,
                               item_index = training_indexed$item_id,
                               rating = training_indexed$rating,
                               index1 = FALSE)

# Store the test ratings in the same sparse format for vectorized prediction.
test_reco_data <- data_memory(user_index = test_indexed$user_id,
                              item_index = test_indexed$item_id,
                              rating = test_indexed$rating,
                              index1 = FALSE)

# Train a matrix factorization model with 20 latent factors.
model_20 <- Reco()
model_20$train(train_reco_data,
               opts = list(dim = 20,
                           costp_l2 = 0.1,
                           costq_l2 = 0.1,
                           lrate = 0.05,
                           niter = 15,
                           nthread = 2,
                           verbose = FALSE))

# Predict all test ratings at once and keep predictions in the 1-5 rating range.
pred_20 <- model_20$predict(test_reco_data, out_memory())
pred_20 <- pmin(5, pmax(1, pred_20))

# Train a matrix factorization model with 50 latent factors.
model_50 <- Reco()
model_50$train(train_reco_data,
               opts = list(dim = 50,
                           costp_l2 = 0.1,
                           costq_l2 = 0.1,
                           lrate = 0.05,
                           niter = 15,
                           nthread = 2,
                           verbose = FALSE))

# Predict all test ratings at once and keep predictions in the 1-5 rating range.
pred_50 <- model_50$predict(test_reco_data, out_memory())
pred_50 <- pmin(5, pmax(1, pred_50))

# Train a matrix factorization model with 100 latent factors.
model_100 <- Reco()
model_100$train(train_reco_data,
                opts = list(dim = 100,
                            costp_l2 = 0.1,
                            costq_l2 = 0.1,
                            lrate = 0.05,
                            niter = 15,
                            nthread = 2,
                            verbose = FALSE))

# Predict all test ratings at once and keep predictions in the 1-5 rating range.
pred_100 <- model_100$predict(test_reco_data, out_memory())
pred_100 <- pmin(5, pmax(1, pred_100))

# Train a matrix factorization model with 200 latent factors.
model_200 <- Reco()
model_200$train(train_reco_data,
                opts = list(dim = 200,
                            costp_l2 = 0.1,
                            costq_l2 = 0.1,
                            lrate = 0.05,
                            niter = 15,
                            nthread = 2,
                            verbose = FALSE))

# Predict all test ratings at once and keep predictions in the 1-5 rating range.
pred_200 <- model_200$predict(test_reco_data, out_memory())
pred_200 <- pmin(5, pmax(1, pred_200))
```

# Evaluation

## Rating Prediction Error

The table below compares the baselines with the matrix factorization
models. Lower RMSE and MAE are better. I included several factor sizes
because the number of latent factors changes the complexity of the
model.

``` r
# Combine baseline and matrix factorization error metrics into one table.
model_results <- bind_rows(
  tibble(Model = "Global Mean Baseline",
         Factors = NA_real_,
         RMSE = sqrt(mean((baseline_predictions$actual_rating -
                             baseline_predictions$Global_Mean_Baseline)^2,
                          na.rm = TRUE)),
         MAE = mean(abs(baseline_predictions$actual_rating -
                          baseline_predictions$Global_Mean_Baseline),
                    na.rm = TRUE)),
  tibble(Model = "Bias Baseline",
         Factors = NA_real_,
         RMSE = sqrt(mean((baseline_predictions$actual_rating -
                             baseline_predictions$Bias_Baseline)^2,
                          na.rm = TRUE)),
         MAE = mean(abs(baseline_predictions$actual_rating -
                          baseline_predictions$Bias_Baseline),
                    na.rm = TRUE)),
  tibble(Model = "recosystem Matrix Factorization",
         Factors = 20,
         RMSE = sqrt(mean((test_indexed$rating - pred_20)^2, na.rm = TRUE)),
         MAE = mean(abs(test_indexed$rating - pred_20), na.rm = TRUE)),
  tibble(Model = "recosystem Matrix Factorization",
         Factors = 50,
         RMSE = sqrt(mean((test_indexed$rating - pred_50)^2, na.rm = TRUE)),
         MAE = mean(abs(test_indexed$rating - pred_50), na.rm = TRUE)),
  tibble(Model = "recosystem Matrix Factorization",
         Factors = 100,
         RMSE = sqrt(mean((test_indexed$rating - pred_100)^2, na.rm = TRUE)),
         MAE = mean(abs(test_indexed$rating - pred_100), na.rm = TRUE)),
  tibble(Model = "recosystem Matrix Factorization",
         Factors = 200,
         RMSE = sqrt(mean((test_indexed$rating - pred_200)^2, na.rm = TRUE)),
         MAE = mean(abs(test_indexed$rating - pred_200), na.rm = TRUE))
)

# Print the model comparison.
model_results %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 3))) %>%
  kable(caption = "Rating Prediction Error by Model")
```

| Model                           | Factors | RMSE  | MAE   |
|:--------------------------------|:--------|:------|:------|
| Global Mean Baseline            | NA      | 0.604 | 0.420 |
| Bias Baseline                   | NA      | 0.560 | 0.345 |
| recosystem Matrix Factorization | 20.000  | 0.571 | 0.403 |
| recosystem Matrix Factorization | 50.000  | 0.572 | 0.402 |
| recosystem Matrix Factorization | 100.000 | 0.573 | 0.403 |
| recosystem Matrix Factorization | 200.000 | 0.575 | 0.405 |

Rating Prediction Error by Model

``` r
# Plot RMSE and MAE for each model.
model_results %>%
  mutate(Model_Label = if_else(is.na(Factors),
                               Model,
                               str_c(Model, " (", Factors, " factors)")),
         Model_Label = factor(Model_Label, levels = Model_Label)) %>%
  pivot_longer(cols = c(RMSE, MAE),
               names_to = "Metric",
               values_to = "Error") %>%
  ggplot(aes(x = Model_Label, y = Error, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  labs(title = "Food.com Rating Prediction Error by Model",
       subtitle = "Lower is better",
       x = NULL,
       y = "Prediction Error",
       fill = "Metric") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
```

![](Project-3--Matrix-Factorization_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
# Plot how prediction error changes as the number of latent factors increases.
model_results %>%
  filter(!is.na(Factors)) %>%
  pivot_longer(cols = c(RMSE, MAE),
               names_to = "Metric",
               values_to = "Error") %>%
  ggplot(aes(x = Factors, y = Error, color = Metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  scale_x_continuous(breaks = c(20, 50, 100, 200)) +
  labs(title = "Matrix Factorization Error by Number of Latent Factors",
       subtitle = "More factors do not automatically mean better predictions",
       x = "Latent Factors",
       y = "Prediction Error",
       color = "Metric") +
  theme_minimal()
```

![](Project-3--Matrix-Factorization_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

One thing that has stood out to me is that the bias baseline remains to
be better than the matrix factorization models, which does not
necessarily mean that matrix factorization is useless. It could meant
that this Food.com subset has a strong average-rating structure: some
users rate high, and some recipes are consistently liked. Matrix
factorization can capture hidden interactions beyond those averages, but
it still needs enough signal to beat a strong baseline.

## Example Predictions

``` r
# Select the matrix factorization model with the lowest RMSE.
best_factors <- model_results %>%
  filter(!is.na(Factors)) %>%
  arrange(RMSE, MAE) %>%
  slice_head(n = 1) %>%
  pull(Factors)

# Use the predictions from the best factor size.
best_predictions <- case_when(
  best_factors == 20 ~ pred_20,
  best_factors == 50 ~ pred_50,
  best_factors == 100 ~ pred_100,
  best_factors == 200 ~ pred_200
)

# Show the first 25 held-out predictions from the best matrix factorization model.
test_indexed %>%
  mutate(predicted_rating = best_predictions) %>%
  select(user, recipe_name, actual_rating = rating, predicted_rating) %>%
  slice_head(n = 25) %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 2))) %>%
  kable(caption = "Matrix Factorization Test Predictions, First 25 Rows")
```

| user | recipe_name | actual_rating | predicted_rating |
|:---|:---|:---|:---|
| 107135 | indian style green beans | 5.00 | 4.86 |
| 895132 | indian style green beans | 5.00 | 4.84 |
| 30298 | world s easiest lemonade ice cream pie | 4.00 | 4.78 |
| 89831 | world s easiest lemonade ice cream pie | 5.00 | 4.98 |
| 130896 | world s easiest lemonade ice cream pie | 5.00 | 4.57 |
| 297380 | world s easiest lemonade ice cream pie | 5.00 | 4.21 |
| 870928 | world s easiest lemonade ice cream pie | 5.00 | 4.64 |
| 286566 | tomato and mushroom omelette | 5.00 | 4.88 |
| 482376 | tomato and mushroom omelette | 5.00 | 4.80 |
| 128473 | tomato and mushroom omelette | 5.00 | 4.95 |
| 68460 | mushroom barley soup | 5.00 | 4.60 |
| 65056 | mushroom barley soup | 5.00 | 4.45 |
| 203111 | mushroom barley soup | 3.00 | 4.06 |
| 234075 | mushroom barley soup | 5.00 | 4.55 |
| 6357 | chocolate orange fudge | 5.00 | 4.85 |
| 206722 | chocolate orange fudge | 5.00 | 4.54 |
| 234222 | chocolate orange fudge | 5.00 | 4.91 |
| 158087 | chocolate orange fudge | 5.00 | 4.54 |
| 381180 | chocolate orange fudge | 5.00 | 4.85 |
| 209154 | chocolate orange fudge | 5.00 | 4.76 |
| 537188 | chocolate orange fudge | 3.00 | 4.58 |
| 544026 | chocolate orange fudge | 4.00 | 4.19 |
| 774187 | chocolate orange fudge | 5.00 | 4.50 |
| 297303 | chocolate orange fudge | 3.00 | 4.50 |
| 87236 | chocolate orange fudge | 5.00 | 4.86 |

Matrix Factorization Test Predictions, First 25 Rows

# Recommendation Example

The table below shows recommendations for one user using the best matrix
factorization model from the evaluation section. I include nutrition
fields because a food recommender should not be judged only by the
predicted rating.

``` r
# Choose an example user with the most ratings.
example_user <- ratings_split %>%
  count(user, sort = TRUE) %>%
  slice_head(n = 1) %>%
  pull(user)

# Get the numeric recosystem user index for the example user.
example_user_id <- user_index %>%
  filter(user == example_user) %>%
  pull(user_id)

# Identify recipes the example user has already rated.
user_rated_recipes <- ratings_split %>%
  filter(user == example_user) %>%
  pull(recipe_id)

# Create a candidate set from recipes the example user has not rated.
candidate_recipes <- ratings_split %>%
  distinct(recipe_id, recipe_name, calories, sodium, sugar,
           protein, saturated_fat, minutes) %>%
  filter(!recipe_id %in% user_rated_recipes) %>%
  inner_join(recipe_index, by = "recipe_id")

# Store candidate recipes in recosystem format for prediction.
candidate_reco_data <- data_memory(user_index = rep(example_user_id, nrow(candidate_recipes)),
                                   item_index = candidate_recipes$item_id,
                                   index1 = FALSE)

# Use the best matrix factorization model to score the candidate recipes.
if (best_factors == 20) {
  recommendation_scores <- model_20$predict(candidate_reco_data, out_memory())
}

if (best_factors == 50) {
  recommendation_scores <- model_50$predict(candidate_reco_data, out_memory())
}

if (best_factors == 100) {
  recommendation_scores <- model_100$predict(candidate_reco_data, out_memory())
}

if (best_factors == 200) {
  recommendation_scores <- model_200$predict(candidate_reco_data, out_memory())
}

# Keep recommendation scores in the 1-5 rating range.
recommendation_scores <- pmin(5, pmax(1, recommendation_scores))

# Show the top 10 recipe recommendations for the example user.
candidate_recipes %>%
  mutate(user = example_user,
         predicted_rating = recommendation_scores) %>%
  arrange(desc(predicted_rating)) %>%
  slice_head(n = 10) %>%
  select(user, recipe_name, predicted_rating,
         calories, sodium, sugar, protein, saturated_fat, minutes) %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 2))) %>%
  kable(caption = str_c("Matrix Factorization Recipe Recommendations for User ", example_user))
```

| user | recipe_name | predicted_rating | calories | sodium | sugar | protein | saturated_fat | minutes |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 140132 | just like dewey s candied walnut and grape salad | 4.97 | 210.30 | 21.00 | 41.00 | 14.00 | 28.00 | 10.00 |
| 140132 | cosmopolitan raspberry twist | 4.96 | 158.30 | 0.00 | 23.00 | 0.00 | 0.00 | 5.00 |
| 140132 | raspberry and white chocolate fudge brownies | 4.95 | 187.60 | 2.00 | 68.00 | 5.00 | 27.00 | 60.00 |
| 140132 | sugar and cinnamon spiced pecans | 4.94 | 867.60 | 4.00 | 174.00 | 18.00 | 37.00 | 70.00 |
| 140132 | dirt pudding | 4.93 | 964.60 | 44.00 | 363.00 | 21.00 | 158.00 | 15.00 |
| 140132 | manwich copycat | 4.93 | 323.40 | 68.00 | 66.00 | 46.00 | 33.00 | 25.00 |
| 140132 | no knead batter dinner rolls | 4.93 | 152.00 | 8.00 | 11.00 | 7.00 | 5.00 | 35.00 |
| 140132 | caramel puff corn | 4.92 | 210.90 | 8.00 | 71.00 | 1.00 | 39.00 | 50.00 |
| 140132 | rumbledethumps celtic potato cabbage cheese gratin | 4.92 | 729.30 | 23.00 | 26.00 | 46.00 | 134.00 | 35.00 |
| 140132 | quick cookie dough cheesecake bars | 4.92 | 391.40 | 9.00 | 37.00 | 9.00 | 47.00 | 50.00 |

Matrix Factorization Recipe Recommendations for User 140132

This table is useful, but it is not enough on its own. A high predicted
rating means the model thinks the user may like the recipe. It does not
mean the recipe is affordable, safe, culturally appropriate, or aligned
with the user’s health needs.

# Limitations and Next Steps

- **Prediction accuracy is not the whole problem:** RMSE and MAE tell me
  how close the model is to held-out ratings. They do not tell me
  whether the recommendations are healthy, equitable, affordable, or
  realistic.

- **Ratings are not the same as well-being:** Food.com ratings reflect
  what people liked on a recipe website. They do not include health
  goals, allergies, medication interactions, kitchen access, budget, or
  cultural food preferences.

- **Latent factors are hard to explain:** Matrix factorization can find
  hidden preference patterns, but those patterns are not always
  understandable to users. That matters in a food or health-related
  context.

- **A hybrid recommender would likely be better:** Matrix factorization
  can capture preference patterns, while content-based features can
  represent nutrition, ingredients, time, and other constraints. I think
  a responsible nutrition recommender would need both.

# Summary

This project implemented regularized matrix factorization for a Food.com
recipe recommender using recommender-specific R packages. I considered
traditional SVD and ALS but chose a regularized sparse matrix
factorization approach because the Food.com utility matrix is mostly
missing and should not be forced into a dense matrix. Compared with
Project 2, this implementation is more scalable. It avoids dense
similarity calculations and repeated row-by-row predictions, uses an
80/20 train-test split, and increases the sample to more than 250,000
observed ratings. The main lesson is that scalability changes how a
recommender should be built. A small manual implementation is useful for
learning the concepts, but larger recommender systems require sparse
data structures and optimized modeling tools. The results also show why
baselines matter. In this project, the bias baseline outperformed the
matrix factorization models, suggesting that user and recipe average
ratings explain much of the predictable behavior in this dataset. At the
same time, the health context still matters. A model that predicts
recipe ratings well is useful, but it is not automatically a responsible
nutrition recommender. A stronger nutrition recommender would combine
matrix factorization to learn user preferences with content-based
features that account for nutrition, ingredients, preparation time,
dietary restrictions, and other user needs.

# Sources

Food.com Recipes and Interactions. Kaggle dataset by Shuyang Li.
<https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions>

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization
techniques for recommender systems. *Computer*, 42(8), 30-37.
<https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf>

recosystem: Matrix Factorization for Recommender Systems.
<https://cran.r-project.org/package=recosystem>

recommenderlab: Lab for Developing and Testing Recommender Algorithms.
<https://cran.r-project.org/package=recommenderlab>
