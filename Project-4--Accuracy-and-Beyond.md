Project 4: Accuracy and Beyond for a Nutrition Recipe Recommender
================
Robert Gomez, DrPH, MPH
June 2026

- [Introduction](#introduction)
- [Approach and Recommender Context](#approach-and-recommender-context)
  - [Accuracy and Beyond](#accuracy-and-beyond)
  - [Algorithms Compared](#algorithms-compared)
  - [Diversity Reranking](#diversity-reranking)
- [Setup](#setup)
- [Loading Recipe and Rating Data](#loading-recipe-and-rating-data)
- [Data Preparation](#data-preparation)
  - [Sparse Rating Subset](#sparse-rating-subset)
  - [80/20 Train-Test Split](#8020-train-test-split)
  - [Utility Matrix](#utility-matrix)
- [Recommendation Models](#recommendation-models)
  - [Bias Baseline](#bias-baseline)
  - [Matrix Factorization with
    recosystem](#matrix-factorization-with-recosystem)
- [Offline Accuracy Evaluation](#offline-accuracy-evaluation)
  - [Rating Prediction Error](#rating-prediction-error)
  - [Select the Model for Recommendation
    Lists](#select-the-model-for-recommendation-lists)
- [Accuracy and Beyond-Accuracy List
  Evaluation](#accuracy-and-beyond-accuracy-list-evaluation)
  - [Evaluation Design](#evaluation-design)
  - [Recommendation and Reranking
    Functions](#recommendation-and-reranking-functions)
  - [Before-and-After List Metrics](#before-and-after-list-metrics)
  - [Example Recommendation Lists](#example-recommendation-lists)
- [Online Evaluation Proposal](#online-evaluation-proposal)
- [Limitations and Next Steps](#limitations-and-next-steps)
- [Summary](#summary)
- [Sources](#sources)

# Introduction

In this assignment, I continued the Food.com nutrition recipe
recommender from the earlier projects, but I shifted the focus from only
rating prediction to the broader question of recommender quality.
Project 3 asked whether matrix factorization could predict held-out
recipe ratings. Project 4 asked a more practical question: what happened
when I tried to improve the recommendation list for a user experience
goal such as diversity, and what did that do to accuracy? That
distinction mattered because a recommender can have good RMSE and still
create a narrow or repetitive user experience. If a user gets ten highly
rated dessert recipes, the model may be accurate in a narrow sense, but
the list may not be as useful as a set of high-quality recipes that
includes more variety in nutrition profile, cooking time, and food type.
In a nutrition context, I also did not want the recommender to optimize
only for engagement or preference. A recipe rating is useful, but it is
not the same thing as health value, safety, affordability, or fit with a
person’s goals. For Project 4, I used the same Food.com recipe and
interaction data so the comparison stayed connected to the earlier work.
The project included:

- **Two offline rating prediction approaches:** a bias baseline and
  regularized matrix factorization.
- **Top-N recommendation evaluation:** precision at 10, recall at 10,
  and hit rate at 10 using held-out relevant recipes.
- **A user experience intervention:** a diversity reranking step that
  tries to keep high predicted ratings while increasing variety in the
  recommendation list.
- **Before-and-after comparison:** accuracy and beyond-accuracy metrics
  for the original accuracy-ranked list and the diversity-reranked list.
- **An online evaluation proposal:** what I would have tested if the
  recommender could be evaluated with real users.

This project was still a recommender systems assignment, not medical
advice or a clinical decision support tool. The Food.com ratings were
preference data from a recipe website. They did not include allergies,
medical conditions, kitchen access, budget, or cultural context. That
limitation shaped what could be concluded from the results.

# Approach and Recommender Context

## Accuracy and Beyond

- Rating prediction accuracy answered one question: how close were the
  model’s predicted ratings to the ratings users actually gave in a
  held-out test set? For that task, RMSE and MAE were appropriate
  because the data included explicit 1-5 ratings.

- Top-N recommendation answered a different question: when the system
  shows a short list of recipes, how many of those recipes are relevant
  to the user? For this project, I treated held-out recipes rated 4 or 5
  stars as relevant and evaluated precision at 10, recall at 10, and hit
  rate at 10.

- Beyond-accuracy metrics asked whether the recommendation list had
  other useful properties. I focused on diversity, novelty, and catalog
  coverage. Diversity measured whether the items in a list were
  different from each other. Novelty measured whether the recommender
  surfaced less obvious or less frequently rated recipes. Catalog
  coverage measured how much of the available recipe catalog appeared in
  recommendation lists across users.

The main takeaway from Project 4 was that accuracy was necessary, but it
was not enough. If the model only recommended the same highly popular
recipes, it could look good offline while producing a stale user
experience.

## Algorithms Compared

The project compared two recommender approaches against the same offline
test data.

- **Bias baseline:** predicted ratings from the global average plus user
  and recipe average-rating effects. This was simple, but it was a
  strong baseline because some users tended to rate high and some
  recipes tended to receive high ratings.

- **Matrix factorization:** learned latent user and recipe factors from
  sparse rating triples using recosystem. This was the same general
  modeling family used in Project 3. It was more flexible than the bias
  baseline because it could learn hidden preference patterns beyond
  average user and item effects.

For the final list-building step, I used the matrix factorization model
because it gave each user a personalized score for candidate recipes. I
then compared two ranking policies:

- **Accuracy-ranked list:** chose the 10 unrated candidate recipes with
  the highest predicted ratings.
- **Diversity-reranked list:** started from strong predicted-rating
  candidates, then reranked with a greedy diversity penalty so the final
  list was not just the same kind of recipe repeated ten times.

## Diversity Reranking

The diversity reranker used recipe metadata that was already relevant to
this project: calories, fat, sodium, carbohydrate, sugar, protein,
saturated fat, and cooking time. I standardized these fields and
computed distances between recipes in that feature space. Then I used a
greedy maximum marginal relevance style rule:

1.  Pick the highest-scoring recipe first.
2.  For each later slot, balance predicted rating against the recipe’s
    distance from the recipes already selected.
3.  Continue until the top 10 list is filled.

The goal was not to make the list random. The goal was to preserve
relevance while reducing redundancy.

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

# Keep predicted ratings in the Food.com 1-5 rating range.
clip_rating <- function(x) {
  pmin(5, pmax(1, x))
}
```

# Loading Recipe and Rating Data

The data came from the Food.com recipes and interactions dataset on
Kaggle. I used the same data source as the earlier projects so the main
change was evaluation design rather than the recommendation domain.

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

# Food.com stored nutrition as a list-like field, so I split it into separate columns.
nutrition_values <- recipes_raw %>%
  mutate(nutrition_clean = str_remove_all(nutrition, "\\[|\\]")) %>%
  separate_wider_delim(nutrition_clean,
                       delim = ",",
                       names = c("calories", "fat", "sugar", "sodium",
                                 "protein", "saturated_fat", "carbohydrate"),
                       too_few = "align_start",
                       too_many = "drop") %>%
  mutate(across(c(calories, fat, sugar, sodium,
                  protein, saturated_fat, carbohydrate), as.numeric))

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

## Sparse Rating Subset

The Food.com interaction matrix was extremely sparse. Most users rated
only a small share of all possible recipes. Because this project
evaluated both rating prediction and top-N recommendation, I kept users
and recipes with enough rating history to support a meaningful
train-test split.

``` r
# Set a seed so the same users, recipes, and train-test split are selected each run.
set.seed(612)

# Count rating activity by user and recipe.
interaction_counts <- interactions %>%
  count(user, name = "user_rating_count")

recipe_counts <- interactions %>%
  count(recipe_id, name = "recipe_rating_count")

# Keep users and recipes with enough history for collaborative filtering.
dense_interactions <- interactions %>%
  left_join(interaction_counts, by = "user") %>%
  left_join(recipe_counts, by = "recipe_id") %>%
  filter(user_rating_count >= 5,
         recipe_rating_count >= 5)

# Select active users and commonly rated recipes.
selected_users <- dense_interactions %>%
  count(user, sort = TRUE) %>%
  slice_head(n = 8000) %>%
  pull(user)

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

# Print the subset summary.
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
  kable(caption = "Sparse Rating Subset Summary")
```

| Metric                          | Value      |
|:--------------------------------|:-----------|
| Users                           | 7,795      |
| Recipes                         | 8,000      |
| Observed ratings                | 253,120    |
| Average rating                  | 4.74       |
| Full possible user-recipe pairs | 62,360,000 |
| Observed share before split     | 0.0041     |

Sparse Rating Subset Summary

## 80/20 Train-Test Split

I used an 80/20 split because the model needed to be evaluated on
ratings it did not see during training. The split started within each
user so test users still had training history.

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

# Calculate how many test ratings are needed to make the total split close to 80/20.
target_test_count <- round(0.20 * nrow(ratings_long))
current_test_count <- sum(ratings_split_initial$data_split == "Test")
additional_test_count <- max(0, target_test_count - current_test_count)

# Add extra test rows while keeping at least one training rating per user.
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
                  "Test share",
                  "Relevant test ratings, 4 or 5 stars"),
       Value = c(format_report_number(nrow(training_data), 0),
                 format_report_number(nrow(test_data), 0),
                 format_report_number(nrow(training_data) / nrow(ratings_split), 3),
                 format_report_number(nrow(test_data) / nrow(ratings_split), 3),
                 format_report_number(sum(test_data$rating >= 4), 0))) %>%
  kable(caption = "80/20 Training and Test Split")
```

| Metric                              | Value   |
|:------------------------------------|:--------|
| Training ratings                    | 202,496 |
| Test ratings                        | 50,624  |
| Training share                      | 0.800   |
| Test share                          | 0.200   |
| Relevant test ratings, 4 or 5 stars | 48,331  |

80/20 Training and Test Split

## Utility Matrix

The utility matrix is the recommender structure for this project. Users
are rows, recipes are columns, and ratings are the known entries.
Missing entries are unknown, not automatic dislikes.

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

# Add numeric indices to the training and test ratings.
training_indexed <- training_data %>%
  inner_join(user_index, by = "user") %>%
  inner_join(recipe_index, by = "recipe_id")

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

## Bias Baseline

The bias baseline was simple, but it was a useful benchmark. It
predicted each test rating using the global average rating, the user’s
average deviation from that global average, and the recipe’s average
deviation from that global average.

``` r
# Calculate the overall average rating in the training data.
global_mean <- mean(training_data$rating, na.rm = TRUE)

# Estimate user and item average-rating effects.
user_bias <- training_data %>%
  group_by(user) %>%
  summarize(user_bias = mean(rating - global_mean, na.rm = TRUE),
            .groups = "drop")

item_bias <- training_data %>%
  group_by(recipe_id) %>%
  summarize(item_bias = mean(rating - global_mean, na.rm = TRUE),
            .groups = "drop")

# Predict the held-out ratings with the bias baseline.
baseline_predictions <- test_data %>%
  left_join(user_bias, by = "user") %>%
  left_join(item_bias, by = "recipe_id") %>%
  mutate(user_bias = replace_na(user_bias, 0),
         item_bias = replace_na(item_bias, 0),
         Bias_Baseline = clip_rating(global_mean + user_bias + item_bias)) %>%
  select(user, recipe_id, recipe_name, actual_rating = rating, Bias_Baseline)

# Show a sample of baseline predictions.
baseline_predictions %>%
  slice_head(n = 20) %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 2))) %>%
  kable(caption = "Bias Baseline Test Predictions, First 20 Rows")
```

| user | recipe_id | recipe_name | actual_rating | Bias_Baseline |
|:---|:---|:---|:---|:---|
| 107135 | 426090 | indian style green beans | 5.00 | 5.00 |
| 895132 | 426090 | indian style green beans | 5.00 | 5.00 |
| 30298 | 33096 | world s easiest lemonade ice cream pie | 4.00 | 5.00 |
| 89831 | 33096 | world s easiest lemonade ice cream pie | 5.00 | 5.00 |
| 130896 | 33096 | world s easiest lemonade ice cream pie | 5.00 | 4.77 |
| 297380 | 33096 | world s easiest lemonade ice cream pie | 5.00 | 4.38 |
| 870928 | 33096 | world s easiest lemonade ice cream pie | 5.00 | 4.78 |
| 286566 | 232041 | tomato and mushroom omelette | 5.00 | 5.00 |
| 482376 | 232041 | tomato and mushroom omelette | 5.00 | 5.00 |
| 128473 | 232041 | tomato and mushroom omelette | 5.00 | 5.00 |
| 68460 | 54550 | mushroom barley soup | 5.00 | 4.73 |
| 65056 | 54550 | mushroom barley soup | 5.00 | 4.58 |
| 203111 | 54550 | mushroom barley soup | 3.00 | 4.27 |
| 234075 | 54550 | mushroom barley soup | 5.00 | 4.80 |
| 6357 | 98783 | chocolate orange fudge | 5.00 | 4.95 |
| 206722 | 98783 | chocolate orange fudge | 5.00 | 4.65 |
| 234222 | 98783 | chocolate orange fudge | 5.00 | 5.00 |
| 158087 | 98783 | chocolate orange fudge | 5.00 | 4.63 |
| 381180 | 98783 | chocolate orange fudge | 5.00 | 5.00 |
| 209154 | 98783 | chocolate orange fudge | 5.00 | 4.90 |

Bias Baseline Test Predictions, First 20 Rows

## Matrix Factorization with recosystem

Matrix factorization learned lower-dimensional user and recipe factors
from the sparse user-recipe-rating triples. I trained two factor sizes
so the more complex model was still compared against a smaller
alternative.

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

# Train a matrix factorization model with 40 latent factors.
model_40 <- Reco()
model_40$train(train_reco_data,
               opts = list(dim = 40,
                           costp_l2 = 0.1,
                           costq_l2 = 0.1,
                           lrate = 0.05,
                           niter = 15,
                           nthread = 2,
                           verbose = FALSE))

# Predict all test ratings at once.
pred_40 <- clip_rating(model_40$predict(test_reco_data, out_memory()))

# Train a matrix factorization model with 80 latent factors.
model_80 <- Reco()
model_80$train(train_reco_data,
               opts = list(dim = 80,
                           costp_l2 = 0.1,
                           costq_l2 = 0.1,
                           lrate = 0.05,
                           niter = 15,
                           nthread = 2,
                           verbose = FALSE))

# Predict all test ratings at once.
pred_80 <- clip_rating(model_80$predict(test_reco_data, out_memory()))
```

# Offline Accuracy Evaluation

## Rating Prediction Error

RMSE and MAE were the rating prediction metrics. Lower values were
better. I included both because RMSE penalizes large errors more
heavily, while MAE is easier to read as an average rating-point error.

``` r
# Combine baseline and matrix factorization error metrics into one table.
model_results <- bind_rows(
  tibble(Model = "Bias Baseline",
         Factors = NA_real_,
         RMSE = sqrt(mean((baseline_predictions$actual_rating -
                             baseline_predictions$Bias_Baseline)^2,
                          na.rm = TRUE)),
         MAE = mean(abs(baseline_predictions$actual_rating -
                          baseline_predictions$Bias_Baseline),
                    na.rm = TRUE)),
  tibble(Model = "recosystem Matrix Factorization",
         Factors = 40,
         RMSE = sqrt(mean((test_indexed$rating - pred_40)^2, na.rm = TRUE)),
         MAE = mean(abs(test_indexed$rating - pred_40), na.rm = TRUE)),
  tibble(Model = "recosystem Matrix Factorization",
         Factors = 80,
         RMSE = sqrt(mean((test_indexed$rating - pred_80)^2, na.rm = TRUE)),
         MAE = mean(abs(test_indexed$rating - pred_80), na.rm = TRUE))
)

# Print the model comparison.
model_results %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 3))) %>%
  kable(caption = "Rating Prediction Error by Model")
```

| Model                           | Factors | RMSE  | MAE   |
|:--------------------------------|:--------|:------|:------|
| Bias Baseline                   | NA      | 0.560 | 0.345 |
| recosystem Matrix Factorization | 40.000  | 0.572 | 0.404 |
| recosystem Matrix Factorization | 80.000  | 0.572 | 0.402 |

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

![](Project-4--Accuracy-and-Beyond_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

The bias baseline was an important reality check. If it was close to or
better than matrix factorization, that meant average user and recipe
effects explained a lot of the predictable rating behavior in this
subset. Matrix factorization was still useful for personalized ranking,
but it should not be assumed to be better just because it is more
advanced.

## Select the Model for Recommendation Lists

``` r
# Select the matrix factorization model with the lowest RMSE.
best_factors <- model_results %>%
  filter(!is.na(Factors)) %>%
  arrange(RMSE, MAE) %>%
  slice_head(n = 1) %>%
  pull(Factors)

# Use the predictions from the best factor size.
best_test_predictions <- case_when(
  best_factors == 40 ~ pred_40,
  best_factors == 80 ~ pred_80
)

# Store the selected model object for list recommendation.
best_model <- if (best_factors == 40) model_40 else model_80

tibble(Metric = c("Selected matrix factorization factors",
                  "Reason for selection"),
       Value = c(best_factors,
                 "Lowest RMSE among the matrix factorization models")) %>%
  kable(caption = "Selected Model for Top-N Recommendation")
```

| Metric | Value |
|:---|:---|
| Selected matrix factorization factors | 40 |
| Reason for selection | Lowest RMSE among the matrix factorization models |

Selected Model for Top-N Recommendation

# Accuracy and Beyond-Accuracy List Evaluation

## Evaluation Design

For top-N evaluation, I used held-out relevant recipes as the offline
answer key. A test recipe was treated as relevant if the user rated it 4
or 5 stars. I evaluated the recommendation lists for a sample of users
who had at least one relevant item in the test set.

This was a valid offline evaluation setup, but it was still limited. The
model could only get credit for recommending recipes that happened to
appear in the test data. It could not get credit for a recipe the user
would have liked but never rated.

``` r
# Identify test users with at least one relevant held-out recipe.
eligible_eval_users <- test_data %>%
  filter(rating >= 4) %>%
  count(user, name = "relevant_test_items") %>%
  arrange(desc(relevant_test_items)) %>%
  slice_head(n = 100) %>%
  pull(user)

# Build recipe-level metadata and popularity features for list evaluation.
recipe_features <- ratings_long %>%
  distinct(recipe_id, recipe_name, calories, fat, sodium, carbohydrate,
           sugar, protein, saturated_fat, minutes) %>%
  mutate(across(c(calories, fat, sodium, carbohydrate, sugar, protein,
                  saturated_fat, minutes),
                ~ replace_na(.x, median(.x, na.rm = TRUE)))) %>%
  inner_join(recipe_index, by = "recipe_id")

recipe_popularity <- training_data %>%
  count(recipe_id, name = "training_rating_count") %>%
  right_join(recipe_features %>% select(recipe_id), by = "recipe_id") %>%
  mutate(training_rating_count = replace_na(training_rating_count, 0),
         popularity_share = (training_rating_count + 1) /
           sum(training_rating_count + 1),
         novelty = -log2(popularity_share))

recipe_features <- recipe_features %>%
  left_join(recipe_popularity, by = "recipe_id")

# Standardize features used for distance-based diversity.
diversity_feature_names <- c("calories", "fat", "sodium", "carbohydrate",
                             "sugar", "protein", "saturated_fat", "minutes")

scaled_diversity_features <- recipe_features %>%
  select(all_of(diversity_feature_names)) %>%
  mutate(across(everything(), ~ log1p(pmax(.x, 0)))) %>%
  scale() %>%
  as.matrix()

rownames(scaled_diversity_features) <- recipe_features$recipe_id

tibble(Metric = c("Evaluation users",
                  "Top-N list length",
                  "Relevant threshold",
                  "Diversity features"),
       Value = c(length(eligible_eval_users),
                 10,
                 "Held-out rating >= 4",
                 str_c(diversity_feature_names, collapse = ", "))) %>%
  kable(caption = "Top-N Evaluation Setup")
```

| Metric | Value |
|:---|:---|
| Evaluation users | 100 |
| Top-N list length | 10 |
| Relevant threshold | Held-out rating \>= 4 |
| Diversity features | calories, fat, sodium, carbohydrate, sugar, protein, saturated_fat, minutes |

Top-N Evaluation Setup

## Recommendation and Reranking Functions

``` r
# Calculate mean pairwise distance within a recommendation list.
mean_pairwise_diversity <- function(recipe_ids, feature_matrix) {
  if (length(recipe_ids) < 2) {
    return(0)
  }

  list_features <- feature_matrix[recipe_ids, , drop = FALSE]
  mean(as.numeric(dist(list_features)))
}

# Rerank candidate recipes by balancing predicted rating and list diversity.
diversity_rerank <- function(candidate_tbl, feature_matrix, k = 10, lambda = 0.30) {
  candidate_tbl <- candidate_tbl %>%
    mutate(score_scaled = as.numeric(scale(predicted_rating)))

  if (all(is.na(candidate_tbl$score_scaled))) {
    candidate_tbl <- candidate_tbl %>%
      mutate(score_scaled = 0)
  }

  selected_ids <- character(0)
  remaining <- candidate_tbl

  for (slot in seq_len(k)) {
    if (slot == 1) {
      next_row <- remaining %>%
        arrange(desc(predicted_rating)) %>%
        slice_head(n = 1)
    } else {
      remaining_distances <- map_dbl(remaining$recipe_id, function(candidate_id) {
        candidate_vector <- feature_matrix[candidate_id, , drop = FALSE]
        selected_vectors <- feature_matrix[selected_ids, , drop = FALSE]
        min(sqrt(rowSums((t(t(selected_vectors) - as.numeric(candidate_vector)))^2)))
      })

      distance_scaled <- as.numeric(scale(remaining_distances))
      distance_scaled[is.na(distance_scaled)] <- 0

      next_row <- remaining %>%
        mutate(diversity_distance = distance_scaled,
               rerank_score = (1 - lambda) * score_scaled +
                 lambda * diversity_distance) %>%
        arrange(desc(rerank_score), desc(predicted_rating)) %>%
        slice_head(n = 1)
    }

    selected_ids <- c(selected_ids, next_row$recipe_id)
    remaining <- remaining %>%
      filter(!recipe_id %in% selected_ids)
  }

  candidate_tbl %>%
    filter(recipe_id %in% selected_ids) %>%
    mutate(rank = match(recipe_id, selected_ids)) %>%
    arrange(rank)
}

# Score candidate recipes for one user and return both ranking policies.
build_user_recommendations <- function(user_id_value, k = 10, candidate_pool = 250) {
  user_numeric_id <- user_index %>%
    filter(user == user_id_value) %>%
    pull(user_id)

  already_rated <- training_data %>%
    filter(user == user_id_value) %>%
    pull(recipe_id)

  candidates <- recipe_features %>%
    filter(!recipe_id %in% already_rated)

  candidate_reco_data <- data_memory(user_index = rep(user_numeric_id, nrow(candidates)),
                                     item_index = candidates$item_id,
                                     index1 = FALSE)

  candidate_scores <- clip_rating(best_model$predict(candidate_reco_data, out_memory()))

  scored_candidates <- candidates %>%
    mutate(user = user_id_value,
           predicted_rating = candidate_scores) %>%
    arrange(desc(predicted_rating))

  accuracy_ranked <- scored_candidates %>%
    slice_head(n = k) %>%
    mutate(policy = "Accuracy-ranked",
           rank = row_number())

  diversity_pool <- scored_candidates %>%
    slice_head(n = candidate_pool)

  diversity_ranked <- diversity_rerank(diversity_pool,
                                       feature_matrix = scaled_diversity_features,
                                       k = k,
                                       lambda = 0.30) %>%
    mutate(policy = "Diversity-reranked")

  bind_rows(accuracy_ranked, diversity_ranked) %>%
    select(policy, user, rank, recipe_id, recipe_name, predicted_rating,
           calories, sodium, sugar, protein, saturated_fat, minutes,
           training_rating_count, novelty)
}

# Build recommendation lists for the evaluation users.
recommendation_lists <- map_dfr(eligible_eval_users, build_user_recommendations)
```

## Before-and-After List Metrics

``` r
# Create the held-out relevant item sets for each evaluation user.
relevant_test_items <- test_data %>%
  filter(user %in% eligible_eval_users,
         rating >= 4) %>%
  group_by(user) %>%
  summarize(relevant_items = list(unique(recipe_id)),
            relevant_count = n_distinct(recipe_id),
            .groups = "drop")

# Calculate user-level top-N accuracy and beyond-accuracy metrics.
user_list_metrics <- recommendation_lists %>%
  group_by(policy, user) %>%
  summarize(recommended_items = list(recipe_id),
            mean_predicted_rating = mean(predicted_rating, na.rm = TRUE),
            mean_novelty = mean(novelty, na.rm = TRUE),
            mean_popularity = mean(training_rating_count, na.rm = TRUE),
            diversity = mean_pairwise_diversity(recipe_id, scaled_diversity_features),
            .groups = "drop") %>%
  left_join(relevant_test_items, by = "user") %>%
  mutate(hits = map2_int(recommended_items, relevant_items,
                         ~ length(intersect(.x, .y))),
         precision_at_10 = hits / 10,
         recall_at_10 = hits / relevant_count,
         hit_rate_at_10 = as.numeric(hits > 0))

# Summarize list metrics by ranking policy.
list_metric_summary <- user_list_metrics %>%
  group_by(policy) %>%
  summarize(`Precision at 10` = mean(precision_at_10, na.rm = TRUE),
            `Recall at 10` = mean(recall_at_10, na.rm = TRUE),
            `Hit Rate at 10` = mean(hit_rate_at_10, na.rm = TRUE),
            `Mean Predicted Rating` = mean(mean_predicted_rating, na.rm = TRUE),
            `Mean Diversity` = mean(diversity, na.rm = TRUE),
            `Mean Novelty` = mean(mean_novelty, na.rm = TRUE),
            `Mean Training Popularity` = mean(mean_popularity, na.rm = TRUE),
            .groups = "drop")

# Add catalog coverage across recommendation lists.
coverage_summary <- recommendation_lists %>%
  group_by(policy) %>%
  summarize(`Catalog Coverage` = n_distinct(recipe_id) / n_distinct(recipe_features$recipe_id),
            .groups = "drop")

list_metric_summary <- list_metric_summary %>%
  left_join(coverage_summary, by = "policy")

list_metric_summary %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 3))) %>%
  kable(caption = "Before-and-After Top-10 Accuracy and Beyond-Accuracy Metrics")
```

| policy | Precision at 10 | Recall at 10 | Hit Rate at 10 | Mean Predicted Rating | Mean Diversity | Mean Novelty | Mean Training Popularity | Catalog Coverage |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Accuracy-ranked | 0.009 | 0.001 | 0.090 | 4.949 | 4.016 | 12.725 | 39.872 | 0.011 |
| Diversity-reranked | 0.012 | 0.002 | 0.120 | 4.949 | 6.466 | 12.690 | 42.760 | 0.016 |

Before-and-After Top-10 Accuracy and Beyond-Accuracy Metrics

``` r
# Show the change caused by diversity reranking.
before_after_change <- list_metric_summary %>%
  pivot_longer(cols = -policy,
               names_to = "Metric",
               values_to = "Value") %>%
  pivot_wider(names_from = policy,
              values_from = Value) %>%
  mutate(Change = `Diversity-reranked` - `Accuracy-ranked`)

before_after_change %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 3))) %>%
  kable(caption = "Change After Diversity Reranking")
```

| Metric                   | Accuracy-ranked | Diversity-reranked | Change |
|:-------------------------|:----------------|:-------------------|:-------|
| Precision at 10          | 0.009           | 0.012              | 0.003  |
| Recall at 10             | 0.001           | 0.002              | 0.000  |
| Hit Rate at 10           | 0.090           | 0.120              | 0.030  |
| Mean Predicted Rating    | 4.949           | 4.949              | -0.001 |
| Mean Diversity           | 4.016           | 6.466              | 2.450  |
| Mean Novelty             | 12.725          | 12.690             | -0.036 |
| Mean Training Popularity | 39.872          | 42.760             | 2.888  |
| Catalog Coverage         | 0.011           | 0.016              | 0.005  |

Change After Diversity Reranking

``` r
# Plot before-and-after changes for selected list metrics.
before_after_change %>%
  filter(Metric %in% c("Precision at 10", "Recall at 10", "Hit Rate at 10",
                       "Mean Diversity", "Mean Novelty", "Catalog Coverage")) %>%
  ggplot(aes(x = reorder(Metric, Change), y = Change, fill = Change > 0)) +
  geom_col(width = 0.65) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#2A9D8F", "FALSE" = "#6C757D"),
                    guide = "none") +
  labs(title = "Metric Change After Diversity Reranking",
       subtitle = "Positive values mean the diversity-reranked list was higher",
       x = NULL,
       y = "Change") +
  theme_minimal()
```

![](Project-4--Accuracy-and-Beyond_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

The diversity intervention had to be judged as a tradeoff, not as a free
improvement. If precision at 10 or recall at 10 decreased, that meant
the reranker gave up some offline accuracy against the held-out ratings.
If diversity, novelty, or coverage improved, that meant the recommender
was doing more than returning the same most obvious recipes. In a
user-facing setting, the right balance would depend on the product goal
and on whether users actually preferred the more varied list.

## Example Recommendation Lists

The table below showed the two top-10 lists for one user. This made the
tradeoff easier to inspect than metrics alone.

``` r
# Select one example user from the evaluation sample.
example_user <- eligible_eval_users[1]

recommendation_lists %>%
  filter(user == example_user) %>%
  arrange(policy, rank) %>%
  select(policy, rank, recipe_name, predicted_rating,
         calories, sodium, sugar, protein, minutes,
         training_rating_count, novelty) %>%
  mutate(across(where(is.numeric), ~ format_report_number(.x, 2))) %>%
  kable(caption = str_c("Accuracy-Ranked and Diversity-Reranked Lists for User ", example_user))
```

| policy | rank | recipe_name | predicted_rating | calories | sodium | sugar | protein | minutes | training_rating_count | novelty |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Accuracy-ranked | 1.00 | just like dewey s candied walnut and grape salad | 5.00 | 210.30 | 21.00 | 41.00 | 14.00 | 10.00 | 10.00 | 14.22 |
| Accuracy-ranked | 2.00 | cosmopolitan raspberry twist | 4.97 | 158.30 | 0.00 | 23.00 | 0.00 | 5.00 | 11.00 | 14.10 |
| Accuracy-ranked | 3.00 | raspberry and white chocolate fudge brownies | 4.93 | 187.60 | 2.00 | 68.00 | 5.00 | 60.00 | 10.00 | 14.22 |
| Accuracy-ranked | 4.00 | no knead batter dinner rolls | 4.93 | 152.00 | 8.00 | 11.00 | 7.00 | 35.00 | 14.00 | 13.78 |
| Accuracy-ranked | 5.00 | dirt pudding | 4.92 | 964.60 | 44.00 | 363.00 | 21.00 | 15.00 | 15.00 | 13.68 |
| Accuracy-ranked | 6.00 | manwich copycat | 4.92 | 323.40 | 68.00 | 66.00 | 46.00 | 25.00 | 9.00 | 14.36 |
| Accuracy-ranked | 7.00 | rumbledethumps celtic potato cabbage cheese gratin | 4.92 | 729.30 | 23.00 | 26.00 | 46.00 | 35.00 | 12.00 | 13.98 |
| Accuracy-ranked | 8.00 | best buffalo wing sauce | 4.92 | 85.00 | 28.00 | 1.00 | 0.00 | 8.00 | 12.00 | 13.98 |
| Accuracy-ranked | 9.00 | sugar and cinnamon spiced pecans | 4.91 | 867.60 | 4.00 | 174.00 | 18.00 | 70.00 | 21.00 | 13.22 |
| Accuracy-ranked | 10.00 | caramel puff corn | 4.91 | 210.90 | 8.00 | 71.00 | 1.00 | 50.00 | 13.00 | 13.88 |
| Diversity-reranked | 1.00 | just like dewey s candied walnut and grape salad | 5.00 | 210.30 | 21.00 | 41.00 | 14.00 | 10.00 | 10.00 | 14.22 |
| Diversity-reranked | 2.00 | cosmopolitan raspberry twist | 4.97 | 158.30 | 0.00 | 23.00 | 0.00 | 5.00 | 11.00 | 14.10 |
| Diversity-reranked | 3.00 | raspberry and white chocolate fudge brownies | 4.93 | 187.60 | 2.00 | 68.00 | 5.00 | 60.00 | 10.00 | 14.22 |
| Diversity-reranked | 4.00 | dirt pudding | 4.92 | 964.60 | 44.00 | 363.00 | 21.00 | 15.00 | 15.00 | 13.68 |
| Diversity-reranked | 5.00 | classic entrecote bordelaise steak in red wine with shallots | 4.91 | 947.80 | 22.00 | 1.00 | 178.00 | 30.00 | 7.00 | 14.68 |
| Diversity-reranked | 6.00 | ruth s homemade vanilla extract | 4.85 | 2,159.60 | 0.00 | 0.00 | 0.00 | 86,405.00 | 12.00 | 13.98 |
| Diversity-reranked | 7.00 | best buffalo wing sauce | 4.92 | 85.00 | 28.00 | 1.00 | 0.00 | 8.00 | 12.00 | 13.98 |
| Diversity-reranked | 8.00 | no knead batter dinner rolls | 4.93 | 152.00 | 8.00 | 11.00 | 7.00 | 35.00 | 14.00 | 13.78 |
| Diversity-reranked | 9.00 | sugar and cinnamon spiced pecans | 4.91 | 867.60 | 4.00 | 174.00 | 18.00 | 70.00 | 21.00 | 13.22 |
| Diversity-reranked | 10.00 | rumbledethumps celtic potato cabbage cheese gratin | 4.92 | 729.30 | 23.00 | 26.00 | 46.00 | 35.00 | 12.00 | 13.98 |

Accuracy-Ranked and Diversity-Reranked Lists for User 140132

# Online Evaluation Proposal

The offline evaluation was useful, but it could not answer the most
important user experience questions. It could say whether a held-out
rating was recovered. It could not tell whether a user would cook the
recipe, trust the recommendation, feel that the list was varied, or come
back to the recommender later.

If online evaluation had been possible, I would have designed a small
A/B test with three arms:

- **Control:** the current accuracy-ranked matrix factorization list.
- **Treatment A:** the diversity-reranked list used in this project.
- **Treatment B:** a hybrid list that combined predicted rating,
  diversity, and explicit nutrition constraints chosen by the user.

The primary online metric would have been recipe save rate or
click-through rate on the top-10 list, depending on what the product
could observe reliably. I would also have tracked downstream metrics
such as recipe completion, repeat usage, hiding or dismissing
recommendations, and whether the user returned to the recommender.
Because this was a food and nutrition context, I would have included
user-control measures, such as whether users adjusted filters for
sodium, sugar, preparation time, ingredients, or dietary pattern.

I would also have collected short qualitative feedback after the
recommendation list was shown:

- Did the list feel relevant?
- Did the list include enough variety?
- Were any recommendations inappropriate because of dietary needs,
  allergies, budget, time, or cultural fit?
- Did the user understand why a recipe was recommended?

For a reasonable online environment, users would have been randomly
assigned to one ranking policy, the list position would have been
logged, and the analysis would have compared both immediate engagement
and later satisfaction. I would have avoided declaring success from
click-through alone because a clicked recipe could still be a poor
recommendation if the user could not cook it or if it conflicted with
their needs.

# Limitations and Next Steps

- **Offline relevance was incomplete:** A held-out 4- or 5-star rating
  was a useful signal, but the test set only included recipes the user
  actually rated. The model received no credit for unobserved recipes
  the user might have liked.

- **Diversity was based on available metadata:** I used nutrition fields
  and cooking time for diversity. That was reasonable for this project,
  but it did not capture cuisine, ingredients, texture, cost, or dietary
  restrictions.

- **Ratings were not health outcomes:** Food.com ratings measured
  preference, not safety or well-being. A high predicted rating should
  not be interpreted as a healthy recommendation.

- **The reranking weight was not fully tuned:** I used one diversity
  weight to show the before-and-after effect. A stronger project would
  tune that weight on validation data and then evaluate once on a final
  test set.

- **Cold start remained a problem:** Users with few ratings and new
  recipes with no ratings were still difficult for collaborative
  filtering. A hybrid model that used ingredients, tags, nutrition, and
  text descriptions would help.

- **Online testing would change the evidence:** Offline metrics could
  guide development, but real user behavior would be needed to decide
  whether the diversity tradeoff was worth it.

# Summary

This project extended the Food.com recipe recommender from rating
prediction into broader recommender evaluation. I compared a bias
baseline with two matrix factorization models using RMSE and MAE, then
used the selected matrix factorization model to build top-10
recommendation lists. The main Project 4 intervention was a diversity
reranker that balances predicted rating with distance across nutrition
and cooking-time features. The most important lesson was that a
recommender should not be judged by one metric alone. RMSE and MAE
helped evaluate rating prediction. Precision at 10, recall at 10, and
hit rate at 10 helped evaluate the recommendation list against held-out
relevant recipes. Diversity, novelty, and catalog coverage showed
whether the list was broader than the most obvious high-scoring items.
The diversity reranker made that tradeoff visible. For a real nutrition
recommender, I would not have stopped here. The next version should be
hybrid, user-controllable, and evaluated with online feedback. It should
also include safeguards for dietary restrictions, allergies,
affordability, and user autonomy before it is treated as more than a
course recommender prototype.

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
