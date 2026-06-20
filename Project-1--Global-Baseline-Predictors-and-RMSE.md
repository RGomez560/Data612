Project 1: Global Baseline Predictors and RMSE
================
Robert Gomez, DrPH, MPH
June 2026

- [<u>**Approach**</u>](#approach)
- [<u>**Base Code**</u>](#base-code)
  - [Loading packages](#loading-packages)
  - [Creating Rating Data](#creating-rating-data)
  - [Transforming Data and Creating Train-Test
    Split](#transforming-data-and-creating-train-test-split)
  - [Creating Training User-Item Matrix and Average
    Table](#creating-training-user-item-matrix-and-average-table)
  - [Calculating Raw Average and
    RMSE](#calculating-raw-average-and-rmse)
  - [Calculating User and Item Bias](#calculating-user-and-item-bias)
  - [Creating Baseline Predictors](#creating-baseline-predictors)
  - [Calculating Baseline RMSE](#calculating-baseline-rmse)
- [<u>**Recommendation**</u>](#recommendation)
  - [Predicting an Unrated Game](#predicting-an-unrated-game)
- [<u>**Summary**</u>](#summary)
- [<u>**Response to Assignment
  Feedback**</u>](#response-to-assignment-feedback)

# <u>**Approach**</u>

In this assignment, I will build a Global Baseline Predictor recommender
system. From a business perspective, this system recommends Nintendo
Switch video games to players based on numeric ratings from other
players. The goal is to start with very little information. First, I
will use the raw average rating from the training data. Then, I will
adjust that estimate by accounting for user bias and item bias. I will
compare the models using RMSE for both the training and test data.

# <u>**Base Code**</u>

## Loading packages

``` r
# Load packages needed for data wrangling and table output.
pacman::p_load(tidyverse,
               knitr)

# Format numeric table values to two decimal places.
format_two_decimals <- function(table) {
  table %>%
    mutate(across(where(is.numeric), ~ format(round(.x, 2), nsmall = 2)))
}
```

## Creating Rating Data

The rating data include six users and six Nintendo Switch games. Ratings
use a 1 to 5 scale. Missing values mean that the user has not rated the
game.

``` r
# Create a small user-item ratings matrix with missing values.
ratings_original <- tribble(~user, ~`Mario Kart 8 Deluxe`, ~`Zelda: Breath of the Wild`, ~`Animal Crossing: New Horizons`, ~`Super Smash Bros. Ultimate`, ~`Super Mario Odyssey`, ~`Splatoon 3`,
                            "Alicia", 5, 4, 5, 3, NA, 4,
                            "Ben", 4, 5, 3, 5, 4, NA,
                            "Carla", 5, NA, 4, 4, 5, 3,
                            "Devon", 3, 5, NA, 5, 4, 2,
                            "Elena", NA, 4, 5, 3, 5, 4,
                            "Farah", 4, 3, 5, NA, 4, 5)

# Print the original matrix so the missing values are visible.
kable(ratings_original,
      caption = "Original User-Item Ratings Matrix")
```

| user | Mario Kart 8 Deluxe | Zelda: Breath of the Wild | Animal Crossing: New Horizons | Super Smash Bros. Ultimate | Super Mario Odyssey | Splatoon 3 |
|:---|---:|---:|---:|---:|---:|---:|
| Alicia | 5 | 4 | 5 | 3 | NA | 4 |
| Ben | 4 | 5 | 3 | 5 | 4 | NA |
| Carla | 5 | NA | 4 | 4 | 5 | 3 |
| Devon | 3 | 5 | NA | 5 | 4 | 2 |
| Elena | NA | 4 | 5 | 3 | 5 | 4 |
| Farah | 4 | 3 | 5 | NA | 4 | 5 |

Original User-Item Ratings Matrix

## Transforming Data and Creating Train-Test Split

I transformed the data into long format and then selected one observed
rating from each user for the test set. The remaining observed ratings
were used for training.

``` r
# Convert the user-item matrix into long format for easier calculations.
ratings_long <- ratings_original %>%
  pivot_longer(cols = -user,
               names_to = "item",
               values_to = "rating")

# Select one observed rating from each user to hold out for testing.
test_set <- tribble(
  ~user,    ~item,
  "Alicia", "Mario Kart 8 Deluxe",
  "Ben",    "Zelda: Breath of the Wild",
  "Carla",  "Super Mario Odyssey",
  "Devon",  "Super Mario Odyssey",
  "Elena",  "Animal Crossing: New Horizons",
  "Farah",  "Animal Crossing: New Horizons"
)

# Flag each row as Training, Test, or Missing.
ratings_split <- ratings_long %>%
  left_join(test_set %>%
              mutate(data_split = "Test"),
            by = c("user", "item")) %>%
  mutate(data_split = case_when(is.na(rating) ~ "Missing",
                                data_split == "Test" ~ "Test",
                                TRUE ~ "Training"))

# Keep the observed training ratings for model fitting.
training_data <- ratings_split %>%
  filter(data_split == "Training")

# Keep the held-out observed ratings for model testing.
test_data <- ratings_split %>%
  filter(data_split == "Test")

# Print the observed ratings and their assigned data split.
ratings_split %>%
  filter(data_split != "Missing") %>%
  arrange(user, item) %>%
  kable(caption = "Training and Test Ratings")
```

| user   | item                          | rating | data_split |
|:-------|:------------------------------|-------:|:-----------|
| Alicia | Animal Crossing: New Horizons |      5 | Training   |
| Alicia | Mario Kart 8 Deluxe           |      5 | Test       |
| Alicia | Splatoon 3                    |      4 | Training   |
| Alicia | Super Smash Bros. Ultimate    |      3 | Training   |
| Alicia | Zelda: Breath of the Wild     |      4 | Training   |
| Ben    | Animal Crossing: New Horizons |      3 | Training   |
| Ben    | Mario Kart 8 Deluxe           |      4 | Training   |
| Ben    | Super Mario Odyssey           |      4 | Training   |
| Ben    | Super Smash Bros. Ultimate    |      5 | Training   |
| Ben    | Zelda: Breath of the Wild     |      5 | Test       |
| Carla  | Animal Crossing: New Horizons |      4 | Training   |
| Carla  | Mario Kart 8 Deluxe           |      5 | Training   |
| Carla  | Splatoon 3                    |      3 | Training   |
| Carla  | Super Mario Odyssey           |      5 | Test       |
| Carla  | Super Smash Bros. Ultimate    |      4 | Training   |
| Devon  | Mario Kart 8 Deluxe           |      3 | Training   |
| Devon  | Splatoon 3                    |      2 | Training   |
| Devon  | Super Mario Odyssey           |      4 | Test       |
| Devon  | Super Smash Bros. Ultimate    |      5 | Training   |
| Devon  | Zelda: Breath of the Wild     |      5 | Training   |
| Elena  | Animal Crossing: New Horizons |      5 | Test       |
| Elena  | Splatoon 3                    |      4 | Training   |
| Elena  | Super Mario Odyssey           |      5 | Training   |
| Elena  | Super Smash Bros. Ultimate    |      3 | Training   |
| Elena  | Zelda: Breath of the Wild     |      4 | Training   |
| Farah  | Animal Crossing: New Horizons |      5 | Test       |
| Farah  | Mario Kart 8 Deluxe           |      4 | Training   |
| Farah  | Splatoon 3                    |      5 | Training   |
| Farah  | Super Mario Odyssey           |      4 | Training   |
| Farah  | Zelda: Breath of the Wild     |      3 | Training   |

Training and Test Ratings

## Creating Training User-Item Matrix and Average Table

I created a user-item matrix, calculated the user averages, calculated
the game averages, and then calculated each game’s difference from the
overall average. The test ratings are removed from this table because
the model should only learn from the training data.

``` r
# Recreate a user-item matrix using only the training ratings.
training_matrix <- ratings_split %>%
  mutate(training_rating = if_else(data_split == "Training", rating, NA_real_)) %>%
  select(user, item, training_rating) %>%
  pivot_wider(names_from = item,
              values_from = training_rating)

# Store the game columns so they can be reused in later calculations.
game_columns <- c("Mario Kart 8 Deluxe", "Zelda: Breath of the Wild",
                  "Animal Crossing: New Horizons", "Super Smash Bros. Ultimate",
                  "Super Mario Odyssey", "Splatoon 3")

# Calculate each user's average rating from the training data.
training_table <- training_matrix %>%
  mutate(user_avg = round(rowMeans(across(all_of(game_columns)), na.rm = TRUE), 2))

# Calculate the overall average rating from the training matrix.
overall_avg <- training_table %>%
  select(all_of(game_columns)) %>%
  as.matrix() %>%
  mean(na.rm = TRUE) %>%
  round(2)

# Calculate each game's average rating from the training data.
game_avg <- round(colMeans(select(training_table, all_of(game_columns)), na.rm = TRUE), 2) %>%
  as_tibble_row()

# Add the game average row.
game_avg <- game_avg %>%
  mutate(user = "Game Avg.",
         user_avg = overall_avg) %>%
  select(user, all_of(game_columns), user_avg)

# Calculate how far each game average is from the overall average.
game_avg_difference <- game_avg %>%
  mutate(across(all_of(game_columns), ~ round(.x - overall_avg, 2)),
         user = "Game Avg. Difference",
         user_avg = overall_avg)

# Combine the user ratings, game averages, and average differences into one table.
final_training_table <- bind_rows(training_table, game_avg, game_avg_difference) %>%
  mutate(user_avg_difference = user_avg - .env$overall_avg,
         across(where(is.numeric), ~ round(.x, 2))) %>%
  rename(User = user,
         `User Avg.` = user_avg,
         `User Avg. Difference` = user_avg_difference)

# Print the training matrix and average differences.
final_training_table %>%
  format_two_decimals() %>%
  kable(caption = "Training User-Item Matrix and Average Differences")
```

| User | Mario Kart 8 Deluxe | Zelda: Breath of the Wild | Animal Crossing: New Horizons | Super Smash Bros. Ultimate | Super Mario Odyssey | Splatoon 3 | User Avg. | User Avg. Difference |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Alicia | NA | 4.00 | 5.00 | 3.00 | NA | 4.00 | 4.00 | 0.04 |
| Ben | 4.00 | NA | 3.00 | 5.00 | 4.00 | NA | 4.00 | 0.04 |
| Carla | 5.00 | NA | 4.00 | 4.00 | NA | 3.00 | 4.00 | 0.04 |
| Devon | 3.00 | 5.00 | NA | 5.00 | NA | 2.00 | 3.75 | -0.21 |
| Elena | NA | 4.00 | NA | 3.00 | 5.00 | 4.00 | 4.00 | 0.04 |
| Farah | 4.00 | 3.00 | NA | NA | 4.00 | 5.00 | 4.00 | 0.04 |
| Game Avg. | 4.00 | 4.00 | 4.00 | 4.00 | 4.33 | 3.60 | 3.96 | 0.00 |
| Game Avg. Difference | 0.04 | 0.04 | 0.04 | 0.04 | 0.37 | -0.36 | 3.96 | 0.00 |

Training User-Item Matrix and Average Differences

## Calculating Raw Average and RMSE

The raw average model uses the same prediction for every user-item pair.
This prediction is the overall mean rating from the training data.

``` r
# Create an RMSE function to evaluate prediction error.
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Assign the overall average rating as the prediction for every user-game pair.
raw_average_predictions <- ratings_long %>%
  select(user, item, actual_rating = rating) %>%
  mutate(raw_average_prediction = overall_avg) %>%
  left_join(ratings_split %>% select(user, item, data_split),
            by = c("user", "item"))

# Print the raw average prediction table.
raw_average_predictions %>%
  select(user, item, actual_rating, data_split, raw_average_prediction) %>%
  arrange(user, item) %>%
  format_two_decimals() %>%
  kable(caption = "Raw Average Predictions for User-Item Combinations")
```

| user | item | actual_rating | data_split | raw_average_prediction |
|:---|:---|:---|:---|:---|
| Alicia | Animal Crossing: New Horizons | 5.00 | Training | 3.96 |
| Alicia | Mario Kart 8 Deluxe | 5.00 | Test | 3.96 |
| Alicia | Splatoon 3 | 4.00 | Training | 3.96 |
| Alicia | Super Mario Odyssey | NA | Missing | 3.96 |
| Alicia | Super Smash Bros. Ultimate | 3.00 | Training | 3.96 |
| Alicia | Zelda: Breath of the Wild | 4.00 | Training | 3.96 |
| Ben | Animal Crossing: New Horizons | 3.00 | Training | 3.96 |
| Ben | Mario Kart 8 Deluxe | 4.00 | Training | 3.96 |
| Ben | Splatoon 3 | NA | Missing | 3.96 |
| Ben | Super Mario Odyssey | 4.00 | Training | 3.96 |
| Ben | Super Smash Bros. Ultimate | 5.00 | Training | 3.96 |
| Ben | Zelda: Breath of the Wild | 5.00 | Test | 3.96 |
| Carla | Animal Crossing: New Horizons | 4.00 | Training | 3.96 |
| Carla | Mario Kart 8 Deluxe | 5.00 | Training | 3.96 |
| Carla | Splatoon 3 | 3.00 | Training | 3.96 |
| Carla | Super Mario Odyssey | 5.00 | Test | 3.96 |
| Carla | Super Smash Bros. Ultimate | 4.00 | Training | 3.96 |
| Carla | Zelda: Breath of the Wild | NA | Missing | 3.96 |
| Devon | Animal Crossing: New Horizons | NA | Missing | 3.96 |
| Devon | Mario Kart 8 Deluxe | 3.00 | Training | 3.96 |
| Devon | Splatoon 3 | 2.00 | Training | 3.96 |
| Devon | Super Mario Odyssey | 4.00 | Test | 3.96 |
| Devon | Super Smash Bros. Ultimate | 5.00 | Training | 3.96 |
| Devon | Zelda: Breath of the Wild | 5.00 | Training | 3.96 |
| Elena | Animal Crossing: New Horizons | 5.00 | Test | 3.96 |
| Elena | Mario Kart 8 Deluxe | NA | Missing | 3.96 |
| Elena | Splatoon 3 | 4.00 | Training | 3.96 |
| Elena | Super Mario Odyssey | 5.00 | Training | 3.96 |
| Elena | Super Smash Bros. Ultimate | 3.00 | Training | 3.96 |
| Elena | Zelda: Breath of the Wild | 4.00 | Training | 3.96 |
| Farah | Animal Crossing: New Horizons | 5.00 | Test | 3.96 |
| Farah | Mario Kart 8 Deluxe | 4.00 | Training | 3.96 |
| Farah | Splatoon 3 | 5.00 | Training | 3.96 |
| Farah | Super Mario Odyssey | 4.00 | Training | 3.96 |
| Farah | Super Smash Bros. Ultimate | NA | Missing | 3.96 |
| Farah | Zelda: Breath of the Wild | 3.00 | Training | 3.96 |

Raw Average Predictions for User-Item Combinations

``` r
# Calculate RMSE for the raw average model on the training data.
raw_training_rmse <- round(rmse(training_data$rating, overall_avg), 2)

# Calculate RMSE for the raw average model on the test data.
raw_test_rmse <- round(rmse(test_data$rating, overall_avg), 2)

# Create a summary table for the raw average results.
raw_results <- tibble(
  Model = "Raw Average",
  Dataset = c("Training", "Test"),
  RMSE = c(raw_training_rmse, raw_test_rmse)
)

# Print the raw average RMSE table.
raw_results %>%
  format_two_decimals() %>%
  kable(caption = "Raw Average RMSE")
```

| Model       | Dataset  | RMSE |
|:------------|:---------|:-----|
| Raw Average | Training | 0.84 |
| Raw Average | Test     | 0.95 |

Raw Average RMSE

## Calculating User and Item Bias

The next step is to calculate how much each user tends to rate above or
below the overall average. After that, I calculate how much each game
tends to rate above or below the average after accounting for user bias.

``` r
# Calculate each user's average difference from the overall average.
user_bias <- training_data %>%
  group_by(user) %>%
  summarize(user_avg_difference = round(mean(rating - overall_avg), 2),
            .groups = "drop")

# Calculate each game's average difference after accounting for user differences.
item_bias <- training_data %>%
  left_join(user_bias, by = "user") %>%
  group_by(item) %>%
  summarize(item_avg_difference = round(mean(rating - overall_avg - user_avg_difference), 2),
            .groups = "drop")

# Print the user average differences.
user_bias %>%
  format_two_decimals() %>%
  kable(caption = "User Average Differences")
```

| user   | user_avg_difference |
|:-------|:--------------------|
| Alicia | 0.04                |
| Ben    | 0.04                |
| Carla  | 0.04                |
| Devon  | -0.21               |
| Elena  | 0.04                |
| Farah  | 0.04                |

User Average Differences

``` r
# Print the game average differences.
item_bias %>%
  format_two_decimals() %>%
  kable(caption = "Game Average Differences")
```

| item                          | item_avg_difference |
|:------------------------------|:--------------------|
| Animal Crossing: New Horizons | 0.00                |
| Mario Kart 8 Deluxe           | 0.06                |
| Splatoon 3                    | -0.35               |
| Super Mario Odyssey           | 0.33                |
| Super Smash Bros. Ultimate    | 0.05                |
| Zelda: Breath of the Wild     | 0.06                |

Game Average Differences

## Creating Baseline Predictors

The baseline predictor is calculated by adding the raw average, the user
average difference, and the game average difference.

``` r
# Create baseline predictions for every user-game pair.
baseline_predictions <- ratings_long %>%
  select(user, item, actual_rating = rating) %>%
  left_join(user_bias, by = "user") %>%
  left_join(item_bias, by = "item") %>%
  mutate(predicted_rating = round(overall_avg + user_avg_difference + item_avg_difference, 2),
         predicted_rating = round(pmin(pmax(predicted_rating, 1), 5), 2)) %>%
  left_join(ratings_split %>% select(user, item, data_split),
            by = c("user", "item"))

# Print the baseline predictor table.
baseline_predictions %>%
  select(user, item, actual_rating, data_split, predicted_rating) %>%
  arrange(user, item) %>%
  format_two_decimals() %>%
  kable(caption = "Baseline Predictor Table")
```

| user | item | actual_rating | data_split | predicted_rating |
|:---|:---|:---|:---|:---|
| Alicia | Animal Crossing: New Horizons | 5.00 | Training | 4.00 |
| Alicia | Mario Kart 8 Deluxe | 5.00 | Test | 4.06 |
| Alicia | Splatoon 3 | 4.00 | Training | 3.65 |
| Alicia | Super Mario Odyssey | NA | Missing | 4.33 |
| Alicia | Super Smash Bros. Ultimate | 3.00 | Training | 4.05 |
| Alicia | Zelda: Breath of the Wild | 4.00 | Training | 4.06 |
| Ben | Animal Crossing: New Horizons | 3.00 | Training | 4.00 |
| Ben | Mario Kart 8 Deluxe | 4.00 | Training | 4.06 |
| Ben | Splatoon 3 | NA | Missing | 3.65 |
| Ben | Super Mario Odyssey | 4.00 | Training | 4.33 |
| Ben | Super Smash Bros. Ultimate | 5.00 | Training | 4.05 |
| Ben | Zelda: Breath of the Wild | 5.00 | Test | 4.06 |
| Carla | Animal Crossing: New Horizons | 4.00 | Training | 4.00 |
| Carla | Mario Kart 8 Deluxe | 5.00 | Training | 4.06 |
| Carla | Splatoon 3 | 3.00 | Training | 3.65 |
| Carla | Super Mario Odyssey | 5.00 | Test | 4.33 |
| Carla | Super Smash Bros. Ultimate | 4.00 | Training | 4.05 |
| Carla | Zelda: Breath of the Wild | NA | Missing | 4.06 |
| Devon | Animal Crossing: New Horizons | NA | Missing | 3.75 |
| Devon | Mario Kart 8 Deluxe | 3.00 | Training | 3.81 |
| Devon | Splatoon 3 | 2.00 | Training | 3.40 |
| Devon | Super Mario Odyssey | 4.00 | Test | 4.08 |
| Devon | Super Smash Bros. Ultimate | 5.00 | Training | 3.80 |
| Devon | Zelda: Breath of the Wild | 5.00 | Training | 3.81 |
| Elena | Animal Crossing: New Horizons | 5.00 | Test | 4.00 |
| Elena | Mario Kart 8 Deluxe | NA | Missing | 4.06 |
| Elena | Splatoon 3 | 4.00 | Training | 3.65 |
| Elena | Super Mario Odyssey | 5.00 | Training | 4.33 |
| Elena | Super Smash Bros. Ultimate | 3.00 | Training | 4.05 |
| Elena | Zelda: Breath of the Wild | 4.00 | Training | 4.06 |
| Farah | Animal Crossing: New Horizons | 5.00 | Test | 4.00 |
| Farah | Mario Kart 8 Deluxe | 4.00 | Training | 4.06 |
| Farah | Splatoon 3 | 5.00 | Training | 3.65 |
| Farah | Super Mario Odyssey | 4.00 | Training | 4.33 |
| Farah | Super Smash Bros. Ultimate | NA | Missing | 4.05 |
| Farah | Zelda: Breath of the Wild | 3.00 | Training | 4.06 |

Baseline Predictor Table

## Calculating Baseline RMSE

``` r
# Filter the baseline predictions to the training rows.
baseline_training_data <- baseline_predictions %>%
  filter(data_split == "Training")

# Filter the baseline predictions to the test rows.
baseline_test_data <- baseline_predictions %>%
  filter(data_split == "Test")

# Calculate RMSE for the baseline predictor on the training data.
baseline_training_rmse <- round(rmse(baseline_training_data$actual_rating,
                                     baseline_training_data$predicted_rating), 2)

# Calculate RMSE for the baseline predictor on the test data.
baseline_test_rmse <- round(rmse(baseline_test_data$actual_rating,
                                 baseline_test_data$predicted_rating), 2)

# Combine the raw average and baseline RMSE results.
final_results <- bind_rows(
  raw_results,
  tibble(Model = "Baseline Predictor",
         Dataset = c("Training", "Test"),
         RMSE = c(baseline_training_rmse, baseline_test_rmse))
)

# Print the final RMSE comparison table.
final_results %>%
  format_two_decimals() %>%
  kable(caption = "RMSE Comparison")
```

| Model              | Dataset  | RMSE |
|:-------------------|:---------|:-----|
| Raw Average        | Training | 0.84 |
| Raw Average        | Test     | 0.95 |
| Baseline Predictor | Training | 0.81 |
| Baseline Predictor | Test     | 0.84 |

RMSE Comparison

# <u>**Recommendation**</u>

## Predicting an Unrated Game

How would Elena rate *Mario Kart 8 Deluxe*?

``` r
# Estimate Elena's rating for Mario Kart 8 Deluxe.
baseline_predictions %>%
  filter(user == "Elena",
         item == "Mario Kart 8 Deluxe") %>%
  transmute(user,
            item,
            predicted_rating = round(predicted_rating, 2)) %>%
  format_two_decimals() %>%
  kable(caption = "Example Prediction")
```

| user  | item                | predicted_rating |
|:------|:--------------------|:-----------------|
| Elena | Mario Kart 8 Deluxe | 4.06             |

Example Prediction

Which missing game rating has the highest predicted value for each user?

``` r
# Recommend the highest predicted missing game for each user.
baseline_predictions %>%
  filter(data_split == "Missing") %>%
  group_by(user) %>%
  slice_max(predicted_rating, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(user, item, predicted_rating) %>%
  format_two_decimals() %>%
  kable(caption = "Recommended Game for Each User")
```

| user   | item                          | predicted_rating |
|:-------|:------------------------------|:-----------------|
| Alicia | Super Mario Odyssey           | 4.33             |
| Ben    | Splatoon 3                    | 3.65             |
| Carla  | Zelda: Breath of the Wild     | 4.06             |
| Devon  | Animal Crossing: New Horizons | 3.75             |
| Elena  | Mario Kart 8 Deluxe           | 4.06             |
| Farah  | Super Smash Bros. Ultimate    | 4.05             |

Recommended Game for Each User

# <u>**Summary**</u>

The raw average rating from the training data was 3.96. This model is
useful as a simple starting point because it gives one prediction for
every user-item combination. The baseline predictor improved on the raw
average model by adding user and game average differences. In this
dataset, the baseline predictor had a lower RMSE for both the training
and test data. This means that accounting for user bias and item bias
gave better predictions than using the same average rating for every
Nintendo Switch game. This is still a simple recommender system. It does
not use game genres, platform behavior, play time, or similarities
between users. However, it creates a clear baseline that more advanced
recommender systems should improve on.

# <u>**Response to Assignment Feedback**</u>

Based on the feedback, I also calculated MAE as a second error metric.
RMSE and MAE both measure prediction error, but they do it in different
ways. RMSE squares the errors before averaging, which gives more weight
to larger mistakes. MAE takes the absolute value of each error and
calculates the average number of rating points that the prediction is
off by.

``` r
# Create an MAE function to calculate the average absolute prediction error.
mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

# Calculate MAE for the raw average model on the training data.
raw_training_mae <- round(mae(training_data$rating, overall_avg), 2)

# Calculate MAE for the raw average model on the test data.
raw_test_mae <- round(mae(test_data$rating, overall_avg), 2)

# Calculate MAE for the baseline predictor on the training data.
baseline_training_mae <- round(mae(baseline_training_data$actual_rating,
                                   baseline_training_data$predicted_rating), 2)

# Calculate MAE for the baseline predictor on the test data.
baseline_test_mae <- round(mae(baseline_test_data$actual_rating,
                               baseline_test_data$predicted_rating), 2)

# Combine the RMSE and MAE results into one table.
feedback_results <- final_results %>%
  left_join(
    tibble(Model = c("Raw Average", "Raw Average",
                     "Baseline Predictor", "Baseline Predictor"),
           Dataset = c("Training", "Test", "Training", "Test"),
           MAE = c(raw_training_mae, raw_test_mae,
                   baseline_training_mae, baseline_test_mae)),
    by = c("Model", "Dataset")
  )

# Print the RMSE and MAE comparison table.
feedback_results %>%
  format_two_decimals() %>%
  kable(caption = "RMSE and MAE Comparison")
```

| Model              | Dataset  | RMSE | MAE  |
|:-------------------|:---------|:-----|:-----|
| Raw Average        | Training | 0.84 | 0.64 |
| Raw Average        | Test     | 0.95 | 0.87 |
| Baseline Predictor | Training | 0.81 | 0.67 |
| Baseline Predictor | Test     | 0.84 | 0.77 |

RMSE and MAE Comparison

The RMSE results show that the baseline predictor improved performance
for both the training and test data. The raw average model had an RMSE
of 0.84 for training and 0.95 for test, while the baseline predictor had
an RMSE of 0.81 for training and 0.84 for test. This suggests that
adjusting for user differences and game differences helped reduce larger
prediction errors. The MAE results tell a slightly different story. The
baseline predictor improved test MAE noticeably, dropping from 0.87 to
0.77, but training MAE went in the other direction, increasing from 0.64
to 0.67. As a result, it seems like the baseline model did not improve
every error measure on the training data, but it did improve the test
results. Overall, the main difference between the two metrics is that
RMSE is more sensitive to larger prediction errors, while MAE is easier
to read as the average rating-point mistake.
