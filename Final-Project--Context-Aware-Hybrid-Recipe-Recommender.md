Final Project: Context-Aware Hybrid Recipe Recommender
================
Robert Gomez, DrPH, MPH
July 2026

- [Introduction](#introduction)
- [Approach](#approach)
- [Recommender System Framing](#recommender-system-framing)
- [Connection to Earlier Projects](#connection-to-earlier-projects)
- [Why These Metrics?](#why-these-metrics)
- [Data and Feature Preparation](#data-and-feature-preparation)
- [Guideline-Informed Synthetic
  Profiles](#guideline-informed-synthetic-profiles)
- [Local Benchmark Models](#local-benchmark-models)
- [Spark ALS Model](#spark-als-model)
- [Local Versus Spark Comparison](#local-versus-spark-comparison)
- [Context-Aware Reranking](#context-aware-reranking)
- [Recommendation-List Evaluation](#recommendation-list-evaluation)
- [Example Recommendations](#example-recommendations)
- [Discussion](#discussion)
- [Limitations](#limitations)
- [Sources](#sources)

# Introduction

For the final project, I built a context-aware hybrid recipe recommender
using the Food.com recipe and interaction data. Overall, this project
extended the earlier DATA 612 work in two related ways. First, it used
the full raw Food.com interaction file where practical, rather than only
the preprocessed training split used in Project 5. Second, it moved
beyond rating prediction by adding a health-context reranking layer.
This mattered because a recipe recommender that only predicted what a
user may rate highly could still recommend recipes that were not
realistic, practical, or aligned with a user’s nutrition goals.

The main recommender model was Spark MLlib alternating least squares, or
ALS. ALS learned from user-recipe ratings and generated candidate
recommendations. However, because Food.com ratings reflected preference
rather than health needs, I then reranked those candidates using recipe
metadata and guideline-informed synthetic health-context profiles. For
example, the profiles simulate contexts such as heart-conscious eating,
lower sugar, high-protein practical meals, quick weeknight meals,
budget/access awareness, and vegetarian preference. As a result, the
project asked not only whether a model could predict ratings, but also
whether the recommendation list could be made more useful for a
nutrition-oriented use case.

That said, it important to highlight that this is still a course
recommender prototype. It is not meant to provide medical advice, and
the synthetic profiles do not represent real patients. Instead, the
profiles were used as a structured way to make missing context visible
and to test whether the recommender could balance predicted preference
with nutrition, practicality, and user context. Ultimately, this made
the project closer to the type of recommender I would want to evaluate
in a public health or health-promotion setting, while still keeping the
claims proportional to the available data.

# Approach

The recommender had three layers that worked together:

- **Preference layer:** Spark ALS estimated which recipes a user may
  like from Food.com ratings.
- **Recipe feature layer:** Food.com metadata described nutrition,
  ingredients, tags, cooking time, steps, and ingredient burden.
- **Context layer:** guideline-informed synthetic profiles represented
  nutrition goals and practical constraints.

In addition, the system compared two main recommendation policies:

- **ALS-only list:** the top 10 recipes were ranked by ALS predicted
  rating.
- **Context-aware hybrid list:** ALS candidates were reranked using a
  profile-specific context-fit score.

The core question was whether the context-aware hybrid list appeared to
improve nutrition and practicality alignment while preserving reasonable
predicted preference. In other words, the project was intentionally
framed as a tradeoff problem rather than a simple accuracy contest.

# Recommender System Framing

In recommender-system terms, the Food.com data forms a sparse utility
matrix, where users are the rows, recipes are the columns, and known
entries are the ratings that users actually gave to recipes. Most
user-recipe pairs are unknown, so the recommender is not simply looking
up existing ratings. Instead, it is estimating missing preferences from
observed patterns. This distinction is important because missing ratings
should be treated as unknown information, not as evidence that a user
disliked a recipe.

This project used a hybrid design. Spark ALS was the collaborative
filtering part because it learned from user-recipe interaction patterns.
In addition, the recipe metadata created an item-profile layer because
each recipe had nutrition, tags, ingredients, cooking time, and
preparation burden. The synthetic health-context profiles added a
user-context layer that Food.com did not naturally provide. Taken
together, these components were why I described the final system as a
context-aware hybrid recommender rather than only a matrix factorization
model.

I also kept the evaluation separate by task. RMSE and MAE evaluate
rating prediction on held-out known ratings. Precision at 10, recall at
10, and hit rate at 10 evaluate a top-10 recommendation list. Coverage,
diversity, novelty, and context alignment evaluate whether the list is
useful beyond simple rating accuracy. Overall, this layered evaluation
is necessary because a recommender can look strong on one metric while
still producing recommendations that are too narrow, too obvious, or
poorly aligned with the user’s actual situation.

# Connection to Earlier Projects

This final project was not a separate recommender problem from the
earlier assignments. Rather, it was the next version of the same
Food.com recipe recommender. Project 3 focused on matrix factorization
with a larger sparse ratings subset. Project 4 moved from rating
prediction into top-10 list quality. Project 5 moved the workflow into
Spark and compared local and distributed implementation tradeoffs. The
final project brought those ideas together and added the missing
health-context layer. As a result, the project was cumulative because it
built on the prior technical work while also responding to the
limitation that the original data did not include user health goals,
access constraints, or practical cooking context.

**How the Final Project Builds on Projects 3, 4, and 5**

| Project | Main focus | Scale or data | Evaluation | What it left open |
|:---|:---|:---|:---|:---|
| Project 3 | Matrix factorization for the Food.com utility matrix. | Larger sparse subset with 7,795 users, 8,000 recipes, and 253,120 observed ratings. | RMSE and MAE for held-out rating prediction. | It did not evaluate top-10 recommendation-list usefulness. |
| Project 4 | Accuracy and beyond-accuracy evaluation for top-10 recipe lists. | Same general Food.com recommender context as Project 3. | RMSE, MAE, precision at 10, recall at 10, hit rate at 10, diversity, novelty, and coverage. | It was still a local workflow and did not add health-context profiles. |
| Project 5 | Spark ALS implementation and local-versus-Spark comparison. | Spark-oriented Food.com split with hundreds of thousands of ratings. | RMSE and MAE compared with local baselines on the Project 5 split. | It showed Spark complexity but did not build a final context-aware list. |
| Final Project | Context-aware hybrid recipe recommender. | Full raw positive Food.com interactions where practical, with recipe metadata and synthetic context profiles. | Rating prediction, top-10 relevance, coverage, diversity, novelty, and nutrition/practical alignment. | This is still offline evaluation and not a real user study. |

# Why These Metrics?

**Why Each Metric Was Included**

| Metric group | Metric | Why it is used | What it does not show |
|:---|:---|:---|:---|
| Traditional prediction | RMSE | Ratings are explicit 1-5 values, and RMSE penalizes larger prediction errors more strongly. | It does not show whether a top-10 list is useful, diverse, healthy, or practical. |
| Traditional prediction | MAE | MAE is easier to interpret as the average rating-point error. | It treats all errors linearly and does not evaluate the final recommendation list. |
| Top-N relevance | Precision at 10 | It measures how many recipes in a short recommendation list are relevant. | It can be low when users have few held-out relevant recipes. |
| Top-N relevance | Recall at 10 | It measures how much of the held-out relevant set the recommender recovered. | It depends on how many relevant recipes each user has in the holdout set. |
| Top-N relevance | Hit Rate at 10 | It gives an intuitive check of whether the list found at least one relevant recipe. | It does not distinguish between one hit and many hits. |
| Beyond accuracy | Coverage | Coverage checks whether the recommender uses more of the candidate catalog rather than repeating the same recipes. | High coverage alone does not mean the recipes are relevant. |
| Beyond accuracy | Diversity | Diversity checks whether the top-10 list avoids near-duplicate recipes and varies across nutrition and practicality features. | Diversity alone does not prove the user wants the variety. |
| Beyond accuracy | Novelty | Novelty checks whether the recommender surfaces less obvious, less frequently rated recipes. | Novel recipes may be unfamiliar for good or bad reasons. |
| Context fit | Nutrition/context alignment | This is the main final-project contribution: comparing recommendations against synthetic health-context goals. | It is based on simulated profiles, not real clinical data. |
| Context fit | Practical alignment | Food recommendations should consider time, ingredient burden, and feasibility. | It uses available proxies, not a full measure of affordability or kitchen access. |

**Meaning of Precision, Recall, and Hit Rate at 10**

| Metric | Plain language question | How to read it |
|:---|:---|:---|
| Precision at 10 | Of the ten recipes I showed the user, how many were relevant? | Higher means the short list had a larger share of held-out recipes rated 4 or 5. |
| Recall at 10 | Of the recipes the user liked in the held-out set, how many did I recover? | Higher means the recommender recovered more of the user’s known relevant recipes. |
| Hit Rate at 10 | Did the list contain at least one relevant recipe? | Higher means more users received at least one recommendation that matched the held-out relevant set. |

# Data and Feature Preparation

The raw Food.com interaction file included free-text reviews. Spark’s
CSV parser could be confused by that text, so I first created a clean
modeling file with only the fields needed for recommendation. This was a
data-engineering step rather than a change in the recommendation
question. The final report used the full positive-rating dataset with
the same train-validation split for the local and Spark comparison.

``` r
# Identify the local Food.com files and create a Spark-friendly working folder.
foodcom_data_candidates <- c(
  "/Users/Bertie/Desktop/claude-codex/Data612/Food.com Data",
  "~/Desktop/Food.com Data"
)
desktop_data_dir <- foodcom_data_candidates[
  file.exists(file.path(path.expand(foodcom_data_candidates),
                        "RAW_interactions.csv")) &
    file.exists(file.path(path.expand(foodcom_data_candidates),
                          "RAW_recipes.csv"))
][1]
desktop_data_dir <- path.expand(desktop_data_dir)

raw_interactions_path <- file.path(desktop_data_dir, "RAW_interactions.csv")
raw_recipes_path <- file.path(desktop_data_dir, "RAW_recipes.csv")
spark_work_dir <- "/private/tmp/data612_final_project"
dir.create(spark_work_dir, showWarnings = FALSE, recursive = TRUE)
clean_interactions_path <- file.path(spark_work_dir, "clean_modeling_interactions.csv")

if (is.na(desktop_data_dir) ||
    !file.exists(raw_interactions_path) ||
    !file.exists(raw_recipes_path)) {
  stop("RAW_interactions.csv and RAW_recipes.csv are required.")
}

# Load the raw interaction identifiers and ratings.
raw_interactions_all <- read_csv(raw_interactions_path,
                                 show_col_types = FALSE,
                                 col_select = c(user_id, recipe_id, rating)) %>%
  transmute(user_id = as.integer(user_id),
            recipe_id = as.integer(recipe_id),
            rating = as.numeric(rating))

# Keep only explicit positive ratings for the recommender models.
raw_interactions_full <- raw_interactions_all %>%
  filter(!is.na(user_id),
         !is.na(recipe_id),
         !is.na(rating),
         rating > 0)

# Full mode uses all positive ratings; the development option is retained only for reproducibility.
if (run_full_dataset) {
  modeling_interactions_base <- raw_interactions_full
} else {
  subset_n <- min(subset_rating_limit, nrow(raw_interactions_full))
  modeling_interactions_base <- raw_interactions_full %>%
    slice_sample(n = subset_n, replace = FALSE)
}

# Create stable zero-based indexes for local recommender packages.
user_lookup <- modeling_interactions_base %>%
  distinct(user_id) %>%
  arrange(user_id) %>%
  mutate(user_index = row_number() - 1L)

recipe_lookup <- modeling_interactions_base %>%
  distinct(recipe_id) %>%
  arrange(recipe_id) %>%
  mutate(recipe_index = row_number() - 1L)

modeling_interactions_indexed <- modeling_interactions_base %>%
  left_join(user_lookup, by = "user_id") %>%
  left_join(recipe_lookup, by = "recipe_id")

# Create one reproducible split so local models and Spark use the same rows.
modeling_interactions_split <- modeling_interactions_indexed %>%
  mutate(data_split = if_else(runif(n()) < 0.8, "Train", "Validation"))

training_data <- modeling_interactions_split %>%
  filter(data_split == "Train")

validation_initial <- modeling_interactions_split %>%
  filter(data_split == "Validation")

# Keep validation rows that have known users and recipes in training.
known_training_users <- training_data %>%
  distinct(user_id)

known_training_recipes <- training_data %>%
  distinct(recipe_id)

validation_data <- validation_initial %>%
  semi_join(known_training_users, by = "user_id") %>%
  semi_join(known_training_recipes, by = "recipe_id")

modeling_interactions <- bind_rows(
  training_data,
  validation_data
) %>%
  arrange(data_split, user_id, recipe_id)

# Write the clean same-split interaction file that Spark will read.
write_csv(modeling_interactions %>%
            select(user_id, recipe_id, rating, data_split),
          clean_interactions_path)

# Report the full available data and the final modeling run.
data_summary <- tibble(
  Metric = c("Run mode",
             "Full raw interaction rows available",
             "Full raw positive ratings available",
             "Full positive-rating users available",
             "Full positive-rating recipes available",
             "Final modeling ratings used",
             "Final training ratings",
             "Final validation ratings",
             "Final modeling users",
             "Final modeling recipes",
             "Average positive rating",
             "Clean same-split Spark input file"),
  Value = c(if_else(run_full_dataset,
                    "Full dataset",
                    str_c("Final full-data run: ",
                          format(subset_rating_limit,
                                 big.mark = ",",
                                 scientific = FALSE,
                                 trim = TRUE),
                          " sampled positive ratings")),
            format(nrow(raw_interactions_all), big.mark = ","),
            format(nrow(raw_interactions_full), big.mark = ","),
            format(n_distinct(raw_interactions_full$user_id), big.mark = ","),
            format(n_distinct(raw_interactions_full$recipe_id), big.mark = ","),
            format(nrow(modeling_interactions), big.mark = ","),
            format(nrow(training_data), big.mark = ","),
            format(nrow(validation_data), big.mark = ","),
            format(n_distinct(modeling_interactions$user_id), big.mark = ","),
            format(n_distinct(modeling_interactions$recipe_id), big.mark = ","),
            sprintf("%.3f", mean(modeling_interactions$rating)),
            clean_interactions_path)
)
```

**Food.com Interaction Data Used for Final Project**

| Metric | Value |
|:---|:---|
| Run mode | Full dataset |
| Full raw interaction rows available | 1,132,367 |
| Full raw positive ratings available | 1,071,520 |
| Full positive-rating users available | 196,098 |
| Full positive-rating recipes available | 226,590 |
| Final modeling ratings used | 1,021,554 |
| Final training ratings | 857,222 |
| Final validation ratings | 164,332 |
| Final modeling users | 167,165 |
| Final modeling recipes | 206,392 |
| Average positive rating | 4.667 |
| Clean same-split Spark input file | /private/tmp/data612_final_project/clean_modeling_interactions.csv |

``` r
# Reuse the same report-number formatting style from Projects 3, 4, and 5.
format_report_number <- function(x, digits = 3) {
  format(round(x, digits),
         big.mark = ",",
         scientific = FALSE,
         trim = TRUE,
         nsmall = digits)
}

format_report_percent <- function(x, digits = 1) {
  str_c(format_report_number(100 * x, digits), "%")
}

# Keep predicted ratings inside the Food.com 1-5 rating range.
clip_rating <- function(x) {
  pmin(5, pmax(1, x))
}

# Use simple local metric functions so the same formulas are used throughout.
RMSE <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

MAE <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

# Scale numeric features to 0-1 so profile weights are comparable.
scale_01 <- function(x, higher_is_better = TRUE) {
  x <- as.numeric(x)
  if (all(is.na(x)) || diff(range(x, na.rm = TRUE)) == 0) {
    return(rep(0.5, length(x)))
  }
  scaled <- (x - min(x, na.rm = TRUE)) / diff(range(x, na.rm = TRUE))
  scaled <- pmin(1, pmax(0, scaled))
  if (higher_is_better) scaled else 1 - scaled
}

# Compute average pairwise distance inside a top-10 list as a diversity proxy.
mean_pairwise_diversity <- function(recipe_ids, feature_tbl) {
  feature_matrix <- feature_tbl %>%
    filter(recipe_id %in% recipe_ids) %>%
    arrange(match(recipe_id, recipe_ids)) %>%
    select(calories_scaled, sodium_scaled, sugar_scaled, protein_scaled,
           saturated_fat_scaled, minutes_scaled, ingredients_scaled) %>%
    as.matrix()
  if (nrow(feature_matrix) < 2) {
    return(NA_real_)
  }
  mean(stats::dist(feature_matrix), na.rm = TRUE)
}

# Read recipe metadata that can be used as item-profile features.
recipes_raw <- read_csv(raw_recipes_path,
                        show_col_types = FALSE,
                        col_select = c(name, id, minutes, tags, nutrition,
                                       n_steps, steps, description,
                                       ingredients, n_ingredients))

# Split the Food.com nutrition vector into separate nutrition fields.
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

# Build the recipe feature table used for profile scoring and diversity.
recipe_features <- nutrition_values %>%
  transmute(recipe_id = as.integer(id),
            recipe_name = name,
            minutes = as.numeric(minutes),
            n_steps = as.numeric(n_steps),
            n_ingredients = as.numeric(n_ingredients),
            tags = replace_na(tags, ""),
            ingredients = replace_na(ingredients, ""),
            calories,
            fat,
            sugar,
            sodium,
            protein,
            saturated_fat,
            carbohydrate) %>%
  filter(!is.na(recipe_id),
         !is.na(recipe_name)) %>%
  mutate(recipe_text = str_to_lower(str_c(recipe_name, tags, ingredients, sep = " ")),
         dessert_flag = str_detect(recipe_text, "dessert|cake|cookie|brownie|candy|sweet|pie|frosting|chocolate"),
         vegetarian_flag = str_detect(recipe_text, "vegetarian|vegan|meatless"),
         meat_flag = str_detect(recipe_text, "chicken|beef|pork|bacon|turkey|fish|salmon|shrimp|sausage|ham|lamb|meat"),
         processed_meat_flag = str_detect(recipe_text, "bacon|sausage|ham|pepperoni|salami|hot dog|prosciutto"),
         shellfish_flag = str_detect(recipe_text, "shrimp|crab|lobster|clam|oyster|scallop"),
         specialty_ingredient_flag = str_detect(recipe_text, "saffron|truffle|foie gras|caviar|prosciutto|mascarpone|gruyere|tahini|harissa"),
         calories_scaled = scale_01(calories, higher_is_better = FALSE),
         sodium_scaled = scale_01(sodium, higher_is_better = FALSE),
         sugar_scaled = scale_01(sugar, higher_is_better = FALSE),
         protein_scaled = scale_01(protein, higher_is_better = TRUE),
         saturated_fat_scaled = scale_01(saturated_fat, higher_is_better = FALSE),
         minutes_scaled = scale_01(minutes, higher_is_better = FALSE),
         steps_scaled = scale_01(n_steps, higher_is_better = FALSE),
         ingredients_scaled = scale_01(n_ingredients, higher_is_better = FALSE),
         protein_per_calorie = protein / pmax(calories, 1),
         protein_per_calorie_scaled = scale_01(protein_per_calorie, higher_is_better = TRUE))
```

**Recipe Metadata Summary**

| Recipes with metadata | Median minutes | Median ingredients | Median calories |
|:----------------------|:---------------|:-------------------|:----------------|
| 231,636.00            | 40.00          | 9.00               | 313.40          |

# Guideline-Informed Synthetic Profiles

I created synthetic profiles that represent health and practical
contexts. The profiles are informed by the Dietary Guidelines for
Americans, but they are not clinical profiles and they are not real
patients. Moreover, I used LLM assistance to draft more realistic
scenarios and then simplified those scenarios so they stayed compatible
with the Food.com variables. The important change from a simple
diet-preference table is that the profiles include practical context
such as avoided ingredients, budget, time, kitchen access, cultural
preference, and weighting choices. As a result, the context layer is
meant to approximate the kinds of constraints that are often present in
real food decisions, even though the underlying data do not directly
observe those constraints.

``` r
# Define simulated user contexts that Food.com does not provide directly.
synthetic_profiles <- tibble::tribble(
  ~profile, ~profile_type, ~scenario, ~nutrition_targets, ~allergies_or_avoided_ingredients, ~budget_access_context, ~time_constraint, ~kitchen_access, ~cultural_preference, ~preference_weights,
  "Heart-conscious",
  "Health-aware",
  "A user trying to reduce sodium and saturated fat while still finding realistic meals.",
  "Lower sodium; lower saturated fat; moderate calories.",
  "Avoid bacon, sausage, ham, and heavily processed meat terms when possible.",
  "Moderate cost; avoid recipes with very high ingredient burden.",
  "Prefer recipes under about one hour when similar options exist.",
  "Standard kitchen access, but simpler preparation is preferred.",
  "No specific cultural preference.",
  "Preference 0.50; nutrition 0.25; practical 0.15; restriction fit 0.10.",
  "Lower-sugar",
  "Health-aware",
  "A user trying to avoid dessert-heavy recommendations and reduce sugar.",
  "Lower sugar; avoid dessert-heavy recipes; preserve reasonable protein.",
  "Avoid dessert, candy, frosting, chocolate-heavy, and syrup-heavy patterns.",
  "Moderate cost; no special budget constraint.",
  "Flexible time, but avoid highly burdensome recipes.",
  "Standard kitchen access.",
  "No specific cultural preference.",
  "Preference 0.50; nutrition 0.25; practical 0.10; restriction fit 0.15.",
  "High-protein practical meals",
  "Performance and practicality",
  "A user looking for protein-forward meals that are not too complicated.",
  "Higher protein; higher protein per calorie; moderate calories.",
  "Avoid very dessert-heavy recipes when possible.",
  "Moderate budget; avoid recipes with many specialty ingredients.",
  "Prefer recipes that fit a normal weeknight or meal-prep window.",
  "Standard kitchen access.",
  "No specific cultural preference.",
  "Preference 0.50; nutrition 0.30; practical 0.15; restriction fit 0.05.",
  "Quick weeknight meals",
  "Practical constraint",
  "A user with limited cooking time after work.",
  "Avoid extreme sodium or sugar values, but practicality is the main goal.",
  "No allergy-style restriction.",
  "Moderate budget; fewer ingredients preferred.",
  "Strong preference for shorter cooking time.",
  "Limited weeknight kitchen time and lower tolerance for many steps.",
  "No specific cultural preference.",
  "Preference 0.50; nutrition 0.10; practical 0.35; restriction fit 0.05.",
  "Budget and access aware",
  "Access-aware",
  "A user who benefits from simpler recipes with fewer ingredients and less specialized preparation.",
  "Balanced general nutrition; avoid extreme sodium or sugar when possible.",
  "Avoid highly specialized or hard-to-source ingredients when possible.",
  "Strong preference for fewer ingredients and simple preparation.",
  "Prefer shorter recipes that do not require a long cooking window.",
  "Basic kitchen access; avoid complicated step burden.",
  "Flexible cultural preference, with emphasis on familiar pantry-style ingredients.",
  "Preference 0.45; nutrition 0.10; practical 0.35; restriction fit 0.10.",
  "Vegetarian preference",
  "Dietary preference",
  "A user who prefers vegetarian-style recipes while still wanting variety and balanced nutrition.",
  "Balanced nutrition within a vegetarian-style preference.",
  "Avoid meat-focused ingredients such as chicken, beef, pork, bacon, fish, shrimp, ham, and sausage.",
  "Moderate budget; no special cost constraint.",
  "Flexible time, but shorter options are preferred when scores are close.",
  "Standard kitchen access.",
  "Vegetarian-style preference.",
  "Preference 0.50; nutrition 0.15; practical 0.10; restriction fit 0.25."
)

# Add numeric weights used by the reranker. These mirror the text description.
synthetic_profiles <- synthetic_profiles %>%
  mutate(preference_weight = case_when(
    profile == "Budget and access aware" ~ 0.45,
    TRUE ~ 0.50
  ),
  nutrition_weight = case_when(
    profile == "Heart-conscious" ~ 0.25,
    profile == "Lower-sugar" ~ 0.25,
    profile == "High-protein practical meals" ~ 0.30,
    profile == "Quick weeknight meals" ~ 0.10,
    profile == "Budget and access aware" ~ 0.10,
    profile == "Vegetarian preference" ~ 0.15,
    TRUE ~ 0.20
  ),
  practical_weight = case_when(
    profile == "Heart-conscious" ~ 0.15,
    profile == "Lower-sugar" ~ 0.10,
    profile == "High-protein practical meals" ~ 0.15,
    profile == "Quick weeknight meals" ~ 0.35,
    profile == "Budget and access aware" ~ 0.35,
    profile == "Vegetarian preference" ~ 0.10,
    TRUE ~ 0.20
  ),
  restriction_weight = case_when(
    profile == "Lower-sugar" ~ 0.15,
    profile == "Vegetarian preference" ~ 0.25,
    profile == "Budget and access aware" ~ 0.10,
    profile == "Heart-conscious" ~ 0.10,
    profile == "High-protein practical meals" ~ 0.05,
    profile == "Quick weeknight meals" ~ 0.05,
    TRUE ~ 0.10
  ))
```

**Guideline-Informed Synthetic Profile Context Data**

| Profile | Profile type | Scenario | Nutrition targets | Allergies or avoided ingredients | Budget/access context | Time constraint | Kitchen access | Cultural preference | Preference weights |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Heart-conscious | Health-aware | A user trying to reduce sodium and saturated fat while still finding realistic meals. | Lower sodium; lower saturated fat; moderate calories. | Avoid bacon, sausage, ham, and heavily processed meat terms when possible. | Moderate cost; avoid recipes with very high ingredient burden. | Prefer recipes under about one hour when similar options exist. | Standard kitchen access, but simpler preparation is preferred. | No specific cultural preference. | Preference 0.50; nutrition 0.25; practical 0.15; restriction fit 0.10. |
| Lower-sugar | Health-aware | A user trying to avoid dessert-heavy recommendations and reduce sugar. | Lower sugar; avoid dessert-heavy recipes; preserve reasonable protein. | Avoid dessert, candy, frosting, chocolate-heavy, and syrup-heavy patterns. | Moderate cost; no special budget constraint. | Flexible time, but avoid highly burdensome recipes. | Standard kitchen access. | No specific cultural preference. | Preference 0.50; nutrition 0.25; practical 0.10; restriction fit 0.15. |
| High-protein practical meals | Performance and practicality | A user looking for protein-forward meals that are not too complicated. | Higher protein; higher protein per calorie; moderate calories. | Avoid very dessert-heavy recipes when possible. | Moderate budget; avoid recipes with many specialty ingredients. | Prefer recipes that fit a normal weeknight or meal-prep window. | Standard kitchen access. | No specific cultural preference. | Preference 0.50; nutrition 0.30; practical 0.15; restriction fit 0.05. |
| Quick weeknight meals | Practical constraint | A user with limited cooking time after work. | Avoid extreme sodium or sugar values, but practicality is the main goal. | No allergy-style restriction. | Moderate budget; fewer ingredients preferred. | Strong preference for shorter cooking time. | Limited weeknight kitchen time and lower tolerance for many steps. | No specific cultural preference. | Preference 0.50; nutrition 0.10; practical 0.35; restriction fit 0.05. |
| Budget and access aware | Access-aware | A user who benefits from simpler recipes with fewer ingredients and less specialized preparation. | Balanced general nutrition; avoid extreme sodium or sugar when possible. | Avoid highly specialized or hard-to-source ingredients when possible. | Strong preference for fewer ingredients and simple preparation. | Prefer shorter recipes that do not require a long cooking window. | Basic kitchen access; avoid complicated step burden. | Flexible cultural preference, with emphasis on familiar pantry-style ingredients. | Preference 0.45; nutrition 0.10; practical 0.35; restriction fit 0.10. |
| Vegetarian preference | Dietary preference | A user who prefers vegetarian-style recipes while still wanting variety and balanced nutrition. | Balanced nutrition within a vegetarian-style preference. | Avoid meat-focused ingredients such as chicken, beef, pork, bacon, fish, shrimp, ham, and sausage. | Moderate budget; no special cost constraint. | Flexible time, but shorter options are preferred when scores are close. | Standard kitchen access. | Vegetarian-style preference. | Preference 0.50; nutrition 0.15; practical 0.10; restriction fit 0.25. |

The profile weights were design assumptions rather than values learned
from real Food.com users. I set them so each synthetic profile matched
its scenario. For example, the quick weeknight profile gave more weight
to practical fit, the vegetarian profile gave more weight to restriction
fit, and the high-protein profile gave more weight to nutrition fit. I
kept predicted preference as the largest single part of most profiles
because the recommender still needed to suggest recipes a user might
want to eat.

In a real application, these weights would likely be user-controlled or
learned from user feedback. A person could decide that dietary
restrictions matter more than taste preference, or that time and budget
matter more on weekdays than weekends. This would make the recommender
more transparent because the user could see how much the system was
prioritizing preference, nutrition, practicality, and restrictions.

# Local Benchmark Models

Before running Spark, I ran local benchmark models on the same full-data
split. This kept the comparison fair because the local and Spark models
saw the same training rows and were evaluated on the same validation
rows. I also tested a local recommenderlab ALS because it was closer to
Spark ALS than the recosystem matrix factorization model. The model
object was created, but the full rating matrix prediction did not
complete. It stopped after about 3 seconds with a LAPACK singular-system
error. Consequently, this comparison was not only about accuracy; it was
also a practical test of whether the added complexity of Spark was
justified by scale.

``` r
# Build a local bias baseline using global, user, and recipe average-rating effects.
local_baseline_start <- Sys.time()
local_global_mean <- mean(training_data$rating, na.rm = TRUE)

local_user_bias <- training_data %>%
  group_by(user_id) %>%
  summarize(user_bias = mean(rating - local_global_mean, na.rm = TRUE),
            .groups = "drop")

local_item_bias <- training_data %>%
  group_by(recipe_id) %>%
  summarize(item_bias = mean(rating - local_global_mean, na.rm = TRUE),
            .groups = "drop")

local_baseline_predictions <- validation_data %>%
  left_join(local_user_bias, by = "user_id") %>%
  left_join(local_item_bias, by = "recipe_id") %>%
  mutate(user_bias = replace_na(user_bias, 0),
         item_bias = replace_na(item_bias, 0),
         prediction = clip_rating(local_global_mean + user_bias + item_bias))

local_baseline_runtime <- as.numeric(difftime(Sys.time(),
                                              local_baseline_start,
                                              units = "secs"))

local_baseline_results <- tibble(
  Model = "Local bias baseline",
  Factors = NA_real_,
  RMSE = RMSE(local_baseline_predictions$rating,
              local_baseline_predictions$prediction),
  MAE = MAE(local_baseline_predictions$rating,
            local_baseline_predictions$prediction),
  `Run time seconds` = local_baseline_runtime,
  `Validation predictions` = nrow(local_baseline_predictions),
  Status = "Completed"
)

# Fit a local recosystem matrix factorization model on the same split.
fit_local_recosystem <- function(factor_count = 20) {
  tryCatch({
    train_reco_data <- data_memory(user_index = training_data$user_index,
                                   item_index = training_data$recipe_index,
                                   rating = training_data$rating,
                                   index1 = FALSE)

    validation_reco_data <- data_memory(user_index = validation_data$user_index,
                                        item_index = validation_data$recipe_index,
                                        rating = validation_data$rating,
                                        index1 = FALSE)

    model <- Reco()
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

    predictions <- clip_rating(model$predict(validation_reco_data, out_memory()))

    tibble(Model = "Local recosystem matrix factorization",
           Factors = factor_count,
           RMSE = RMSE(validation_data$rating, predictions),
           MAE = MAE(validation_data$rating, predictions),
           `Run time seconds` = unname(train_time["elapsed"]),
           `Validation predictions` = length(predictions),
           Status = "Completed")
  }, error = function(e) {
    tibble(Model = "Local recosystem matrix factorization",
           Factors = factor_count,
           RMSE = NA_real_,
           MAE = NA_real_,
           `Run time seconds` = NA_real_,
           `Validation predictions` = nrow(validation_data),
           Status = str_c("Failed: ", conditionMessage(e)))
  })
}

local_recosystem_results <- fit_local_recosystem(20)

# Add the observed full local recommenderlab ALS stress-test result.
# This stress test was run outside the Rmd so the final render would not hang.
local_als_stress_result_path <- "/Users/Bertie/Desktop/claude-codex/Data612/data612_local_als_full_stress_test_results.csv"

if (file.exists(local_als_stress_result_path)) {
  local_als_stress_result <- read_csv(local_als_stress_result_path,
                                      show_col_types = FALSE)

  local_recommenderlab_als_results <- local_als_stress_result %>%
    transmute(Model = "Local recommenderlab ALS",
              Factors = as.numeric(factors),
              RMSE = as.numeric(RMSE),
              MAE = as.numeric(MAE),
              `Run time seconds` = as.numeric(runtime_seconds),
              `Validation predictions` = as.numeric(validation_predictions),
              Status = "Not completed: full rating matrix prediction was attempted and stopped with a LAPACK singular-system error.")
} else {
  local_recommenderlab_als_results <- tibble(
    Model = "Local recommenderlab ALS",
    Factors = 20,
    RMSE = NA_real_,
    MAE = NA_real_,
    `Run time seconds` = NA_real_,
    `Validation predictions` = NA_real_,
    Status = "Not completed: separate full-data stress-test result file was not found."
  )
}

local_model_results <- bind_rows(local_baseline_results,
                                 local_recosystem_results,
                                 local_recommenderlab_als_results)
```

**Full-data Same-Split Local Benchmark Results**

| Model | Factors | RMSE | MAE | Run time seconds | Validation predictions | Status |
|:---|:---|:---|:---|:---|:---|:---|
| Local bias baseline | NA | 0.686 | 0.400 | 2.839 | 164,332 | Completed |
| Local recosystem matrix factorization | 20 | 0.884 | 0.610 | 0.413 | 164,332 | Completed |
| Local recommenderlab ALS | 20 | Not completed | Not completed | 3.025 | Not completed | Not completed: full rating matrix prediction was attempted and stopped with a LAPACK singular-system error. |

# Spark ALS Model

Spark ALS was trained on the same clean interaction split used by the
local benchmark models. To make the comparison more balanced, ran Spark
ALS with the same comparable settings used by the local matrix
factorization models: 20 latent factors, 10 training iterations, and
0.10 regularization. The added recommenderlab ALS check made the local
comparison conceptually closer to Spark ALS. However, the separate
full-data local ALS attempt stopped during prediction before it could
produce RMSE or MAE, while Spark ALS completed the full validation
workflow.

``` python
import csv
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Start one local Spark application for the final project workflow.
spark = (
    SparkSession.builder
    .appName("data612-final-context-aware-foodcom")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Set shared file paths for Spark outputs that R will read later.
clean_path = "/private/tmp/data612_final_project/clean_modeling_interactions.csv"
out_dir = "/private/tmp/data612_final_project"
metrics_path = os.path.join(out_dir, "spark_model_metrics.csv")
recs_path = os.path.join(out_dir, "spark_candidate_recommendations.csv")
heldout_path = os.path.join(out_dir, "spark_heldout_relevant.csv")
eval_setup_path = os.path.join(out_dir, "spark_eval_setup.csv")

# Load explicit positive ratings as the sparse utility-matrix entries.
ratings = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(clean_path)
    .select(
        F.col("user_id").cast("int").alias("user_id"),
        F.col("recipe_id").cast("int").alias("recipe_id"),
        F.col("rating").cast("double").alias("rating"),
        F.col("data_split").cast("string").alias("data_split"),
    )
    .where((F.col("rating") > 0) & F.col("user_id").isNotNull() & F.col("recipe_id").isNotNull())
)

# Use the same holdout split created in R for the local benchmark models.
train = ratings.where(F.col("data_split") == "Train").drop("data_split")
validation = ratings.where(F.col("data_split") == "Validation").drop("data_split")
train = train.cache()
validation = validation.cache()
train_rows = train.count()
validation_rows = validation.count()

# Build a simple bias baseline from global, user, and item average-rating effects.
global_mean = train.agg(F.avg("rating").alias("global_mean")).collect()[0]["global_mean"]
user_bias = (
    train.groupBy("user_id")
    .agg(F.avg(F.col("rating") - F.lit(global_mean)).alias("user_bias"))
)
item_bias = (
    train.groupBy("recipe_id")
    .agg(F.avg(F.col("rating") - F.lit(global_mean)).alias("item_bias"))
)
baseline_start = time.time()
baseline_predictions = (
    validation.join(user_bias, "user_id", "left")
    .join(item_bias, "recipe_id", "left")
    .fillna({"user_bias": 0.0, "item_bias": 0.0})
    .withColumn("prediction", F.least(F.lit(5.0), F.greatest(F.lit(1.0), F.lit(global_mean) + F.col("user_bias") + F.col("item_bias"))))
)
baseline_runtime_seconds = time.time() - baseline_start

RMSE_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
MAE_eval = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
baseline_RMSE = RMSE_eval.evaluate(baseline_predictions)
baseline_MAE = MAE_eval.evaluate(baseline_predictions)

# Train Spark MLlib ALS as the collaborative-filtering candidate generator.
als = ALS(
    userCol="user_id",
    itemCol="recipe_id",
    ratingCol="rating",
    rank=20,
    maxIter=10,
    regParam=0.10,
    coldStartStrategy="drop",
    implicitPrefs=False,
    seed=612,
)

# Fit ALS and evaluate held-out rating prediction error.
start = time.time()
model = als.fit(train)
runtime_seconds = time.time() - start
als_predictions = (
    model.transform(validation)
    .withColumn("prediction", F.least(F.lit(5.0), F.greatest(F.lit(1.0), F.col("prediction"))))
    .cache()
)
als_prediction_rows = als_predictions.count()
als_RMSE = RMSE_eval.evaluate(als_predictions)
als_MAE = MAE_eval.evaluate(als_predictions)

# Select evaluation users who have enough history and at least one relevant holdout.
train_user_counts = train.groupBy("user_id").agg(F.count("*").alias("train_count"))
relevant_by_user = (
    validation.where(F.col("rating") >= 4)
    .groupBy("user_id")
    .agg(F.countDistinct("recipe_id").alias("relevant_count"))
)
available_users = model.userFactors.select(F.col("id").alias("user_id"))
eval_users = (
    relevant_by_user.join(train_user_counts, "user_id")
    .join(available_users, "user_id")
    .where((F.col("relevant_count") >= 1) & (F.col("train_count") >= 5))
    .orderBy(F.desc("relevant_count"), F.desc("train_count"))
    .limit(30)
    .cache()
)
eval_user_count = eval_users.count()

# Held-out ratings of 4 or 5 are treated as relevant for top-10 evaluation.
heldout_relevant = (
    validation.where(F.col("rating") >= 4)
    .join(eval_users.select("user_id"), "user_id")
    .select("user_id", "recipe_id")
    .distinct()
    .cache()
)

# Estimate item popularity from training data for the novelty calculation.
item_popularity = train.groupBy("recipe_id").agg(F.count("*").alias("training_rating_count"))
seen_training = train.select("user_id", "recipe_id").distinct()

# Ask ALS for a larger candidate set, then remove recipes already seen in training.
recommendations = model.recommendForUserSubset(eval_users.select("user_id"), 200)
candidate_recs = (
    recommendations
    .select("user_id", F.explode("recommendations").alias("rec"))
    .select("user_id",
            F.col("rec.recipe_id").cast("int").alias("recipe_id"),
            F.col("rec.rating").cast("double").alias("als_score"))
    .join(seen_training, ["user_id", "recipe_id"], "left_anti")
    .join(item_popularity, "recipe_id", "left")
    .fillna({"training_rating_count": 0})
)
rank_window = Window.partitionBy("user_id").orderBy(F.desc("als_score"))
candidate_recs = (
    candidate_recs
    .withColumn("candidate_rank", F.row_number().over(rank_window))
    .where(F.col("candidate_rank") <= 100)
    .orderBy("user_id", "candidate_rank")
    .cache()
)
candidate_rows = candidate_recs.count()
training_user_count = train.select("user_id").distinct().count()
training_recipe_count = train.select("recipe_id").distinct().count()

# Write compact outputs so the visible R chunks can build the final report tables.
with open(metrics_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "rank", "max_iter", "reg_param", "RMSE", "MAE", "runtime_seconds", "validation_predictions", "status"])
    writer.writerow(["Spark ALS", 20, 10, 0.10, als_RMSE, als_MAE, runtime_seconds, als_prediction_rows, "Completed"])
```

``` python
with open(eval_setup_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerow(["training_rows", train_rows])
    writer.writerow(["validation_rows", validation_rows])
    writer.writerow(["evaluation_users", eval_user_count])
    writer.writerow(["candidate_recommendation_rows", candidate_rows])
    writer.writerow(["total_training_ratings", train_rows])
    writer.writerow(["training_user_count", training_user_count])
    writer.writerow(["training_recipe_count", training_recipe_count])
```

``` python
with open(recs_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "recipe_id", "als_score", "training_rating_count", "candidate_rank"])
    for row in candidate_recs.collect():
        writer.writerow([row["user_id"], row["recipe_id"], row["als_score"], row["training_rating_count"], row["candidate_rank"]])
```

``` python
with open(heldout_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["user_id", "recipe_id"])
    for row in heldout_relevant.collect():
        writer.writerow([row["user_id"], row["recipe_id"]])
```

``` python

spark.stop()
```

**Full-data Spark ALS Rating Prediction Results**

| Model | Rank | Max iter | Reg param | RMSE | MAE | Run time seconds | Validation predictions | Status |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Spark ALS | 20 | 10 | 0.10 | 1.005 | 0.673 | 17.938 | 164,332 | Completed |

For the final version, Spark ALS provided the personalized candidate
lists that were evaluated with top-N and beyond-accuracy metrics,
including coverage, diversity, novelty, and context fit. Therefore, the
values were read as the result of the full-data workflow rather than as
a separate historical comparison.

**Spark Top-N Evaluation Setup**

| Metric                        | Value   |
|:------------------------------|:--------|
| Training rows                 | 857,222 |
| Validation rows               | 164,332 |
| Evaluation users              | 30      |
| Candidate recommendation rows | 2,800   |
| Total training ratings        | 857,222 |
| Training user count           | 167,165 |
| Training recipe count         | 206,392 |

Compared with Projects 4 and 5, the final project was designed to
support the largest practical Food.com interaction set from this
sequence. The table below showed the final full-data run alongside the
earlier project sizes.

**Project 4, Project 5, and Final Project Modeling Size Comparison**

| Metric                     | Project 4 | Project 5 | Final project |
|:---------------------------|:----------|:----------|:--------------|
| Users                      | 7,795     | 24,875    | 167,165       |
| Recipes                    | 8,000     | 146,257   | 206,392       |
| Observed/modeling ratings  | 253,120   | 698,901   | 1,021,554     |
| Training ratings           | 202,496   | 559,121   | 857,222       |
| Test or validation ratings | 50,624    | 123,143   | 164,332       |

The utility matrix was still extremely sparse. This was normal for
recommender systems because most users only rated a small fraction of
all available recipes. Missing entries should be treated as unknown
preferences, not dislikes. Otherwise, the model would incorrectly assume
that a user rejected thousands of recipes simply because the user never
rated them.

**Training Utility Matrix Sparsity**

| Metric                           | Value          |
|:---------------------------------|:---------------|
| Possible user-recipe pairs       | 34,501,518,680 |
| Possible training matrix entries | 34,501,518,680 |
| Observed training ratings        | 857,222        |
| Observed share                   | 0.0025%        |
| Sparsity                         | 99.9975%       |

# Local Versus Spark Comparison

Project 5 showed that Spark was a scalability tool, not an automatic
accuracy improvement. Ultimately, the comparison helped determine
whether Spark was needed because the workload had become too large for
local tools, not because Spark should be assumed to perform better by
default.

**Full-data Same-Split Local and Spark Results**

| Model | Factors | RMSE | MAE | Run time seconds | Validation predictions | Status |
|:---|:---|:---|:---|:---|:---|:---|
| Local bias baseline | NA | 0.686 | 0.400 | 2.839 | 164,332 | Completed |
| Local recosystem matrix factorization | 20 | 0.884 | 0.610 | 0.413 | 164,332 | Completed |
| Local recommenderlab ALS | 20 | Not completed | Not completed | 3.025 | Not completed | Not completed: full rating matrix prediction was attempted and stopped with a LAPACK singular-system error. |
| Spark ALS | 20 | 1.005 | 0.673 | 17.938 | 164,332 | Completed |

The factor count was NA for the bias baseline because that model did not
learn latent factors. The run time was reported for the completed
models, and any model that did not complete was kept in the table so the
limitation was visible. The Spark ALS rerun used the same comparable
parameter values as the local factor models: 20 factors, 10 iterations,
and 0.10 regularization. Adding local recommenderlab ALS made the
comparison more direct because it tested a local ALS-style method rather
than only comparing Spark ALS to recosystem. In the separate full-data
stress test, the recommenderlab model object was created, and the full
rating matrix prediction was attempted. The attempt stopped after about
3 seconds with a LAPACK singular-system error, so RMSE and MAE were not
available for that model. The ratings were also highly positive, which
helped the simple local bias baseline perform well by learning the
overall tendency for users to give high ratings. Therefore, the
comparison was fairer after adding the ALS check, but it still showed
that Spark was the practical tool for the full-data recommender
workflow.

**Local versus Spark Implementation Tradeoffs**

| Dimension | Local R approach | Spark ALS approach |
|:---|:---|:---|
| Setup | Simpler R package workflow used in earlier projects. | Requires Java, PySpark, a Spark session, and Spark DataFrames. |
| Data scale | Works well for sampled or laptop-scale sparse rating triples. | Better fit when the project uses the largest practical raw interaction file. |
| Main algorithm role | Good for baselines, fast matrix factorization experiments, and checking whether a local ALS-style model is practical. | Used here as the collaborative-filtering candidate generator. |
| Strength | Fast iteration and easier debugging. | Scalable joins, distributed ALS, and batch top-N candidate generation. |
| Weakness | Can run into local memory or runtime limits as data grows. | More operational complexity and not guaranteed to beat simple baselines. |
| Final project use | Used conceptually as the comparison point from Projects 3, 4, and 5. | Used directly for the final candidate lists that are reranked with context. |

# Context-Aware Reranking

For each synthetic profile, I scored the ALS candidate recipes using
available recipe metadata. The final hybrid score combined predicted
preference and profile fit. I weighted predicted preference slightly
more heavily because the system still needed to recommend recipes the
user was likely to enjoy. However, the profile weight was large enough
for the context layer to change the list when a recipe was a better fit
for the user’s nutrition, access, or practicality context. In turn, this
made the reranking step a transparent way to examine the tradeoff
between preference and context.

``` r
# Join ALS candidates to recipe item-profile features.
candidates <- candidate_recs %>%
  left_join(recipe_features, by = "recipe_id") %>%
  filter(!is.na(recipe_name)) %>%
  mutate(als_scaled = scale_01(als_score, higher_is_better = TRUE),
         novelty = -log2((training_rating_count + 1) /
                           (as.numeric(spark_eval_setup$value[spark_eval_setup$metric == "total_training_ratings"]) +
                              as.numeric(spark_eval_setup$value[spark_eval_setup$metric == "training_recipe_count"]))))

# Score each candidate recipe against one synthetic profile.
score_profile <- function(candidate_tbl, profile_name) {
  profile_weights <- synthetic_profiles %>%
    filter(profile == profile_name) %>%
    slice(1)

  preference_weight <- profile_weights$preference_weight
  nutrition_weight <- profile_weights$nutrition_weight
  practical_weight <- profile_weights$practical_weight
  restriction_weight <- profile_weights$restriction_weight

  candidate_tbl <- candidate_tbl %>%
    mutate(profile = profile_name,
           nutrition_fit = case_when(
             profile_name == "Heart-conscious" ~
               0.45 * sodium_scaled +
               0.35 * saturated_fat_scaled +
               0.20 * calories_scaled,
             profile_name == "Lower-sugar" ~
               0.55 * sugar_scaled +
               0.20 * as.numeric(!dessert_flag) +
               0.15 * protein_scaled +
               0.10 * calories_scaled,
             profile_name == "High-protein practical meals" ~
               0.55 * protein_scaled +
               0.35 * protein_per_calorie_scaled +
               0.10 * calories_scaled,
             profile_name == "Quick weeknight meals" ~
               0.40 * sodium_scaled +
               0.40 * sugar_scaled +
               0.20 * calories_scaled,
             profile_name == "Budget and access aware" ~
               0.35 * sodium_scaled +
               0.35 * sugar_scaled +
               0.30 * calories_scaled,
             profile_name == "Vegetarian preference" ~
               0.25 * sodium_scaled +
               0.25 * sugar_scaled +
               0.25 * protein_scaled +
               0.25 * calories_scaled,
             TRUE ~ 0.5
           ),
           practical_fit = case_when(
             profile_name == "Quick weeknight meals" ~
               0.45 * minutes_scaled + 0.35 * steps_scaled + 0.20 * ingredients_scaled,
             profile_name == "Budget and access aware" ~
               0.45 * ingredients_scaled + 0.35 * steps_scaled + 0.20 * minutes_scaled,
             profile_name == "High-protein practical meals" ~
               0.45 * ingredients_scaled + 0.35 * minutes_scaled + 0.20 * steps_scaled,
             TRUE ~
               0.40 * ingredients_scaled + 0.30 * minutes_scaled + 0.30 * steps_scaled
           ),
           restriction_fit = case_when(
             profile_name == "Heart-conscious" ~ as.numeric(!processed_meat_flag),
             profile_name == "Lower-sugar" ~ as.numeric(!dessert_flag),
             profile_name == "High-protein practical meals" ~ as.numeric(!dessert_flag),
             profile_name == "Budget and access aware" ~ as.numeric(!specialty_ingredient_flag),
             profile_name == "Vegetarian preference" ~ as.numeric(vegetarian_flag | !meat_flag),
             TRUE ~ 1
           ),
           profile_fit = nutrition_weight * nutrition_fit +
             practical_weight * practical_fit +
             restriction_weight * restriction_fit,
           final_score = preference_weight * als_scaled + profile_fit)
  candidate_tbl
}

# The ALS-only policy keeps the original top-10 ALS ranking.
als_only_lists <- candidates %>%
  group_by(user_id) %>%
  arrange(desc(als_score), .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  ungroup() %>%
  mutate(policy = "ALS only",
         profile = "ALS only",
         final_score = als_scaled,
         profile_fit = NA_real_)

# The hybrid policy reranks the ALS candidates with profile-specific context fit.
hybrid_lists <- map_dfr(synthetic_profiles$profile, ~ score_profile(candidates, .x)) %>%
  group_by(user_id, profile) %>%
  arrange(desc(final_score), desc(als_score), .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  ungroup() %>%
  mutate(policy = "Context-aware hybrid")

recommendation_lists <- bind_rows(als_only_lists, hybrid_lists)
```

# Recommendation-List Evaluation

The top-N evaluation treats held-out recipes rated 4 or 5 as relevant.
Since this is an offline evaluation, these metrics are not the same as a
real users. However, they are still useful because they show how the
list changes when context is added. In particular, they allow me to
compare whether the hybrid reranker changes relevance, diversity,
novelty, coverage, and practical alignment in ways that appear
interpretable.

``` r
# Count each user's held-out relevant recipes.
heldout_counts <- heldout_relevant %>%
  group_by(user_id) %>%
  summarize(relevant_count = n_distinct(recipe_id), .groups = "drop")

# Mark which recommended recipes are hits against the held-out relevant set.
recommended_with_hits <- recommendation_lists %>%
  left_join(heldout_relevant %>% mutate(relevant_hit = TRUE),
            by = c("user_id", "recipe_id")) %>%
  mutate(relevant_hit = replace_na(relevant_hit, FALSE))

# Compute user-level top-10 and beyond-accuracy metrics.
user_list_metrics <- recommended_with_hits %>%
  group_by(policy, profile, user_id) %>%
  summarize(list_n = n(),
            hits = sum(relevant_hit),
            mean_predicted_rating = mean(als_score, na.rm = TRUE),
            mean_profile_fit = mean(profile_fit, na.rm = TRUE),
            mean_sodium = mean(sodium, na.rm = TRUE),
            mean_sugar = mean(sugar, na.rm = TRUE),
            mean_saturated_fat = mean(saturated_fat, na.rm = TRUE),
            mean_protein = mean(protein, na.rm = TRUE),
            mean_minutes = mean(minutes, na.rm = TRUE),
            mean_ingredients = mean(n_ingredients, na.rm = TRUE),
            mean_novelty = mean(novelty, na.rm = TRUE),
            diversity = mean_pairwise_diversity(recipe_id, recipe_features),
            .groups = "drop") %>%
  left_join(heldout_counts, by = "user_id") %>%
  mutate(relevant_count = replace_na(relevant_count, 0),
         precision_at_10 = hits / pmax(list_n, 1),
         recall_at_10 = if_else(relevant_count > 0, hits / relevant_count, NA_real_),
         hit_rate_at_10 = as.numeric(hits > 0))

# Coverage is calculated over the candidate catalog, not the full metadata table.
candidate_pool_size <- n_distinct(candidates$recipe_id)

coverage_summary <- recommendation_lists %>%
  group_by(policy, profile) %>%
  summarize(catalog_coverage = n_distinct(recipe_id) / candidate_pool_size,
            .groups = "drop")

# Average user-level metrics to compare ALS-only and context-aware policies.
list_metric_summary <- user_list_metrics %>%
  group_by(policy, profile) %>%
  summarize(`Precision at 10` = mean(precision_at_10, na.rm = TRUE),
            `Recall at 10` = mean(recall_at_10, na.rm = TRUE),
            `Hit Rate at 10` = mean(hit_rate_at_10, na.rm = TRUE),
            `Mean ALS Score` = mean(mean_predicted_rating, na.rm = TRUE),
            `Mean Diversity` = mean(diversity, na.rm = TRUE),
            `Mean Novelty` = mean(mean_novelty, na.rm = TRUE),
            `Mean Sodium` = mean(mean_sodium, na.rm = TRUE),
            `Mean Sugar` = mean(mean_sugar, na.rm = TRUE),
            `Mean Saturated Fat` = mean(mean_saturated_fat, na.rm = TRUE),
            `Mean Protein` = mean(mean_protein, na.rm = TRUE),
            `Mean Minutes` = mean(mean_minutes, na.rm = TRUE),
            `Mean Ingredients` = mean(mean_ingredients, na.rm = TRUE),
            .groups = "drop") %>%
  left_join(coverage_summary, by = c("policy", "profile"))
```

**Top-N and Beyond-Accuracy Metrics**

| Policy | Profile | Precision at 10 | Recall at 10 | Hit rate at 10 | Mean ALS score | Mean diversity | Mean novelty | Mean sodium | Mean sugar | Mean saturated fat | Mean protein | Mean minutes | Mean ingredients | Catalog coverage |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| ALS only | ALS only | 1.0% | 0.0% | 10.0% | 5.259 | 0.108 | 17.567 | 29.070 | 41.180 | 50.603 | 42.747 | 81.360 | 8.750 | 13.8% |
| Context-aware hybrid | Budget and access aware | 0.7% | 0.0% | 6.7% | 5.244 | 0.078 | 17.570 | 17.633 | 46.230 | 31.710 | 26.170 | 56.037 | 6.367 | 13.3% |
| Context-aware hybrid | Heart-conscious | 1.0% | 0.0% | 10.0% | 5.252 | 0.094 | 17.599 | 21.050 | 37.853 | 40.970 | 34.757 | 74.860 | 7.590 | 13.8% |
| Context-aware hybrid | High-protein practical meals | 1.7% | 0.0% | 16.7% | 5.242 | 0.099 | 17.513 | 34.897 | 24.043 | 47.667 | 56.030 | 94.993 | 8.337 | 12.9% |
| Context-aware hybrid | Lower-sugar | 1.3% | 0.0% | 13.3% | 5.245 | 0.101 | 17.534 | 33.813 | 29.840 | 49.033 | 44.563 | 77.090 | 8.553 | 12.8% |
| Context-aware hybrid | Quick weeknight meals | 0.7% | 0.0% | 6.7% | 5.256 | 0.092 | 17.584 | 23.563 | 39.323 | 41.190 | 35.120 | 71.133 | 7.557 | 13.1% |
| Context-aware hybrid | Vegetarian preference | 0.7% | 0.0% | 6.7% | 5.221 | 0.095 | 17.494 | 20.637 | 72.283 | 33.127 | 19.810 | 45.353 | 7.323 | 14.7% |

``` r
# Plot the beyond-accuracy metrics so policy differences are easier to compare.
list_metric_summary %>%
  select(policy, profile, `Mean Diversity`, `Mean Novelty`, catalog_coverage) %>%
  pivot_longer(cols = c(`Mean Diversity`, `Mean Novelty`, catalog_coverage),
               names_to = "Metric",
               values_to = "Value") %>%
  mutate(Profile_Label = if_else(policy == "ALS only", "ALS only", profile),
         Profile_Label = fct_reorder(Profile_Label, Value, .fun = mean, .desc = FALSE)) %>%
  ggplot(aes(x = Profile_Label, y = Value, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  coord_flip() +
  labs(title = "Beyond-Accuracy Metrics by Recommendation Policy",
       x = NULL,
       y = "Metric Value",
       fill = "Metric") +
  theme_minimal()
```

![](Final-Project--Context-Aware-Hybrid-Recipe-Recommender_files/figure-gfm/metric-plot-1.png)<!-- -->

# Example Recommendations

The tables below show example recommendations for three context-aware
profiles. This was a qualitative check. Precision, recall, hit rate, ALS
score, diversity, novelty, nutrition, practicality, and coverage were
list-level metrics, so I reported them in a small profile-level table
before showing the individual recipe examples. The recipe table then
focused on the recipe names and features that made the lists easier to
interpret.

``` r
# Pick one evaluation user and show a small qualitative recommendation example.
example_user <- recommendation_lists %>%
  count(user_id, sort = TRUE) %>%
  slice(1) %>%
  pull(user_id)

example_profiles <- c("Heart-conscious", "Quick weeknight meals", "High-protein practical meals")
example_metric_profiles <- c("ALS only", example_profiles)
```

**Example Profile-Level Metrics**

| Profile | Precision at 10 | Recall at 10 | Hit rate at 10 | Mean ALS score | Mean diversity | Mean novelty | Mean sodium | Mean sugar | Mean saturated fat | Mean protein | Mean minutes | Mean ingredients | Catalog coverage |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| ALS only | 1.0% | 0.0% | 10.0% | 5.259 | 0.108 | 17.567 | 29.070 | 41.180 | 50.603 | 42.747 | 81.360 | 8.750 | 13.8% |
| Heart-conscious | 1.0% | 0.0% | 10.0% | 5.252 | 0.094 | 17.599 | 21.050 | 37.853 | 40.970 | 34.757 | 74.860 | 7.590 | 13.8% |
| Quick weeknight meals | 0.7% | 0.0% | 6.7% | 5.256 | 0.092 | 17.584 | 23.563 | 39.323 | 41.190 | 35.120 | 71.133 | 7.557 | 13.1% |
| High-protein practical meals | 1.7% | 0.0% | 16.7% | 5.242 | 0.099 | 17.513 | 34.897 | 24.043 | 47.667 | 56.030 | 94.993 | 8.337 | 12.9% |

**Example Recipe Recommendations**

| Profile | Recipe | ALS score | Calories | Sodium | Sugar | Saturated fat | Protein | Minutes | Number of ingredients | Novelty |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Heart-conscious | mixed berries marnier | 5.463 | 172.9 | 0.0 | 103.0 | 0.0 | 6.0 | 5 | 6 | 18.44 |
| Heart-conscious | meal replacement fruit smoothies | 5.414 | 100.9 | 6.0 | 0.0 | 2.0 | 19.0 | 5 | 5 | 18.44 |
| Heart-conscious | pasta toss with a kick | 5.434 | 107.5 | 0.0 | 25.0 | 5.0 | 4.0 | 30 | 11 | 18.02 |
| Heart-conscious | cheese arepas | 5.456 | 320.3 | 12.0 | 9.0 | 8.0 | 14.0 | 30 | 8 | 18.02 |
| Heart-conscious | quinoa super tacos | 5.460 | 695.2 | 44.0 | 15.0 | 9.0 | 58.0 | 20 | 9 | 18.02 |
| Heart-conscious | wee men | 5.528 | 194.0 | 1.0 | 105.0 | 15.0 | 3.0 | 30 | 3 | 17.70 |
| Heart-conscious | grilled salmon with vegetables | 5.464 | 173.3 | 8.0 | 2.0 | 20.0 | 18.0 | 60 | 14 | 18.02 |
| Heart-conscious | stewed tomatoes jefferson | 5.409 | 126.4 | 3.0 | 21.0 | 27.0 | 3.0 | 20 | 6 | 16.85 |
| Heart-conscious | crock pot tacos | 5.583 | 347.7 | 48.0 | 21.0 | 52.0 | 45.0 | 130 | 8 | 17.70 |
| Heart-conscious | salmon penne | 5.697 | 1,185.7 | 13.0 | 1.0 | 155.0 | 103.0 | 45 | 9 | 18.02 |
| High-protein practical meals | wee men | 5.528 | 194.0 | 1.0 | 105.0 | 15.0 | 3.0 | 30 | 3 | 17.70 |
| High-protein practical meals | cheese arepas | 5.456 | 320.3 | 12.0 | 9.0 | 8.0 | 14.0 | 30 | 8 | 18.02 |
| High-protein practical meals | grilled salmon with vegetables | 5.464 | 173.3 | 8.0 | 2.0 | 20.0 | 18.0 | 60 | 14 | 18.02 |
| High-protein practical meals | meal replacement fruit smoothies | 5.414 | 100.9 | 6.0 | 0.0 | 2.0 | 19.0 | 5 | 5 | 18.44 |
| High-protein practical meals | microwave chili dog wraps | 5.423 | 475.9 | 59.0 | 14.0 | 71.0 | 41.0 | 15 | 5 | 18.02 |
| High-protein practical meals | crock pot tacos | 5.583 | 347.7 | 48.0 | 21.0 | 52.0 | 45.0 | 130 | 8 | 17.70 |
| High-protein practical meals | quinoa super tacos | 5.460 | 695.2 | 44.0 | 15.0 | 9.0 | 58.0 | 20 | 9 | 18.02 |
| High-protein practical meals | new england baked cod in cheese sauce | 5.375 | 414.9 | 12.0 | 4.0 | 77.0 | 75.0 | 60 | 6 | 17.21 |
| High-protein practical meals | salmon penne | 5.697 | 1,185.7 | 13.0 | 1.0 | 155.0 | 103.0 | 45 | 9 | 18.02 |
| High-protein practical meals | steamed clams with chorizo and tomatoes | 5.398 | 893.5 | 164.0 | 28.0 | 70.0 | 179.0 | 40 | 10 | 18.02 |
| Quick weeknight meals | mixed berries marnier | 5.463 | 172.9 | 0.0 | 103.0 | 0.0 | 6.0 | 5 | 6 | 18.44 |
| Quick weeknight meals | meal replacement fruit smoothies | 5.414 | 100.9 | 6.0 | 0.0 | 2.0 | 19.0 | 5 | 5 | 18.44 |
| Quick weeknight meals | microwave chili dog wraps | 5.423 | 475.9 | 59.0 | 14.0 | 71.0 | 41.0 | 15 | 5 | 18.02 |
| Quick weeknight meals | quinoa super tacos | 5.460 | 695.2 | 44.0 | 15.0 | 9.0 | 58.0 | 20 | 9 | 18.02 |
| Quick weeknight meals | wee men | 5.528 | 194.0 | 1.0 | 105.0 | 15.0 | 3.0 | 30 | 3 | 17.70 |
| Quick weeknight meals | cheese arepas | 5.456 | 320.3 | 12.0 | 9.0 | 8.0 | 14.0 | 30 | 8 | 18.02 |
| Quick weeknight meals | pasta toss with a kick | 5.434 | 107.5 | 0.0 | 25.0 | 5.0 | 4.0 | 30 | 11 | 18.02 |
| Quick weeknight meals | salmon penne | 5.697 | 1,185.7 | 13.0 | 1.0 | 155.0 | 103.0 | 45 | 9 | 18.02 |
| Quick weeknight meals | grilled salmon with vegetables | 5.464 | 173.3 | 8.0 | 2.0 | 20.0 | 18.0 | 60 | 14 | 18.02 |
| Quick weeknight meals | crock pot tacos | 5.583 | 347.7 | 48.0 | 21.0 | 52.0 | 45.0 | 130 | 8 | 17.70 |

# Discussion

The final recommender showed why a recipe recommendation system should
not be evaluated with one metric. RMSE and MAE were useful because the
data included explicit ratings, but they only measured rating
prediction. Precision at 10, recall at 10, and hit rate at 10 evaluated
the recommendation list against held-out liked recipes. Coverage,
diversity, and novelty asked a different question: whether the
recommender used the catalog well or fell back to narrow, obvious,
repetitive lists. Overall, the results were interpreted as a set of
tradeoffs rather than a single winner-take-all score.

For this project, novelty was especially important because a recipe
recommender should not only surface the most frequently rated recipes. A
popular recipe may be safe from a prediction standpoint, but it may not
help a user discover useful new options. Diversity was also important
because ten similar recipes could be less useful than a list that gave
the user several realistic choices. Coverage mattered because a
recommender that only used a small part of the catalog could be
technically accurate but limited as a user experience. In a
nutrition-focused setting, these beyond-accuracy measures were also
connected to autonomy because users may benefit from having multiple
feasible options rather than one narrow type of recipe repeated in
different forms.

The context-aware hybrid reranker was not meant to replace user
preference. Instead, it tried to balance predicted preference with
practical health-context fit. In a real application, users should be
able to control these goals directly, especially if recommendations are
connected to food access, health goals, or dietary restrictions. This
project used synthetic profiles because Food.com did not include real
health context, allergies, budget, kitchen access, or cultural
preference data. As a result, the profiles were understood as a
transparent simulation of missing context, not as a substitute for real
user-centered design or clinical validation.

# Limitations

- **Synthetic profiles are not real users:** I used guideline-informed,
  LLM-assisted profiles to make the scenarios realistic, but they are
  still simulated.
- **Food.com nutrition fields are limited:** Some values are percent
  daily values rather than full nutrient measurements.
- **Offline metrics are incomplete:** A real recommender would need
  online testing or user feedback.
- **The holdout is reproducible but not temporal:** Food.com includes
  dates, and a production-style recommender would ideally train on
  earlier interactions and test on later interactions. I kept the random
  80/20 holdout so the final project stayed comparable to the earlier
  course projects.
- **Spark ALS still has cold-start limits:** New users and new recipes
  need content features or onboarding questions.
- **Context weights are transparent but not clinically tuned:** The
  reranking weights are designed for interpretation, not medical
  guidance.

# Sources

Food.com Recipes and Interactions. Kaggle dataset by Shuyang Li.
<https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions>

Apache Spark MLlib Collaborative Filtering Documentation.
<https://spark.apache.org/docs/latest/ml-collaborative-filtering.html>

U.S. Department of Agriculture and U.S. Department of Health and Human
Services. *Dietary Guidelines for Americans, 2020-2025*. 9th Edition.
December 2020.
<https://www.dietaryguidelines.gov/resources/2020-2025-dietary-guidelines-online-materials>

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization
techniques for recommender systems. *Computer*, 42(8), 30-37.
<https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf>

Trattner, C., & Elsweiler, D. (2017). *Food recommender systems:
Important contributions, challenges and future research directions*.
<https://arxiv.org/abs/1711.02760>

Figueroa, C. A., Torkamaan, H., Bhattacharjee, A., Hauptmann, H., Guan,
K. W., & Sedrakyan, G. (2025). Designing health recommender systems to
promote health equity: A socioecological perspective. *Journal of
Medical Internet Research*, 27, e60138. <https://doi.org/10.2196/60138>
