Final Project Planning Document: A Context-Aware Hybrid Recipe
Recommender
================
Robert Gomez, DrPH, MPH
July 2026

- [Introduction](#introduction)
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Why Add Guideline-Informed Health
  Context?](#why-add-guideline-informed-health-context)
- [Proposed Guideline-Informed Synthetic
  Profiles](#proposed-guideline-informed-synthetic-profiles)
- [Recommender Design](#recommender-design)
- [Planned Workflow](#planned-workflow)
- [Evaluation Plan](#evaluation-plan)
- [Expected Challenges](#expected-challenges)
- [Sources](#sources)

# Introduction

For my final project, I plan to build on the Food.com nutrition recipe
recommender that I developed across the earlier assignments. The earlier
projects moved from simple baselines, to content-based filtering, to
matrix factorization, to top-N recommendation evaluation and diversity
reranking. The final project will extend that same line of work by
building a context-aware hybrid recommender that combines user
preference, recipe metadata, and guideline-informed synthetic
health-context profiles. The main goal is to recommend recipes that a
user is likely to enjoy, while also making the recommendation list more
useful for a nutrition-focused setting. A high predicted rating is
helpful, but it is not enough by itself. Food recommendations can affect
everyday behavior, and a recipe that is popular or highly rated may
still be a poor fit for someone who is trying to reduce sodium, limit
added sugar, cook quickly, or increase protein. The Food.com data
includes ratings and recipe metadata, but it does not include individual
user health goals, dietary restrictions, food access, or clinical
context. This final project will address that limitation by adding
guideline-informed synthetic user health profiles and using them to
guide a health-aware reranking step. This project will still be a
recommender systems project, not medical advice or a clinical decision
support system. The synthetic profiles will not represent real patients.
They will be used as a structured way to test how recommendations change
when user context is added to a rating-based recommender.

# Project Goal

The final system will have three main layers:

- **Preference layer:** use Spark ALS to learn from Food.com user-recipe
  ratings and generate candidate recipes that a user is predicted to
  like.

- **Recipe feature layer:** use Food.com metadata, including nutrition
  values, ingredients, tags, cooking time, number of ingredients, and
  steps, to describe each recipe.

- **Context layer:** create guideline-informed synthetic user
  health-context profiles that represent common nutrition and lifestyle
  goals, then use those profiles to rerank candidate recipes.

The practical question for the project is:

> Compared with ALS-only recommendations, can a context-aware hybrid
> recommender preserve predicted user preference while improving
> nutrition and lifestyle fit?

# Dataset

The primary recommender data source will be the Food.com Recipes and
Interactions dataset. I have already used this dataset in earlier
projects, so the final project needs a unique element. The unique
element will be the addition of a public health-context source,
guideline-informed synthetic health profiles, and a hybrid reranking
layer that uses recipe metadata more directly.

| Data_Source | Role_In_Project | Planned_Fields |
|:---|:---|:---|
| Food.com user interactions | Collaborative filtering and rating prediction | user_id, recipe_id, rating, date |
| Food.com recipe metadata | Recipe feature engineering and reranking | recipe name, minutes, tags, nutrition, ingredients, steps |
| Dietary Guidelines for Americans, 2020-2025 | Public contextual source for health-aware profile design | guidance related to sodium, saturated fat, added sugars, calories, and healthy dietary patterns |
| Guideline-informed synthetic health-context profiles | Simulated personalization layer for health-aware reranking | profile type, nutrition targets, allergies or avoided ingredients, budget, time, kitchen access, cultural preference, and preference weights |

Planned Data Sources for the Final Project

For the final project, I plan to move beyond the Project 5 setup by
using the full raw Food.com interaction file where practical. Project 5
used the preprocessed training file, which had 698,901 interaction rows.
The full RAW_interactions.csv file has 1,132,367 rows, including
1,071,520 positive ratings, 226,570 unique users, and 231,637 unique
recipes. That larger file is a better match for the final project
requirement because it crosses the one-million-rating threshold and
creates a clearer reason to use Spark.

| Data_Version | Rows_or_Count | Role_In_Final_Project |
|:---|:---|:---|
| Project 5 preprocessed training interactions | 698,901 rows | Earlier Spark ALS implementation |
| Full RAW_interactions.csv | 1,132,367 rows | Planned final project interaction source |
| Positive ratings in RAW_interactions.csv | 1,071,520 ratings | Main explicit-feedback modeling data |
| Unique users in RAW_interactions.csv | 226,570 users | User side of the recommender problem |
| Unique recipes in RAW_interactions.csv | 231,637 recipes | Item side of the recommender problem |

Planned Scale Increase from Project 5 to the Final Project

Using the full raw interaction data will require one additional
preparation step. Spark ALS expects integer user and item identifiers,
so I will create stable integer indexes for users and recipes before
model training. I may still use smaller samples during development and
tuning so the notebook can be built safely, but the final goal will be
to train and evaluate on the largest practical Food.com interaction set
available on this machine.

# Why Add Guideline-Informed Health Context?

The Food.com ratings show what users liked on a recipe website. They do
not show whether a recipe was appropriate for a user’s health goals,
food access, dietary restrictions, cultural preferences, kitchen access,
budget, allergies, or time constraints. This is an important limitation
because nutrition recommenders should not treat preference and health
fit as the same thing. For example, a user may rate salty foods highly.
A regular recommender may learn that pattern and continue recommending
high-sodium recipes. That may be accurate from a rating-prediction
perspective, but it may not be the best recommendation if the user is
trying to follow a lower-sodium eating pattern. Since the dataset does
not include that kind of context, I will create synthetic user profiles
to simulate it. To keep the profiles grounded, I will use the public
[Dietary Guidelines for Americans,
2020-2025](https://www.dietaryguidelines.gov/resources/2020-2025-dietary-guidelines-online-materials)
as the contextual source. These guidelines are useful for this project
because they discuss healthy dietary patterns and recommend limiting
dietary components such as sodium, saturated fat, and added sugars. I
will use them to define transparent profile logic for reranking recipe
recommendations. The guideline-informed synthetic profiles will be used
for evaluation and demonstration. They will not be described as real
people, real patients, or clinical guidance. Their purpose is to make
the missing context explicit and test how a recommender could respond to
that context. The profiles will include diet-related information, but
they will not only be diet labels. They will also include the practical
and equity-related concerns discussed in the earlier projects. I will
also do my best to make the synthetic profile data realistic. One way I
plan to do that is by using large language models to help draft
plausible user-context scenarios, then reviewing and simplifying those
scenarios so they stay consistent with the public guidelines and with
the limits of the Food.com metadata.

# Proposed Guideline-Informed Synthetic Profiles

The synthetic profiles will represent common recommendation situations.
Each profile will define a set of nutrition, preference, and practical
constraints. These profiles will guide the reranking step after ALS
generates candidate recipes.

| Context_Dimension | How_It_Will_Be_Represented |
|:---|:---|
| Nutrition goals | Lower sodium, lower saturated fat, lower sugar, higher protein, or more balanced overall nutrition |
| Dietary restrictions or avoided ingredients | Synthetic flags for vegetarian preference, common ingredient avoidance, or allergy-style exclusions |
| Time and preparation burden | Limits or penalties based on minutes, number of steps, and number of ingredients |
| Food access and budget proxy | Preference for simpler recipes with fewer ingredients and less specialized preparation |
| Kitchen access | Preference for recipes that appear less equipment- or preparation-intensive |
| Cultural and preference fit | Use tags and ingredients to avoid treating one generic healthy pattern as appropriate for every user |

Planned Health-Context Dimensions

| Profile | Health_Context | Practical_Context | Likely_Reranking_Logic |
|:---|:---|:---|:---|
| Heart-conscious | Lower sodium and saturated fat | Avoid long or complex recipes when similar lower-burden options exist | Use Dietary Guidelines context to penalize high sodium and high saturated fat; favor moderate calories and practical preparation |
| Lower-sugar | Lower sugar with reasonable protein | Avoid dessert-heavy or highly sweet recipe patterns | Use Dietary Guidelines context to penalize high sugar; favor recipes with reasonable protein and non-dessert tags |
| High-protein practical meals | Higher protein and protein per calorie | Prefer recipes that do not require excessive time or ingredients | Reward higher protein while penalizing very long cooking time and high ingredient burden |
| Quick weeknight meals | No specific clinical goal, but still avoid extreme nutrition values | Limited cooking time, fewer steps, and fewer ingredients | Penalize long cooking time, many steps, many ingredients, and extreme sodium or sugar values |
| Budget and access aware | Balanced general nutrition | Prefer simpler recipes with fewer ingredients and less specialized preparation | Favor moderate nutrition values, fewer ingredients, and shorter preparation |
| Vegetarian preference | Balanced nutrition within a vegetarian-style preference | Avoid meat-focused tags and ingredients while preserving variety | Use tags and ingredients to favor vegetarian recipes while still considering nutrition and predicted preference |

Planned Guideline-Informed Synthetic Health-Context Profiles

These profiles are intentionally simple. I want the final project to
show the recommender design clearly rather than create synthetic data
that looks more precise than it really is. The public guidelines provide
context for the profile logic, while the Food.com metadata provides the
actual recipe-level features used in reranking. The profiles will be
useful because they allow the same user-rating model to produce
different recommendation lists under different contexts.

# Recommender Design

The planned recommender will use a mixed approach. Spark ALS will learn
latent user and recipe factors from explicit Food.com ratings. After ALS
produces candidate recommendations, I will apply a hybrid reranking
step. The reranking step will use nutrition and content features from
the recipe data, along with the guideline-informed synthetic
health-context profile. In theory the design is as follows:

| Component | Meaning |
|:---|:---|
| ALS predicted rating | How much the user is expected to like the recipe based on rating history |
| Nutrition fit score | How well the recipe matches the synthetic profile’s nutrition goals |
| Practical fit score | How well the recipe matches cooking time, steps, and ingredient constraints |
| Penalty score | Penalty for recipes that conflict with the profile’s constraints |

Planned Final Recommendation Score Components

The final score will be a weighted combination of predicted preference
and context fit:

``` text
final_score =
  als_predicted_rating +
  nutrition_fit_bonus +
  practical_fit_bonus -
  constraint_penalty
```

The weights will be simple and transparent. The main purpose is to
compare recommendation behavior before and after context is added.

# Planned Workflow

| Step | Description |
|:---|:---|
| 1\. Load data | Load Food.com ratings and recipe metadata. |
| 2\. Prepare ratings | Keep valid user, recipe, and rating fields; create train and validation splits. |
| 3\. Train baseline models | Use a global mean or bias baseline so Spark ALS has a comparison point. |
| 4\. Train Spark ALS | Use Spark MLlib ALS on the user-recipe rating data. |
| 5\. Generate candidate recipes | Create top candidate recommendations for selected users. |
| 6\. Build recipe features | Extract nutrition, cooking time, ingredient, and tag features. |
| 7\. Create guideline-informed synthetic profiles | Use the Dietary Guidelines context to define synthetic nutrition and lifestyle profiles. |
| 8\. Rerank recommendations | Apply profile-specific reranking to ALS candidate lists. |
| 9\. Evaluate results | Compare ALS-only and context-aware hybrid recommendations. |

Planned Final Project Workflow

# Evaluation Plan

The project will use both rating-prediction and recommendation-list
evaluation. This is important because the model should not be judged
only by RMSE. RMSE and MAE tell me whether the model predicts held-out
ratings well. They do not tell me whether the final top-10 list is
diverse, practical, or aligned with a user’s health context.

| Evaluation_Area | Metric_or_Check | Purpose |
|:---|:---|:---|
| Rating prediction | RMSE and MAE | Compare Spark ALS with simple baselines on held-out ratings. |
| Top-N relevance | Precision at 10, recall at 10, hit rate at 10 if feasible | Evaluate whether recommended recipes include held-out liked recipes. |
| Nutrition alignment | Average sodium, sugar, calories, saturated fat, and protein in top-10 lists | Compare ALS-only recommendations with context-aware recommendations. |
| Practical alignment | Average minutes, steps, and number of ingredients | Check whether profiles such as quick weeknight meals change the recommendation list. |
| Diversity and coverage | Recipe feature diversity and catalog coverage | Avoid recommending the same narrow set of recipes repeatedly. |
| Qualitative review | Example recommendation lists for selected users and profiles | Show how recommendations change in a way that can be interpreted. |

Planned Evaluation Approach

The main expected comparison will be ALS-only recommendations versus
hybrid context-aware recommendations. I expect the context-aware lists
may give up a small amount of predicted rating in some cases, but they
should improve alignment with the selected profile. That tradeoff is
part of the point of the project.

# Expected Challenges

There are several limitations I expect to manage:

- **Guideline-informed synthetic data cannot replace real user
  context.** The profiles will help test system behavior, but they will
  not prove that the recommender works for real people with real health
  needs.

- **Nutrition values in Food.com have limitations.** Calories are direct
  values, but several other fields are stored as percent daily value
  fields. They are useful for comparison but should not be treated as
  complete clinical nutrition data.

- **Ratings are not the same as health outcomes.** The system can learn
  preference patterns, but Food.com ratings do not show whether a recipe
  improved health, met dietary restrictions, or was realistic for the
  user.

- **Spark adds operational complexity.** Project 5 showed that Spark is
  useful for learning distributed recommender implementation, but it
  also adds Java, PySpark, session management, and tuning issues.

- **Reranking requires judgment.** If the health-context weights are too
  strong, the system may ignore user preference. If they are too weak,
  the context layer will not matter. I will keep the weighting simple
  and explain the tradeoffs.

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
