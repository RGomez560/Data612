Research Discussion 1
================
Robert Gomez, DrPH, MPH
June 2026

- [<u>**Peloton’s Recommender
  System**</u>](#pelotons-recommender-system)
  - [<u>**Overview**</u>](#overview)
  - [<u>**Design**</u>](#design)
  - [<u>**User Experience**</u>](#user-experience)
- [<u>**Attacks on Recommender
  Systems**</u>](#attacks-on-recommender-systems)
  - [<u>**Overview**</u>](#overview-1)
  - [<u>**Prevention Strategy**</u>](#prevention-strategy)
  - [<u>**Conclusion**</u>](#conclusion)

# <u>**Peloton’s Recommender System**</u>

Articles used:

- Peloton. (2021). *How We Built: An Early-Stage Recommender System*.
  <https://www.onepeloton.com/press/articles/designing-an-early-stage-recommender-system>
- Peloton. (2021). *How We Built: An Early-Stage Machine Learning Model
  for Recommendations*.
  <https://www.onepeloton.com/press/articles/how-we-built-machine-learning>
- Peloton. (2022). *Revamping Peloton Homescreen Experience with
  Personalized Rows*.
  <https://www.onepeloton.com/press/articles/revamping-peloton-homescreen-experience-with-personalized-rows>
- Peloton. (2025). *From Contextual Recommender Systems to a
  Transformer-Based Architecture*.
  <https://careers.onepeloton.com/en/blog/product-and-tech/from-contextual-recommender-systems-to-a-transformer-based-architecture/>
- Peloton. (2025). *How Peloton Creates Personalized Workout Plans Based
  On Your Goals*.
  <https://www.onepeloton.com/en-GB/blog/personalized-workout-plan/>

## <u>**Overview**</u>

Peloton is a fitness platform that uses a commercial recommender system
to surface workout content for its members. The recommendation problem
is more constrained than typical content recommendation because a
suggested workout needs to fit a member’s available time, equipment,
preferred instructors, fitness level, recent training history, and
goals. Based on Peloton’s own technical articles, the system appears to
be a hybrid recommender. It likely combines content-based filtering,
collaborative filtering, sequence modeling, and context-aware ranking.
This is consistent with the data Peloton has available, detailed class
metadata and a large amount of member behavior data.

## <u>**Design**</u>

Peloton’s recommender likely uses both user-level and class-level
signals. User-level signals may include workout history, class
completions, instructor preferences, bookmarks, duration preferences,
and recent activity. Class-level signals may include modality,
instructor, music, difficulty, equipment, class length, and body focus.

Likely recommender components:

- Content-based filtering: recommends classes with similar metadata,
  such as instructor, workout type, duration, music, and difficulty.

- Collaborative filtering: learns from patterns across members with
  similar class histories or instructor preferences.

- Sequence modeling: considers the order of recent workouts so the next
  recommendation fits the member’s current training pattern.

- Context-aware ranking: adjusts recommendations based on time
  available, device, equipment, and the home-screen row being shown.

- Goal-based personalization: connects class recommendations to weekly
  plans, member goals, and performance estimates.

## <u>**User Experience**</u>

Peloton’s recommender reduces the amount of searching needed before
starting a workout. Given the size of the class library, recommendations
can help members move from browsing to exercising more quickly. The
recommendations may still be off-target when the system lacks
information about soreness, injury, schedule changes, or motivation. A
class can match a member’s past behavior and still be a poor fit for
that day. The system functions reasonably well as a discovery tool, but
has limitations as a substitute for more individualized training
guidance.

# <u>**Attacks on Recommender Systems**</u>

Example citation:

- Pierri, F., DeVerna, M. R., Yang, K.-C., Axelrod, D., Bryden, J., &
  Menczer, F. (2022). *One Year of COVID-19 Vaccine Misinformation on
  Twitter: Longitudinal Study*. <https://arxiv.org/abs/2209.01675>

## <u>**Overview**</u>

The Washington Post article about IMDb users targeting a movie before
release shows how coordinated users can manipulate ratings and
visibility. A health-related example is COVID-19 vaccine misinformation
on Twitter. Pierri et al. (2022) found that a small group of
misinformation superspreaders accounted for a large share of vaccine
misinformation reshares. This kind of activity can affect recommendation
systems because platforms often use engagement signals shares, clicks,
and ratings as proxies for quality.

## <u>**Prevention Strategy**</u>

To prevent this kind of abuse, a recommender system should not treat
every signal as equally trustworthy. New accounts should carry less
weight by default, and accounts showing coordinated behavior should be
flagged or down-weighted. Sudden spikes in ratings, shares, or negative
feedback should also be reviewed before they influence rankings. The
system should use anomaly detection to identify coordinated behavior.
For a health platform, this is especially important because manipulated
recommendations could affect public trust, treatment decisions, or
preventive health behavior.

## <u>**Conclusion**</u>

The attack example shows why recommendation systems need safeguards.
Engagement is useful, but it can be manipulated. A better system should
weigh verified behavior, detect suspicious patterns, and protect model
training from coordinated abuse.
