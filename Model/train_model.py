import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data.json"

MODEL_PATH = BASE_DIR / "regression_model.pkl"


# =========================================================
# LOAD DATASET
# =========================================================

with open(DATA_PATH, "r", encoding="utf-8") as file:

    data = json.load(file)


print(
    f"Total users in dataset: {len(data)}"
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def extract_features(user):

    distribution = user.get(
        "RatingDistribution",
        {}
    )


    # -----------------------------------------------------
    # Create list of problem ratings
    # -----------------------------------------------------

    problem_ratings = []

    for rating, count in distribution.items():

        rating = int(rating)

        count = int(count)

        problem_ratings.extend(
            [rating] * count
        )


    # -----------------------------------------------------
    # Rated problems solved
    #
    # IMPORTANT:
    # This matches the live prediction pipeline.
    # -----------------------------------------------------

    solved_count = sum(
        int(count)
        for count in distribution.values()
    )


    # -----------------------------------------------------
    # Problem rating statistics
    # -----------------------------------------------------

    if problem_ratings:

        avg_problem_rating = np.mean(
            problem_ratings
        )

        problem_rating_std = np.std(
            problem_ratings
        )

        max_problem_rating = max(
            problem_ratings
        )

    else:

        avg_problem_rating = 0

        problem_rating_std = 0

        max_problem_rating = 0


    # -----------------------------------------------------
    # Difficulty distribution
    # -----------------------------------------------------

    # Easy: below 1300
    easy_solved = sum(
        count
        for rating, count in distribution.items()
        if int(rating) < 1300
    )


    # Medium: 1300 - 2000
    medium_solved = sum(
        count
        for rating, count in distribution.items()
        if 1300 <= int(rating) <= 2000
    )


    # Hard: above 2000
    hard_solved = sum(
        count
        for rating, count in distribution.items()
        if int(rating) > 2000
    )


    # -----------------------------------------------------
    # Account age
    # -----------------------------------------------------

    registration_time = user.get(
        "registrationTimeSeconds",
        0
    )


    registration_date = datetime.fromtimestamp(
        registration_time,
        tz=timezone.utc
    )


    current_date = datetime.now(
        timezone.utc
    )


    registration_age_days = (
        current_date - registration_date
    ).days


    # -----------------------------------------------------
    # Return features
    #
    # IMPORTANT:
    # Keep this EXACT order in prediction.py
    # -----------------------------------------------------

    return [

        avg_problem_rating,

        solved_count,

        max_problem_rating,

        easy_solved,

        medium_solved,

        hard_solved,

        problem_rating_std,

        registration_age_days

    ]


# =========================================================
# BUILD DATASET
# =========================================================

X = []

y = []


for user in data:

    # Skip users without a current rating
    if "rating" not in user:

        continue


    rating = user["rating"]


    features = extract_features(
        user
    )


    X.append(features)

    y.append(rating)


X = np.array(
    X,
    dtype=float
)


y = np.array(
    y,
    dtype=float
)


print(
    f"Training samples: {len(X)}"
)


print(
    f"Number of features: {X.shape[1]}"
)


# =========================================================
# FEATURE NAMES
# =========================================================

feature_names = [

    "avg_problem_rating",

    "solved_count",

    "max_problem_rating",

    "easy_solved",

    "medium_solved",

    "hard_solved",

    "problem_rating_std",

    "registration_age_days"

]


print("\nFeatures:")

for feature in feature_names:

    print(
        f"- {feature}"
    )


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)


print("\nTraining set size:", len(X_train))

print("Testing set size:", len(X_test))


# =========================================================
# MODELS
# =========================================================

models = {

    "Linear Regression":
        LinearRegression(),


    "Random Forest":
        RandomForestRegressor(

            n_estimators=300,

            random_state=42,

            n_jobs=-1

        ),


    "Gradient Boosting":
        GradientBoostingRegressor(

            n_estimators=200,

            learning_rate=0.05,

            max_depth=3,

            random_state=42

        )

}


# =========================================================
# MODEL EVALUATION
# =========================================================

results = {}


print("\n" + "=" * 60)

print("MODEL EVALUATION")

print("=" * 60)


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    # Train
    model.fit(
        X_train,
        y_train
    )


    # Predict
    predictions = model.predict(
        X_test
    )


    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )


    r2 = r2_score(
        y_test,
        predictions
    )


    # Store results
    results[name] = {

        "model": model,

        "mae": mae,

        "rmse": rmse,

        "r2": r2

    }


    print(
        f"MAE : {mae:.2f}"
    )

    print(
        f"RMSE: {rmse:.2f}"
    )

    print(
        f"R²  : {r2:.4f}"
    )


# =========================================================
# SELECT BEST MODEL
# =========================================================

# We use MAE as the primary selection metric.
# Lower MAE means lower average prediction error.

best_model_name = min(

    results,

    key=lambda name:
        results[name]["mae"]

)


best_model = results[
    best_model_name
]["model"]


# =========================================================
# DISPLAY SELECTED MODEL
# =========================================================

print("\n" + "=" * 60)

print("SELECTED MODEL")

print("=" * 60)


print(
    f"Model: {best_model_name}"
)


print(
    f"MAE: "
    f"{results[best_model_name]['mae']:.2f}"
)


print(
    f"RMSE: "
    f"{results[best_model_name]['rmse']:.2f}"
)


print(
    f"R²: "
    f"{results[best_model_name]['r2']:.4f}"
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

if hasattr(
    best_model,
    "feature_importances_"
):

    print("\n" + "=" * 60)

    print("FEATURE IMPORTANCE")

    print("=" * 60)


    feature_importance = sorted(

        zip(
            feature_names,
            best_model.feature_importances_
        ),

        key=lambda item:
            item[1],

        reverse=True

    )


    for feature, importance in feature_importance:

        print(
            f"{feature}: "
            f"{importance:.4f}"
        )


# =========================================================
# SAVE MODEL PACKAGE
# =========================================================

model_package = {

    "model": best_model,

    "features": feature_names,

    "model_name": best_model_name,

    "mae":
        results[best_model_name]["mae"],

    "rmse":
        results[best_model_name]["rmse"],

    "r2":
        results[best_model_name]["r2"]

}


joblib.dump(
    model_package,
    MODEL_PATH
)


# =========================================================
# SUCCESS MESSAGE
# =========================================================

print("\nModel saved successfully:")

print(
    MODEL_PATH
)