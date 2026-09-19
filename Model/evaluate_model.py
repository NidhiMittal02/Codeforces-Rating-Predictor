import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import matplotlib.pyplot as plt

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

PROJECT_DIR = BASE_DIR.parent

IMAGES_DIR = PROJECT_DIR / "Images"

IMAGES_DIR.mkdir(
    exist_ok=True
)


# =========================================================
# LOAD DATA
# =========================================================

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

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
    # Problem ratings
    # -----------------------------------------------------

    problem_ratings = []

    for rating, count in distribution.items():

        rating = int(rating)

        count = int(count)

        problem_ratings.extend(
            [rating] * count
        )


    # -----------------------------------------------------
    # Number of rated problems solved
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

    easy_solved = sum(
        count
        for rating, count in distribution.items()
        if int(rating) < 1300
    )


    medium_solved = sum(
        count
        for rating, count in distribution.items()
        if 1300 <= int(rating) <= 2000
    )


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
    # SAME FEATURE ORDER AS TRAINING
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
# CREATE X AND Y
# =========================================================

X = []

y = []


for user in data:

    if "rating" not in user:

        continue


    X.append(
        extract_features(user)
    )

    y.append(
        user["rating"]
    )


X = np.array(
    X,
    dtype=float
)


y = np.array(
    y,
    dtype=float
)


print(
    f"Evaluation samples: {len(X)}"
)


# =========================================================
# SAME TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model_package = joblib.load(
    MODEL_PATH
)


model = model_package["model"]

model_name = model_package["model_name"]

feature_names = model_package["features"]


print(
    f"\nLoaded model: {model_name}"
)


# =========================================================
# PREDICTIONS
# =========================================================

predictions = model.predict(
    X_test
)


# =========================================================
# EVALUATION METRICS
# =========================================================

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


print("\n" + "=" * 60)

print("MODEL PERFORMANCE")

print("=" * 60)


print(
    f"Model : {model_name}"
)

print(
    f"MAE   : {mae:.2f}"
)

print(
    f"RMSE  : {rmse:.2f}"
)

print(
    f"R²    : {r2:.4f}"
)


# =========================================================
# GRAPH 1
# ACTUAL VS PREDICTED
# =========================================================

plt.figure(
    figsize=(9, 7)
)


plt.scatter(
    y_test,
    predictions,
    alpha=0.6
)


# Perfect prediction reference line

min_rating = min(
    y_test.min(),
    predictions.min()
)

max_rating = max(
    y_test.max(),
    predictions.max()
)


plt.plot(
    [min_rating, max_rating],
    [min_rating, max_rating],
    linestyle="--"
)


plt.xlabel(
    "Actual Codeforces Rating"
)


plt.ylabel(
    "Predicted Codeforces Rating"
)


plt.title(
    f"Actual vs Predicted Rating - {model_name}"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


actual_predicted_path = (
    IMAGES_DIR /
    "actual_vs_predicted.png"
)


plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\nSaved: {actual_predicted_path}"
)


# =========================================================
# GRAPH 2
# FEATURE IMPORTANCE
# =========================================================

if hasattr(
    model,
    "feature_importances_"
):

    importances = model.feature_importances_


    # Sort from smallest to largest
    sorted_indices = np.argsort(
        importances
    )


    sorted_features = [
        feature_names[i]
        for i in sorted_indices
    ]


    sorted_importances = [
        importances[i]
        for i in sorted_indices
    ]


    plt.figure(
        figsize=(10, 7)
    )


    plt.barh(
        sorted_features,
        sorted_importances
    )


    plt.xlabel(
        "Feature Importance"
    )


    plt.ylabel(
        "Feature"
    )


    plt.title(
        f"Feature Importance - {model_name}"
    )


    plt.grid(
        axis="x",
        alpha=0.3
    )


    plt.tight_layout()


    feature_importance_path = (
        IMAGES_DIR /
        "feature_importance.png"
    )


    plt.savefig(
        feature_importance_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Saved: {feature_importance_path}"
    )


else:

    print(
        "\nFeature importance is not available "
        "for this model."
    )


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 60)

print("EVALUATION COMPLETE")

print("=" * 60)