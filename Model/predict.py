import numpy as np
import requests
import joblib
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

MODEL_PATH = Path(__file__).resolve().parent / "regression_model.pkl"

model_package = joblib.load(MODEL_PATH)

# The .pkl contains a dictionary.
# The actual ML model is stored inside the "model" key.
model = model_package["model"]

feature_names = model_package["features"]
model_name = model_package["model_name"]

API_TIMEOUT = 10


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def calculate_features(
    distribution,
    solved_count,
    registration_time
):
    """
    Generate the same 8 features used during training.
    """

    problem_ratings = []

    for rating, count in distribution.items():

        rating = int(rating)
        count = int(count)

        problem_ratings.extend([rating] * count)

    # -----------------------------------------------------
    # Problem rating statistics
    # -----------------------------------------------------

    if problem_ratings:

        avg_problem_rating = np.mean(problem_ratings)

        problem_rating_std = np.std(problem_ratings)

        max_problem_rating = max(problem_ratings)

    else:

        avg_problem_rating = 0

        problem_rating_std = 0

        max_problem_rating = 0


    # -----------------------------------------------------
    # Problem difficulty distribution
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
    # Registration age
    # -----------------------------------------------------

    registration_date = datetime.fromtimestamp(
        registration_time,
        tz=timezone.utc
    )

    current_date = datetime.now(timezone.utc)

    registration_age_days = (
        current_date - registration_date
    ).days


    # -----------------------------------------------------
    # EXACT SAME ORDER AS TRAINING
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
# PREDICTION FUNCTION
# =========================================================

def func(handle):

    handle = handle.strip()

    if not handle:
        return False, "Please enter a Codeforces username."


    # =====================================================
    # 1. GET SUBMISSIONS
    # =====================================================

    submissions_url = (
        "https://codeforces.com/api/user.status"
        f"?handle={handle}&from=1&count=25000"
    )

    try:

        response = requests.get(
            submissions_url,
            timeout=API_TIMEOUT
        )

        response.raise_for_status()

        submission_data = response.json()

    except requests.exceptions.Timeout:

        return False, "Codeforces API request timed out."

    except requests.exceptions.RequestException:

        return False, "Unable to connect to Codeforces API."

    except ValueError:

        return False, "Invalid response received from Codeforces API."


    if submission_data.get("status") != "OK":

        return False, submission_data.get(
            "comment",
            "Codeforces API returned an error."
        )


    submissions = submission_data.get(
        "result",
        []
    )


    # =====================================================
    # 2. BUILD RATING DISTRIBUTION
    # =====================================================

    solved_problem_ids = set()

    rating_distribution = Counter()

    for submission in submissions:

        if submission.get("verdict") != "OK":
            continue

        problem = submission.get(
            "problem",
            {}
        )

        if "rating" not in problem:
            continue

        contest_id = problem.get("contestId")
        index = problem.get("index")

        problem_id = (
            contest_id,
            index
        )

        # Don't count the same problem twice
        if problem_id in solved_problem_ids:
            continue

        solved_problem_ids.add(problem_id)

        rating_distribution[
            problem["rating"]
        ] += 1


    # Number of rated problems solved
    rated_solved_count = sum(
        rating_distribution.values()
    )


    # =====================================================
    # 3. GET USER INFORMATION
    # =====================================================

    user_url = (
        "https://codeforces.com/api/user.info"
        f"?handles={handle}"
    )

    try:

        response = requests.get(
            user_url,
            timeout=API_TIMEOUT
        )

        response.raise_for_status()

        user_data = response.json()

    except requests.exceptions.Timeout:

        return False, "Codeforces user API request timed out."

    except requests.exceptions.RequestException:

        return False, "Unable to connect to Codeforces user API."

    except ValueError:

        return False, "Invalid user data received from Codeforces API."


    if user_data.get("status") != "OK":

        return False, user_data.get(
            "comment",
            "Unable to retrieve Codeforces user."
        )


    users = user_data.get(
        "result",
        []
    )

    if not users:

        return False, "Codeforces user was not found."


    user = users[0]


    # =====================================================
    # 4. GET REGISTRATION TIME
    # =====================================================

    registration_time = user.get(
        "registrationTimeSeconds",
        0
    )


    # =====================================================
    # 5. CREATE FEATURES
    # =====================================================

    features = calculate_features(
        distribution=rating_distribution,
        solved_count=rated_solved_count,
        registration_time=registration_time
    )


    X = np.array(
        [features],
        dtype=float
    )


    # =====================================================
    # 6. PREDICT
    # =====================================================

    try:

        prediction = model.predict(X)

        predicted_rating = float(
            prediction[0]
        )

    except Exception as e:

        return False, f"Prediction failed: {str(e)}"


    # =====================================================
    # 7. RETURN RESULT
    # =====================================================

    return True, [
        predicted_rating,
        user
    ]