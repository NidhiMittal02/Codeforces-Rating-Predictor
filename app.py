from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from flask_cors import CORS

from Model.predict import func


# =========================================================
# FLASK APP SETUP
# =========================================================

app = Flask(__name__)

CORS(app)

app.config["CACHE_TYPE"] = "simple"

cache = Cache(app)

app.secret_key = "development-secret-key"


# =========================================================
# RANK COLORS
# =========================================================

RANK_COLORS = {
    "Newbie": "#A9A9A9",
    "Pupil": "#32CD32",
    "Specialist": "#00ffcc",
    "Expert": "#1E90FF",
    "Candidate Master": "#800080",
    "Master": "#FFD700",
    "International Master": "#DAA520",
    "Grandmaster": "#DAA520",
    "International Grandmaster": "#DC143C",
    "Legendary Grandmaster": "#FF0000",
    "Tourist": "#FFFFFF"
}


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "GET":
        return render_template("index.html")


    # Get username from form
    username = request.form.get(
        "profile_name",
        ""
    ).strip()


    # Run prediction
    result = func(username)


    # Handle errors
    if not result[0]:

        return render_template(
            "index.html",
            flag=True,
            error_message=result[1]
        )


    # Get prediction and user information
    prediction = result[1][0]

    user_data = result[1][1]


    # Determine Codeforces rank
    rank = user_data.get(
        "rank",
        ""
    ).title()


    # Add rank color
    user_data["rank_color"] = RANK_COLORS.get(
        rank,
        "#A9A9A9"
    )


    if "rating" not in user_data:

        message = (
            "Your current Codeforces rating is not available, "
            "so the prediction error cannot be calculated."
        )

    else:

        actual_rating = user_data["rating"]

        if prediction > actual_rating:

            message = (
                "Hopefully, you will reach your "
                "expected rating soon :)"
            )

        else:

            message = (
                "You are above your expected rating, "
                "Bravo!!!"
            )


    # Render result page
    return render_template(
        "result.html",
        user_data=user_data,
        prediction=prediction,
        msg=message
    )


# =========================================================
# ABSOLUTE VALUE FILTER
# =========================================================

@app.template_filter("abs")
def absolute_value(value):

    return abs(value)


# =========================================================
# API ENDPOINT
# =========================================================

@app.route(
    "/get/<string:s>",
    methods=["GET"]
)
def get_prediction(s):

    username = s.strip()


    # Run prediction
    result = func(username)


    # Handle errors
    if not result[0]:

        return jsonify({
            "error": result[1]
        }), 404


    prediction = result[1][0]

    user_data = result[1][1]


    # Determine rank
    rank = user_data.get(
        "rank",
        ""
    ).title()


    # Add rank color
    user_data["rank_color"] = RANK_COLORS.get(
        rank,
        "#A9A9A9"
    )


    # Return JSON response
    return jsonify({

        "user_data": user_data,

        "prediction": prediction

    }), 200


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )