from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        news = request.form["news"]

        news_vector = vectorizer.transform([news])

        result = model.predict(news_vector)[0]

        if result == 0:
            prediction = "FAKE NEWS"
        else:
            prediction = "REAL NEWS"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)