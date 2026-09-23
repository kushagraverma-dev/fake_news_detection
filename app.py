from flask import Flask, render_template, request
import joblib
from utils.preprocessing import clean_text

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    news_text = ""

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()

        if news_text:
            cleaned = clean_text(news_text)
            features = vectorizer.transform([cleaned])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = round(max(probabilities) * 100, 2)
            result = "REAL NEWS" if prediction == 1 else "FAKE NEWS"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        news_text=news_text
    )

if __name__ == "__main__":
    app.run(debug=True)
