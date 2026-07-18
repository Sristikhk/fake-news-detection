from flask import Flask, render_template, request
import joblib
import re
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# Download NLTK Data
# -------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# -------------------------
# Load SpaCy Model
# -------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

# -------------------------
# Stopwords & Lemmatizer
# -------------------------
stop_words = set(stopwords.words("english"))
stop_words.update(nlp.Defaults.stop_words)

lemmatizer = WordNetLemmatizer()

# -------------------------
# Load Model & Vectorizer
# -------------------------
model = joblib.load("model/svm_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")


# -------------------------
# Text Cleaning Function
# -------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)

    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)

    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'d", " would", text)

    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = []

    for word in text.split():

        if word not in stop_words:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)


# -------------------------
# Home Page
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------
# Prediction Page
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    word_count = len(news.split())
    char_count = len(news)

    cleaned_news = clean_text(news)

    vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector)

    score = model.decision_function(vector)

    confidence = round(
        (1 / (1 + pow(2.71828, -abs(score[0])))) * 100,
        2
    )

    if prediction[0] == 1:
        result = "🟥 Fake News"
    else:
        result = "🟩 Real News"

    return render_template(
        "index.html",
        prediction=result,
        news=news,
        word_count=word_count,
        char_count=char_count,
        confidence=confidence
    )
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/developer")
def developer():
    return render_template("developer.html")
# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)