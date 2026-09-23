# Fake News Detection - Python ML Project

A simple educational Fake News Detection web application built with Python, NLP, Scikit-learn and Flask.

## Features

- News article text input
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classifier
- Fake/Real prediction
- Confidence score
- Flask web interface
- Easy custom dataset training

## Project Structure

```text
fake-news-detection/
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
│   └── news.csv
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── templates/
│   └── index.html
├── static/
│   └── css/
│       └── style.css
└── utils/
    └── preprocessing.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Run the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Dataset

The included CSV is only a small demonstration dataset. For a serious project, use a larger, properly labeled dataset and evaluate the model on held-out data.

CSV format:

```csv
text,label
"Example real news article",1
"Example fake news article",0
```

Where:

- `1` = Real
- `0` = Fake

## Important

The application is an educational text classifier. It does not independently verify facts, sources, or claims. A prediction can be wrong, especially when the input differs from the training data.
