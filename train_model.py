import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ── 1. Data load ──────────────────────────────────────────────────
df = pd.read_csv('./data/fraud_dataset_csv.xls')   # yahan apna file path dalo
df.dropna(inplace=True)

# ── 2. Text preprocessing ─────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # URLs hataao
    text = re.sub(r'[^a-z\s]', ' ', text)        # sirf letters rakho
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['label']   # 0 = legit, 1 = fraud

# ── 3. Train/Test split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Pipeline (TF-IDF + Logistic Regression) ───────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),     # unigrams + bigrams
        sublinear_tf=True       # log scaling
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs'
    ))
])

# ── 5. Train ──────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

# ── 6. Evaluate ───────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

# ── 7. Save model ─────────────────────────────────────────────────
joblib.dump(pipeline, 'fraud_nlp_model.pkl ')
print("Model saved → fraud_nlp_model.pkl")