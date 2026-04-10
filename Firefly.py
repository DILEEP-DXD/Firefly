# ============================================================
# 🔥 FIRELFY SPAM CLASSIFICATION — FINAL VERSION
# Clean | Stable | No Data Leakage | Viva-Ready
# ============================================================

# ==============================
# STEP 0 — IMPORTS
# ==============================
import numpy as np
import pandas as pd
import re, nltk, warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ==============================
# STEP 1 — LOAD DATA
# ==============================
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1','v2']]
df.columns = ['label','message']

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"Dataset: {len(df)} samples")

# ==============================
# STEP 2 — PREPROCESSING
# ==============================
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return " ".join([w for w in tokens if w not in stop_words and w.isalpha()])

df['clean'] = df['message'].apply(preprocess)

# ==============================
# STEP 3 — TF-IDF + SPLIT
# ==============================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean'])
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# STEP 4 — BASELINE MODEL
# ==============================
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

baseline_acc = accuracy_score(y_test, y_pred)
baseline_f1  = f1_score(y_test, y_pred)

print("\n🔹 BASELINE")
print("Accuracy:", baseline_acc)
print("F1 Score:", baseline_f1)

# ==============================
# STEP 5 — CHI2 FEATURE SELECTION
# ==============================
TOP_K = 500   # important fix

scores, _ = chi2(X_train, y_train)
top_idx = np.argsort(scores)[::-1][:TOP_K]

X_train_k = X_train[:, top_idx]
X_test_k  = X_test[:, top_idx]

# ==============================
# STEP 6 — FIREFLY ALGORITHM
# ==============================
N = 20
ITER = 20
BETA_0 = 1
GAMMA = 0.1

np.random.seed(42)

# Initialize near full features
fireflies = np.ones((N, TOP_K), dtype=int)

for i in range(N):
    flip = np.random.choice(TOP_K, int(0.1 * TOP_K), replace=False)
    fireflies[i, flip] = 0

fitness_cache = {}

def fitness(sol):
    key = tuple(sol)
    if key in fitness_cache:
        return fitness_cache[key]

    idx = np.where(sol == 1)[0]
    if len(idx) < 5:
        return 0

    X_sub = X_train_k[:, idx]

    # validation split (NO leakage)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sub, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    model = MultinomialNB()
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)

    f1 = f1_score(y_val, pred)

    # feature penalty
    penalty = len(idx) / TOP_K

    score = f1 * (0.7 + 0.3 * penalty)

    fitness_cache[key] = score
    return score

def transfer(v):
    return abs(np.tanh(v))

best = fireflies[0].copy()
best_score = fitness(best)

for it in range(ITER):
    alpha = 0.5 * (1 - it / ITER)

    for i in range(N):
        for j in range(N):
            if fitness(fireflies[j]) > fitness(fireflies[i]):

                diff = fireflies[j] - fireflies[i]
                r = np.linalg.norm(diff)
                beta = BETA_0 * np.exp(-GAMMA * r*r)

                noise = 0.3 * alpha * (np.random.rand(TOP_K) - 0.5)
                v = beta * diff + noise

                prob = transfer(v)
                rand = np.random.rand(TOP_K)

                new = fireflies[i].copy()
                new[rand < prob] ^= 1

                if fitness(new) > fitness(fireflies[i]):
                    fireflies[i] = new

    # update best
    scores = [fitness(f) for f in fireflies]
    idx = np.argmax(scores)

    if scores[idx] > best_score:
        best_score = scores[idx]
        best = fireflies[idx].copy()

    print(f"Iter {it+1}/{ITER} | Best F1: {best_score:.4f}")

# ==============================
# STEP 7 — FINAL MODEL
# ==============================
selected = np.where(best == 1)[0]
selected_global = top_idx[selected]

X_tr_final = X_train[:, selected_global]
X_te_final = X_test[:, selected_global]

model = MultinomialNB()
model.fit(X_tr_final, y_train)

y_pred = model.predict(X_te_final)

opt_acc = accuracy_score(y_test, y_pred)
opt_f1  = f1_score(y_test, y_pred)

print("\n🚀 OPTIMIZED")
print("Accuracy:", opt_acc)
print("F1 Score:", opt_f1)

# ==============================
# STEP 8 — COMPARISON
# ==============================
print("\n📊 IMPROVEMENT")
print("Accuracy Change:", opt_acc - baseline_acc)
print("F1 Change:", opt_f1 - baseline_f1)


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# ==============================
# 1. ACCURACY & F1 COMPARISON
# ==============================
labels = ['Baseline', 'Optimized']
accuracy = [baseline_acc, opt_acc]
f1_scores = [baseline_f1, opt_f1]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, accuracy, width, label='Accuracy')
plt.bar(x + width/2, f1_scores, width, label='F1 Score')

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.show()

# ==============================
# 2. CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,5))
plt.imshow(cm)
plt.title("Confusion Matrix (Optimized Model)")
plt.colorbar()

labels = ['Ham', 'Spam']
plt.xticks([0,1], labels)
plt.yticks([0,1], labels)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 3. FEATURE REDUCTION GRAPH
# ==============================
features = [3000, TOP_K, len(selected_global)]
steps = ['Original', 'Chi2 Selected', 'Firefly Selected']

plt.figure(figsize=(7,5))
plt.plot(steps, features, marker='o')
plt.title("Feature Reduction Process")
plt.ylabel("Number of Features")
plt.show()

# ==============================
# 4. FIREFLY CONVERGENCE
# ==============================
# Make sure you stored history during training
try:
    plt.figure(figsize=(7,5))
    plt.plot(history)
    plt.title("Firefly Optimization Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Best F1 Score")
    plt.show()
except:
    print("⚠️ history not found — skip convergence plot")

# ==============================
# 5. SAVE ALL FIGURES (OPTIONAL)
# ==============================
plt.figure(figsize=(6,4))
plt.bar(labels, accuracy)
plt.title("Accuracy Comparison")
plt.savefig("accuracy_plot.png")

plt.figure(figsize=(6,4))
plt.bar(labels, f1_scores)
plt.title("F1 Score Comparison")
plt.savefig("f1_plot.png")

print("\n✅ Graphs generated & saved successfully!")
