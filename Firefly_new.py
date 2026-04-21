import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def main():

    # ======================================================
    # STEP 1: INITIALIZATION
    # ======================================================
    print("\n[STEP 1] Loading Data")

    file_path = '../spam.csv' if os.path.exists('../spam.csv') else 'spam.csv'
    df = pd.read_csv(file_path, encoding='latin-1')

    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Baseline
    base_model = LogisticRegression(C=1.0)
    base_model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, base_model.predict(X_test))

    print(f"Baseline Accuracy: {baseline_acc:.4f}")

    # ======================================================
    # PARAMETERS 
    # ======================================================
    beta0 = 1.0
    gamma = 0.3
    alpha = 0.25

    num_fireflies = 20
    max_iterations = 25

    min_C, max_C = 0.01, 10

    # ======================================================
    # STEP 2: FITNESS
    # ======================================================
    def fitness(c):
        c = max(0.001, c)
        model = LogisticRegression(C=c, max_iter=200)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    # ======================================================
    # STEP 3: INITIALIZE FIREFLIES
    # ======================================================
    positions = np.random.uniform(min_C, max_C, num_fireflies)
    brightness = np.zeros(num_fireflies)

    best_history = []
    mean_history = []

    position_history = []

    print("\n[STEP 4–7] Optimization Running...\n")

    for iteration in range(max_iterations):

        # Evaluate fitness
        for i in range(num_fireflies):
            brightness[i] = fitness(positions[i])

        best = np.max(brightness)
        mean = np.mean(brightness)

        best_history.append(best)
        mean_history.append(mean)
        position_history.append(positions.copy())

        print(f"Iteration {iteration+1}/25 → Best Accuracy: {best:.4f}")

        # Movement
        for i in range(num_fireflies):
            for j in range(num_fireflies):

                if brightness[j] > brightness[i]:

                    # STEP 3: Distance
                    d = abs(positions[i] - positions[j])

                    # STEP 4: Attractiveness
                    beta = beta0 * np.exp(-gamma * d**2)

                    # STEP 5: Movement
                    positions[i] += beta * (positions[j] - positions[i]) + alpha * (np.random.rand() - 0.5)

                    # STEP 6: Update
                    positions[i] = np.clip(positions[i], min_C, max_C)

    # ======================================================
    # FINAL RESULT
    # ======================================================
    best_idx = np.argmax(brightness)

    print("\n========== FINAL RESULT ==========")
    print(f"Baseline Accuracy : {baseline_acc:.4f}")
    print(f"Optimized Accuracy: {brightness[best_idx]:.4f}")
    print(f"Best C Value      : {positions[best_idx]:.4f}")
    print("=================================")

    # ======================================================
    # 🔥 VISUALIZATION SECTION
    # ======================================================

    # 1. Optimization Curve
    plt.figure(figsize=(10,5))
    plt.plot(best_history, label="Best Accuracy")
    plt.plot(mean_history, label="Mean Accuracy")
    plt.title("Optimization Progress")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("plot1_optimization.png")

    # 2. Firefly Movement
    plt.figure(figsize=(10,5))
    for i in range(len(position_history)):
        plt.scatter([i]*num_fireflies, position_history[i], alpha=0.6)
    plt.title("Firefly Movement")
    plt.xlabel("Iteration")
    plt.ylabel("C Value")
    plt.grid()
    plt.savefig("plot2_movement.png")

    # 3. Parameter Effect (Gamma)
    distances = np.linspace(0, 5, 100)
    beta_values = beta0 * np.exp(-gamma * distances**2)

    plt.figure(figsize=(8,5))
    plt.plot(distances, beta_values)
    plt.title("Effect of Gamma on Attractiveness")
    plt.xlabel("Distance")
    plt.ylabel("Attractiveness")
    plt.grid()
    plt.savefig("plot3_gamma_effect.png")

    # 4. Alpha Effect
    alpha_values = np.linspace(0, 1, 100)
    plt.figure(figsize=(8,5))
    plt.plot(alpha_values, alpha_values)
    plt.title("Alpha Controls Randomness")
    plt.xlabel("Alpha")
    plt.ylabel("Randomness Strength")
    plt.grid()
    plt.savefig("plot4_alpha.png")

    # 5. Final Accuracy Distribution
    plt.figure(figsize=(8,5))
    plt.hist(brightness, bins=10)
    plt.title("Final Firefly Accuracy Distribution")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig("plot5_distribution.png")

    print("\n🔥 All plots generated successfully!")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.savefig("plot6_confusion_matrix.png")
    plt.show()
    
    return cm

    
if __name__ == "__main__":
    main()
