import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

df = pd.read_csv("songs_normalize.csv")

df["genre"] = df["genre"].apply(lambda x: x.split(",")[0].strip().lower())

genre_counts = df["genre"].value_counts()
valid_genres = genre_counts[genre_counts > 1].index
df = df[df["genre"].isin(valid_genres)]

X = df.drop(columns=["artist", "song", "genre"])
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_values = range(1, 11)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"Best k = {best_k}, CV accuracy = {cv_scores[best_k - 1]:.4f}")

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
test_accuracy = final_knn.score(X_test_scaled, y_test)
print(f"Test Accuracy (k={best_k}): {test_accuracy:.4f}")
