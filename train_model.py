"""Train + compare models on data/samples.csv -> model/sign_model.pkl, metrics.json, confusion_matrix.png"""
import json, numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from features import mirror_features

df = pd.read_csv("data/samples.csv", header=None)
y = df.iloc[:, 0].astype(str).values
X = df.iloc[:, 1:].values.astype(np.float32)
print(f"Dataset: {len(X)} samples, {len(set(y))} classes, {X.shape[1]} features each")
X = np.vstack([X, np.array([mirror_features(r) for r in X])])   # left/right-hand augmentation
y = np.concatenate([y, y])
print(f"After mirror augmentation: {len(X)} samples")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
    "MLP (neural net)": make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=600, random_state=42)),
}
results, best_name, best_acc = {}, None, -1
for name, m in models.items():
    cv = cross_val_score(m, Xtr, ytr, cv=5).mean()
    m.fit(Xtr, ytr); acc = m.score(Xte, yte)
    results[name] = {"cv_accuracy": round(float(cv), 4), "test_accuracy": round(float(acc), 4)}
    print(f"{name:18s} 5-fold CV: {cv*100:5.1f}%   test: {acc*100:5.1f}%")
    if acc > best_acc: best_name, best_acc = name, acc

best = models[best_name]
pred = best.predict(Xte)
print(f"\nBest model: {best_name} ({best_acc*100:.1f}%)\n")
print(classification_report(yte, pred))
labels = sorted(set(y))
fig, ax = plt.subplots(figsize=(max(6, len(labels)*0.5), max(5, len(labels)*0.45)))
ConfusionMatrixDisplay(confusion_matrix(yte, pred, labels=labels), display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
plt.title(f"Confusion matrix - {best_name}"); plt.tight_layout(); plt.savefig("model/confusion_matrix.png", dpi=150)
json.dump({"samples": int(len(X)), "classes": labels, "best_model": best_name, "results": results},
          open("model/metrics.json", "w"), indent=2)
joblib.dump(best, "model/sign_model.pkl")
print("Saved: model/sign_model.pkl, model/metrics.json, model/confusion_matrix.png")
