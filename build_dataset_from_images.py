"""Build a landmark dataset from a LARGE image dataset (e.g. Kaggle 'ASL Alphabet', 87,000 images).
Folder layout:  dataset_root/A/*.jpg, dataset_root/B/*.jpg ...
Usage: python build_dataset_from_images.py path/to/asl_alphabet_train --per-class 400
Output: data/samples.csv  (then run: python train_model.py)"""
import os, csv, sys, argparse, cv2, numpy as np, mediapipe as mp
from features import extract_features

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--per-class", type=int, default=400)
a = ap.parse_args()
os.makedirs("data", exist_ok=True)
hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
total = 0
with open("data/samples.csv", "w", newline="") as f:
    w = csv.writer(f)
    for label in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, label)
        if not os.path.isdir(d): continue
        n = 0
        for name in sorted(os.listdir(d)):
            if n >= a.per_class: break
            img = cv2.imread(os.path.join(d, name))
            if img is None: continue
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks: continue
            pts = np.array([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark])
            w.writerow([label, *extract_features(pts)]); n += 1
        total += n; print(f"{label}: {n} samples")
print("Total samples:", total, "-> data/samples.csv")
