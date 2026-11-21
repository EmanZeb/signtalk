# 🤟 SignTalk – Real-Time Sign Language Translator (AI + Computer Vision)

Translates hand gestures from a webcam into **text** and **speech**, and lets non-signers reply via
**typed text or voice (speech-to-text)** shown as large on-screen text — a two-way communication aid for
hearing/speech-impaired people in schools, hospitals, workplaces and public places.

## How it works
Webcam → **MediaPipe Hands** (21 3-D landmarks) → gesture classifier → temporal smoothing + hold-to-confirm →
word added to sentence → **Text-to-Speech** (browser Web Speech API / offline pyttsx3).

Two recognition modes (auto-selected):
1. **Built-in gestures** – rule-based on finger states (works instantly, 14 signs, no training needed).
2. **Custom ML model** – record your own signs (e.g. ASL alphabet), train a RandomForest on normalised landmarks.

## Live web version (no install)
The `web/` folder is a static, browser-only version (MediaPipe Tasks Vision in JavaScript) - deployed on Vercel
(Root Directory = `web`). Open the live URL, click **Start camera**, allow camera access.
To run it locally: `cd web && python -m http.server 8000` then open http://localhost:8000

## Setup (Python 3.9 – 3.11 recommended)
```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
```bash
python app.py        # web UI -> open http://127.0.0.1:5000 (Chrome/Edge)
python desktop.py    # OR desktop window with offline voice (Q quit, C clear, U undo, S speak)
```
Camera not opening? `set CAM=1` (Windows) or `CAM=1 python app.py`. Hold time per word: `set HOLD=0.8`.

## Train your own signs (optional)
```bash
python collect_data.py A B C D     # SPACE = record, N = next label, Q = quit (~200 samples each)
python train_model.py              # prints accuracy, saves model/sign_model.pkl
python app.py                      # now uses your model automatically
```
Delete `model/sign_model.pkl` to return to built-in gestures.

## Files
| File | Purpose |
|---|---|
| `engine.py` | Pipeline: detection, classification, smoothing, sentence building |
| `gestures.py` | Rule-based classifier + sign guide |
| `features.py` | Landmark normalisation (translation/scale invariant) |
| `app.py`, `templates/index.html` | Flask web app: live video, TTS, speech-to-text |
| `desktop.py` | OpenCV desktop app |
| `collect_data.py`, `train_model.py` | Custom dataset + training |

## Limitations & future work
Static signs only; add an LSTM over landmark sequences for motion signs, multi-language TTS, mobile app,
facial-expression/grammar support.

**Tech:** Python, OpenCV, MediaPipe, scikit-learn, Flask, Web Speech API, pyttsx3.

## Data & model pipeline
- **In-browser dataset builder** (`web/`): record your own signs, KNN learns instantly, export dataset as JSON.
- **Large dataset import**: `python build_dataset_from_images.py <ASL_Alphabet_folder> --per-class 400` (e.g. Kaggle "ASL Alphabet", 87k images) extracts 63 normalised landmark features per image.
- **Training & evaluation**: `python train_model.py` compares RandomForest, KNN, SVM and an MLP neural network
  (5-fold CV + held-out test), saves the best model, `model/metrics.json` and `model/confusion_matrix.png`.
- Features: 21 landmarks x (x,y,z), wrist-translated and scale-normalised; left/right-hand mirror augmentation.
