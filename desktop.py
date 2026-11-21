"""Desktop version (OpenCV window + offline voice). Keys: Q quit | C clear | U undo | S speak sentence"""
import threading, cv2
from engine import SignEngine

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


def speak(text):
    if not pyttsx3 or not text: return
    def run():
        e = pyttsx3.init(); e.say(text); e.runAndWait()
    threading.Thread(target=run, daemon=True).start()


engine = SignEngine()
cap = cv2.VideoCapture(0)
last_v = 0
print("Running - Q to quit")
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = engine.process(cv2.flip(frame, 1))
    if engine.version != last_v:
        last_v = engine.version
        if engine.sentence: speak(engine.sentence[-1])
    cv2.imshow("SignTalk - Real-Time Sign Language Translator", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    elif k == ord("c"): engine.clear()
    elif k == ord("u"): engine.undo()
    elif k == ord("s"): speak(" ".join(engine.sentence))
cap.release(); cv2.destroyAllWindows()
