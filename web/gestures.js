// Port of gestures.py - rule-based static gesture classifier (21 MediaPipe landmarks, {x,y,z})
const GESTURES = {
  "11111": "Hello", "01111": "Stop", "01100": "Peace", "01000": "You", "00001": "I",
  "01110": "Water", "11001": "I love you", "10001": "Call me", "11000": "Help",
  "11100": "Thank you", "00000": "Yes"
};
const GUIDE = [
  ["Hello","Open palm, all 5 fingers spread"],["Stop","4 fingers up, thumb folded in"],
  ["Peace","Index + middle finger up (V sign)"],["You","Only index finger pointing up"],
  ["I","Only pinky finger up"],["Water","Index, middle, ring up (W shape)"],
  ["I love you","Thumb + index + pinky out"],["Call me","Thumb + pinky out (phone sign)"],
  ["Help","Thumb + index out (L shape)"],["Thank you","Thumb + index + middle out"],
  ["Yes","Closed fist"],["Good","Thumbs up"],["No","Thumbs down"],
  ["OK","Thumb & index touch in a circle, other 3 up"]
];
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function fingerStates(lm) {
  const palm = dist(lm[0], lm[9]) + 1e-6;
  const thumb = +(dist(lm[4], lm[17]) > dist(lm[3], lm[17]) * 1.05 && dist(lm[4], lm[5]) > 0.55 * palm);
  const others = [[8,6],[12,10],[16,14],[20,18]].map(([t,p]) => +(dist(lm[t], lm[0]) > dist(lm[p], lm[0]) * 1.05));
  return { fs: [thumb, ...others], palm };
}
function classify(lm) {
  const { fs, palm } = fingerStates(lm);
  const [thumb, idx, mid, ring, pinky] = fs;
  if (dist(lm[4], lm[8]) < 0.35 * palm && mid && ring && pinky) return "OK";
  if (!(idx || mid || ring || pinky)) {
    const dy = lm[2].y - lm[4].y;
    if (dy > 0.45 * palm) return "Good";
    if (dy < -0.45 * palm) return "No";
    if (thumb) return null;
    return "Yes";
  }
  return GESTURES[fs.join("")] || null;
}
if (typeof module !== "undefined") module.exports = { classify, GUIDE };
