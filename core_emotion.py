
# core_emotion.py
import os
import math
import json
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- Utility: default VA mapping (fallback) ----------
# If you already have EMOTION2VA mapping elsewhere, you can import or override it.
EMOTION2VA = {
    "afraid":(-0.85,0.9),"angry":(-0.9,0.85),"annoyed":(-0.5,0.6),"anticipating":(0.2,0.5),
    "anxious":(-0.7,0.8),"apprehensive":(-0.65,0.7),"ashamed":(-0.7,0.2),"caring":(0.6,0.3),
    "confident":(0.7,0.4),"content":(0.4,0.1),"devastated":(-0.95,-0.4),"disappointed":(-0.6,-0.2),
    "disgusted":(-0.7,0.3),"embarrassed":(-0.5,0.2),"excited":(0.8,0.9),"faithful":(0.5,0.2),
    "furious":(-0.95,0.9),"grateful":(0.6,0.2),"guilty":(-0.6,0.1),"hopeful":(0.5,0.5),
    "impressed":(0.5,0.4),"jealous":(-0.5,0.6),"joyful":(0.9,0.6),"lonely":(-0.6,-0.3),
    "nostalgic":(0.0,-0.1),"prepared":(0.3,0.4),"proud":(0.6,0.5),"sad":(-0.9,-0.4),
    "sentimental":(0.1,0.0),"surprised":(0.2,0.8),"terrified":(-0.98,0.95),"trusting":(0.6,0.2)
}
# ensure safe fallback for unknown labels
def _va_fallback(label):
    return EMOTION2VA.get(label, (0.0, 0.0))

# ---------- EmotionExtractorLocal (robust wrapper) ----------
class EmotionExtractorLocal:
    """
    Wrapper for a local transformerfƒs sequence-classification model.
    Exposes `predict(text)` -> (label, valence, arousal, probs_list)

    - local_model_path must contain a transformers-compatible model & tokenizer
    - model.config.id2label is used if present, otherwise integer labels are returned
    """
    def __init__(self, local_model_path, device=None):
        if not os.path.isdir(local_model_path):
            raise FileNotFoundError(f"Local model folder not found: {local_model_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[EmotionExtractorLocal] Loading model from: {local_model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(local_model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        cfg = getattr(self.model.config, "id2label", None)
        if cfg:
            try:
                # model may store id2label as dict or list-like
                self.id2label = {int(k): v for k, v in cfg.items()}
            except Exception:
                # fallback: enumerate values
                try:
                    self.id2label = {int(k): v for k, v in enumerate(cfg.values())}
                except Exception:
                    self.id2label = None
        else:
            self.id2label = None

    def predict(self, text):
        """
        Returns:
            label (str),
            valence (float in [-1,1]),
            arousal (float in [-1,1]),
            probs (list of floats)  -- full softmax
        """
        if not text or not text.strip():
            # empty -> neutral
            return "content", 0.0, 0.0, [1.0]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy().ravel().tolist()
            idx = int(np.argmax(probs))
        label = None
        if self.id2label is not None:
            label = self.id2label.get(idx, str(idx))
        else:
            label = str(idx)
        # map label -> (v,a) fallback via EMOTION2VA
        v, a = _va_fallback(label)
        return label, float(v), float(a), probs
    def get_VA(self, text):
         with torch.no_grad():
             inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
             outputs = self.model(**inputs)
             logits = outputs.logits
             probs = F.softmax(logits, dim=-1)
             idx = int(torch.argmax(probs, dim=-1).cpu().numpy())
             label = self.id2label.get(idx, "neutral")
             v, a = EMOTION2VA.get(label, (0.0, 0.0))
             return label, float(v), float(a), probs.cpu().numpy().tolist()

# ---------- Sentence splitter ----------
def split_sentences(text):
    if not text or text.strip() == "":
        return []
    txt = text.replace("؟", ".").replace("!", ".").replace("؛", ".")
    s = [t.strip() for t in txt.split(".") if t.strip()]
    if len(s) == 1:
        # duplicate single sentence so splines have >=2 points
        s = s + s
    return s

# ---------- Fingerprint computations ----------
def shannon_entropy_from_counts(counts):
    # counts: iterable of counts
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

def euclidean_distance(a, b):
    a = np.array(a); b = np.array(b)
    return float(np.linalg.norm(a - b))

# ---------- Core class that ties everything ----------
class EmotionCore:
    """
    High-level API:
    ec = EmotionCore(model_path)
    result = ec.analyze_story(story_text)

    result dict contains:
        - sentences: [...]
        - labels: [...]
        - probs: [list of probs per sentence]
        - V_points, A_points: lists (per-sentence valence & arousal)
        - fingerprint: {
            'entropy': float,
            'flow_complexity': float,
            'valence_range': float,
            'arousal_range': float,
            'valence_std': float,
            'arousal_std': float,
            'dominant_label': str,
            'label_histogram': {label:count,...},
            'label_probs_mean': [..] (averaged soft probs across sentences)
          }
    """
    def __init__(self, local_model_path):
        self.extractor = EmotionExtractorLocal(local_model_path)

    def analyze_story(self, text, debug=False):
        sentences = split_sentences(text)
        if len(sentences) == 0:
            # empty input -> neutral defaults
            sentences = [""]*2
        labels = []
        probs_list = []
        V_points = []
        A_points = []
        for s in sentences:
            lbl, v, a, probs = self.extractor.predict(s)
            labels.append(lbl)
            probs_list.append(probs)
            V_points.append(float(v))
            A_points.append(float(a))
        # histogram of labels
        hist = Counter(labels)
        entropy = shannon_entropy_from_counts(hist.values())
        # compute flow complexity: sum of Euclidean distances between consecutive VA points
        flow_comp = 0.0
        for i in range(1, len(V_points)):
            flow_comp += euclidean_distance((V_points[i-1], A_points[i-1]), (V_points[i], A_points[i]))
        # basic statistics
        v_arr = np.array(V_points)
        a_arr = np.array(A_points)
        v_range = float(v_arr.max() - v_arr.min())
        a_range = float(a_arr.max() - a_arr.min())
        v_std = float(v_arr.std())
        a_std = float(a_arr.std())
        dominant = hist.most_common(1)[0][0] if len(hist)>0 else None
        # mean soft probabilities across sentences (pad/prob lengths to same size)
        max_len = max(len(p) for p in probs_list)
        probs_padded = np.array([p + [0.0]*(max_len-len(p)) for p in probs_list])
        mean_probs = np.mean(probs_padded, axis=0).tolist()
        # assemble fingerprint
        fingerprint = {
            "entropy": float(entropy),
            "flow_complexity": float(flow_comp),
            "valence_range": v_range,
            "arousal_range": a_range,
            "valence_std": v_std,
            "arousal_std": a_std,
            "dominant_label": dominant,
            "label_histogram": dict(hist),
            "label_sequence": labels,
            "mean_label_probs": mean_probs,
            "num_sentences": len(sentences)
        }
        result = {
            "sentences": sentences,
            "labels": labels,
            "probs": probs_list,
            "V_points": V_points,
            "A_points": A_points,
            "fingerprint": fingerprint
        }
        if debug:
            print("[EmotionCore] sentences:", sentences)
            print("[EmotionCore] labels:", labels)
            print("[EmotionCore] V_points:", V_points)
            print("[EmotionCore] A_points:", A_points)
            print("[EmotionCore] fingerprint:", json.dumps(fingerprint, indent=2))
        return result

    def save_result(self, result, out_path):
        # safe JSON serializable dump
        o = {
            "sentences": result["sentences"],
            "labels": result["labels"],
            "probs": result["probs"],
            "V_points": result["V_points"],
            "A_points": result["A_points"],
            "fingerprint": result["fingerprint"]
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(o, f, ensure_ascii=False, indent=2)
        return out_path

# ---------- quick example (if run as script) ----------
if __name__ == "__main__":
    # DEMO (requires a valid local model path)
    MODEL_PATH = "./Models/goemotions"
    story = "He felt happy this morning. Then he got a sudden call that worried him. Later he breathed a sigh of relief."
    ec = EmotionCore(MODEL_PATH)
    res = ec.analyze_story(story, debug=True)
    ec.save_result(res, "demo_emotion_result.json")
    print("Saved demo_emotion_result.json")