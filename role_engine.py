#role_engine.py
import numpy as np
from scipy.ndimage import gaussian_filter1d

# -----------------------------
# Emotion → Rhythm Family mapping
# -----------------------------
RHYTHM_FAMILIES = {
    "joy":      ["joyful","excited","impressed","proud","confident"],
    "sad":      ["sad","lonely","devastated","disappointed","nostalgic"],
    "anger":    ["angry","furious","annoyed","disgusted"],
    "fear":     ["afraid","terrified","anxious","apprehensive"],
    "calm":     ["content","trusting","caring","faithful"],
    "tension":  ["anticipating","surprised","jealous","embarrassed"],
    "dark":     ["ashamed","guilty","sentimental"],
}

def map_emotion_to_family(dominant_emotion):
    for fam, arr in RHYTHM_FAMILIES.items():
        if dominant_emotion in arr:
            return fam
    return "calm"

# -----------------------------
# Rhythm templates
# -----------------------------
RHYTHM_TEMPLATES = {
    "joy":      [1,1,0.5,1,0.5,1],
    "sad":      [1.5,1.5,2,1.5],
    "anger":    [0.5,0.5,0.25,0.25,1,0.5],
    "fear":     [0.25,0.5,0.25,1,0.5],
    "calm":     [2,2,1.5,2],
    "tension":  [1,0.5,1,0.5,2],
    "dark":     [1,1,1,1]
}

def expand_rhythm(template, total_time):
    result = []
    t = 0
    i = 0
    while t < total_time:
        dur = template[i % len(template)]
        result.append(dur)
        t += dur
        i += 1
    return result

def smooth_curve(curve, factor=4.0):
    return gaussian_filter1d(curve, sigma=factor)

def build_role_phrases(role_curves, dominant_emotion, progression, duration_sec):
    family = map_emotion_to_family(dominant_emotion)
    base_rhythm = RHYTHM_TEMPLATES[family]
    rhythm = expand_rhythm(base_rhythm, duration_sec)

    phrases = {}
    for role, data in role_curves.items():
        V = smooth_curve(data["V"], 3)
        A = smooth_curve(data["A"], 3)
        Theta = smooth_curve(data["Theta"], 3)

        if role == "bass":
            pitches = np.zeros(len(rhythm))
            for i,r in enumerate(rhythm):
                chord = progression[i % len(progression)]
                pitches[i] = chord[0] - 12
        elif role in ("pad","strings"):
            pitches = np.zeros(len(rhythm))
            for i,r in enumerate(rhythm):
                chord = progression[i % len(progression)]
                idx = int((Theta[int(i*len(Theta)/len(rhythm))]*3)) % 3
                pitches[i] = chord[idx]
        elif role == "violin":
            pitches = (V * 12 + progression[0][0]).astype(int)
        elif role == "choir":
            pitches = (V*6 + (progression[0][0]+7)).astype(int)
        elif role == "piano":
            pitches = (0.6*V + 0.4*Theta)
            pitches = (pitches * 10 + progression[0][0]).astype(int)
        else:
            pitches = (V*12 + progression[0][0]).astype(int)

        vel = (A - A.min())/(A.max()-A.min()+1e-9)
        vel = (vel * 60 + 50).astype(int)

        idx = np.linspace(0, len(pitches)-1, len(rhythm)).astype(int)
        pitches = pitches[idx]
        vel = vel[idx]

        pitches = np.clip(pitches, 28, 100)

        onsets, durations = [], []
        t = 0
        for d in rhythm:
            onsets.append(t)
            durations.append(d*0.9)
            t += d

        phrases[role] = {
            "pitches": pitches,
            "velocities": vel,
            "durations": durations,
            "onsets": onsets,
        }

    return phrases