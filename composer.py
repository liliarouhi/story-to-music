# composer.py
"""
Musical Engine (composer.py)

وظایف:
- انتخاب key/mode بر اساس fingerprint (entropy, flow_complexity, mean V/A)
- تولید progression (non-repetitive, adaptive harmonic rhythm)
- SpiralArray (simplified) برای اندازه‌گیری "tonal distance" و انتخاب آکوردهای نزدیک
- map منحنی‌های نقش به نت MIDI با quantize روی scale
- ایجاد voice-leading ساده برای آکوردها (4-part SATB style)
- تولید pretty_midi object برای export
"""

import math
import numpy as np
import pretty_midi
from scipy.signal import medfilt
from collections import deque

# ---------------------------
# Scale intervals
# ---------------------------
MAJOR = [0,2,4,5,7,9,11]
MINOR = [0,2,3,5,7,8,10]
PENTATONIC_MAJOR = [0,2,4,7,9]
PENTATONIC_MINOR = [0,3,5,7,10]

# ---------------------------
# Simple Spiral Array (tonal distance)
# ---------------------------
class SpiralArray:
    """
    Lightweight SpiralArray: map pitch class to 3D point along helix.
    Use Euclidean distance between pitch classes to approximate tonal closeness.
    """
    def __init__(self):
        self.pc_pos = {}
        for pc in range(12):
            ang = 2*math.pi*pc/12.0
            r = 1.0 + 0.15*math.cos(3*ang)  # slight radius modulation
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            z = 0.12 * pc
            self.pc_pos[pc] = np.array([x,y,z])

    def distance_pc(self, pc_a, pc_b):
        a = self.pc_pos[pc_a % 12]
        b = self.pc_pos[pc_b % 12]
        return np.linalg.norm(a - b)

    def chord_score(self, pitch, chord): 
        # avg distance from pitch to chord tones (all pitch classes)
        d = np.mean([self.distance_pc(pitch, c) for c in chord])
        return d

SPIRAL = SpiralArray()

# ---------------------------
# Helper: scale generation
# ---------------------------
def get_scale(root_note=60, mode="major"):
    """
    return array of MIDI notes covering several octaves for a scale
    root_note: midi of root at some reference octave (e.g., C4=60)
    mode: "major" or "minor" (or "pent_major"/"pent_minor")
    """
    if mode == "major":
        intervals = MAJOR
    elif mode == "minor":
        intervals = MINOR
    elif mode == "pent_major":
        intervals = PENTATONIC_MAJOR
    elif mode == "pent_minor":
        intervals = PENTATONIC_MINOR
    else:
        intervals = MAJOR
    # produce three octaves around root
    notes = []
    for octv in (-1,0,1,2):
        base = root_note + 12*octv
        for i in intervals:
            notes.append(base + i)
    notes = np.array(notes, dtype=int)
    return notes

# ---------------------------
# Key selection based on fingerprint + V/A statistics
# ---------------------------
def select_key_and_mode(fingerprint, V_points, A_points, explicit_preference=None):
    """
    Choose root MIDI note and mode.
    fingerprint contains entropy and flow_complexity among other fields.
    - Higher valence -> choose 'major' and brighter root
    - Lower valence -> 'minor'
    - entropy and flow shift root circle-of-fifths a bit
    """
    mean_v = float(np.mean(V_points))
    mean_a = float(np.mean(A_points))
    ent = fingerprint.get("entropy", 0.0)
    flow = fingerprint.get("flow_complexity", 0.0)

    # base mode
    mode = "major" if mean_v >= 0 else "minor"

    # choose root index (0=C,1=C#,...) shifted by fingerprint
    # scale shift uses flow+entropy to move root around circle of fifths
    shift = int(round((ent * 2.0 + flow * 0.5) * 3))  # tweak coefficients as needed
    # center around C major / A minor
    if mode == "major":
        root_pc = (0 + shift) % 12   # C-based
        root_midi = 60 + (root_pc - 0)  # C4 offset (will wrap)
    else:
        root_pc = (9 + shift) % 12   # A minor base (A=9)
        root_midi = 57 + (root_pc - 9)
    # optional explicit preference overrides
    if explicit_preference:
        root_midi = explicit_preference

    return int(root_midi), mode

# ---------------------------
# Chord progression generator (adaptive)
# ---------------------------
COMMON_PROGRESSIONS_MAJOR = [
    ["I","vi","IV","V"],
    ["I","V","vi","IV"],
    ["I","IV","V","I"],
    ["vi","IV","I","V"]
]
COMMON_PROGRESSIONS_MINOR = [
    ["i","VI","III","VII"],
    ["i","iv","VII","III"],
    ["i","VI","iv","V"]
]

# degree mapping helpers
DEGREE_TO_INDEX = {"I":0,"II":1,"III":2,"IV":3,"V":4,"VI":5,"VII":6,
                   "i":0,"ii":1,"iii":2,"iv":3,"v":4,"vi":5,"vii":6}

def progression_from_fingerprint(mean_v, mean_a, entropy, flow, length_bars=8):
    """
    Create progressive non-trivial chord-series.
    - mean_v: valence
    - mean_a: arousal (influences harmonic rhythm)
    - entropy/flow influence variety and occasional secondary dominants
    """
    mode = "major" if mean_v >= 0 else "minor"
    if mode == "major":
        base = COMMON_PROGRESSIONS_MAJOR[np.random.randint(len(COMMON_PROGRESSIONS_MAJOR))]
    else:
        base = COMMON_PROGRESSIONS_MINOR[np.random.randint(len(COMMON_PROGRESSIONS_MINOR))]
    # extend by rotating/augmenting with substitutions
    prog = []
    i = 0
    while len(prog) < length_bars:
        chord = base[i % len(base)]
        # occasional substitution if entropy high
        if np.random.rand() < min(0.5, entropy * 0.7):
            # substitute with II or iii etc.
            chord = np.random.choice(base)
        prog.append(chord)
        i += 1
    # apply occasional secondary dominants if arousal high
    if mean_a > 0.6 and np.random.rand() < 0.8:
        idx = np.random.randint(1, max(2, length_bars-1))
        prog[idx] = "V/V" if mode=="major" else "V"
    return prog, mode

# ---------------------------
# Map chord symbols to MIDI notes given root midi and mode
# ---------------------------
def chord_symbol_to_notes(sym, root_midi, mode="major"):
    """
    map 'I','V','vi','IV','V/V','ii' ... to MIDI triads (root, third, fifth)
    basic handling; V/V handled by mapping to applied dominant (simple)
    """
    scale_intervals = MAJOR if mode=="major" else MINOR
    # mapping degree to interval index
    mapping = {"I":0,"II":1,"III":2,"IV":3,"V":4,"VI":5,"VII":6,
               "i":0,"ii":1,"iii":2,"iv":3,"v":4,"vi":5,"vii":6}
    if sym == "V/V":
        # applied dominant: V of V -> compute V chord root and then primary V of that
        v_root_pc = (root_midi + scale_intervals[4]) % 12
        # take as major V in that key: return triad rooted at v_root_pc + 7 semitones etc.
        # Simplify: return triad built on (root_midi + scale_intervals[4])
        applied_root = root_midi + scale_intervals[4]
        return [applied_root, applied_root + 4, applied_root + 7]
    deg = sym
    deg_index = mapping.get(deg, 0)
    root = root_midi + scale_intervals[deg_index]
    third = root + scale_intervals[(deg_index + 2) % 7] - scale_intervals[0]
    fifth = root + scale_intervals[(deg_index + 4) % 7] - scale_intervals[0]
    return [int(root), int(third), int(fifth)]

# ---------------------------
# Quantize pitches to scale with SpiralArray preference
# ---------------------------
def quantize_to_scale(pitches, scale_notes):
    """
    For each desired pitch value (float), choose nearest pitch in scale_notes (numpy array)
    """
    out = []
    scale = np.array(scale_notes)
    for p in pitches:
        diffs = np.abs(scale - p)
        idx = int(np.argmin(diffs))
        out.append(int(scale[idx]))
    return np.array(out, dtype=int)

def nearest_chord_tone(pitch, chord):
    """
    given a pitch (midi), choose the chord tone (root/3/5 across octaves) nearest to it
    """
    cand = []
    for base in chord:
        cand.extend([base-12, base, base+12, base+24])
    cand = np.array(cand)
    idx = int(np.argmin(np.abs(cand - pitch)))
    return int(cand[idx])

# ---------------------------
# Build pretty_midi from role_curves produced by geometry module
# ---------------------------

def build_pretty_midi_from_role_curves(role_curves, fingerprint, V_points=None, A_points=None, duration_sec=30.0, sf2_program_map=None):
    # fallback برای V_points/A_points اگر None باشند
    if V_points is None or A_points is None:
        V_points = []
        A_points = []
        for r,d in role_curves.items():
            V_points.append(float(np.mean(d["V"])))
            A_points.append(float(np.mean(d["A"])))

    mean_v = float(np.mean(V_points))
    mean_a = float(np.mean(A_points))
    ent = fingerprint.get("entropy", 0.0)
    flow = fingerprint.get("flow_complexity", 0.0)

    # انتخاب key و mode با مقادیر امن
    root_midi, mode = select_key_and_mode(fingerprint, V_points, A_points)
    scale_notes = get_scale(root_midi, mode=mode)

    # تعیین progression و voicing
    progression, mode_prog = progression_from_fingerprint(mean_v, mean_a, ent, flow, length_bars=8)
    chord_seq = [chord_symbol_to_notes(sym, root_midi, mode=mode_prog) for sym in progression]

    # دوره زمانی هارمونیک
    base_hr = max(4.0, duration_sec / max(1, len(progression)))
    harmonic_period = max(1.0, base_hr * (1.0 - min(0.8, mean_a*0.6)))

    pm = pretty_midi.PrettyMIDI()
    metadata = {"progression": progression, "chord_voicings": chord_seq, "mode": mode_prog, "root_midi": root_midi}

    for role, data in role_curves.items():
        V = np.array(data["V"])
        A = np.array(data["A"])
        Tau = np.array(data["Tau"]) if "Tau" in data else np.zeros_like(V)
        Theta = np.array(data["Theta"]) if "Theta" in data else np.zeros_like(V)

        # تنظیم octave span و پایه بر اساس نقش
        if role == "piano":
            octave_span = 3; base_note = root_midi
        elif role == "violin":
            octave_span = 2; base_note = root_midi + 12
        elif role == "strings":
            octave_span = 2; base_note = root_midi
        elif role == "pad":
            octave_span = 1; base_note = root_midi - 12
        elif role == "bass":
            octave_span = 1; base_note = root_midi - 24
        elif role == "choir":
            octave_span = 2; base_note = root_midi + 12
        else:
            octave_span = 2; base_note = root_midi

        # تبدیل V و Theta به pitch هدف
        v_norm = (V - V.min()) / (V.max() - V.min() + 1e-12)
        theta_norm = (Theta - Theta.min()) / (Theta.max() - Theta.min() + 1e-12)
        weights = {"piano":(0.7,0.3),"violin":(0.4,0.6),"strings":(0.5,0.5),
                   "pad":(0.3,0.7),"bass":(0.9,0.1),"choir":(0.6,0.4)}
        wv, wt = weights.get(role, (0.6,0.4))
        combined = wv * v_norm + wt * theta_norm
        float_pitches = scale_notes[0] + combined * (scale_notes[-1] - scale_notes[0])
        quantized = quantize_to_scale(float_pitches, scale_notes)
        lo, hi = role_pitch_range_default(role)
        quantized = np.clip(quantized, lo, hi)

        # velocities از Tau و A
        tau_norm = (Tau - Tau.min()) / (Tau.max() - Tau.min() + 1e-12)
        a_norm = (A - A.min()) / (A.max() - A.min() + 1e-12)
        vel = 0.6 * tau_norm + 0.4 * a_norm
        vel_min, vel_max = role_vel_range_default(role)
        velocities = np.clip((vel_min + vel * (vel_max - vel_min)).astype(int), 20, 127)

        # onsets و durations
        N = len(quantized)
        onsets = np.linspace(0.0, duration_sec, N+1)[:-1]
        durations = []
        for i in range(N):
            if role in ("pad","choir","strings"):
                durations.append(min(duration_sec - onsets[i], max(0.5, duration_sec / max(6, N/2))))
            elif role == "piano":
                durations.append(min(duration_sec - onsets[i], max(0.12, duration_sec / N * 0.9)))
            elif role == "bass":
                durations.append(min(duration_sec - onsets[i], max(0.2, duration_sec / max(6, N/3))))
            else:
                durations.append(min(duration_sec - onsets[i], duration_sec / N))

        # ساخت instrument
        program = sf2_program_map.get(role, 0) if sf2_program_map else 0
        inst = pretty_midi.Instrument(program=int(program), is_drum=False, name=role)
        for pitch, velv, on, dur in zip(quantized, velocities, onsets, durations):
            note = pretty_midi.Note(velocity=int(velv), pitch=int(round(pitch)),
                                    start=float(on), end=float(min(duration_sec, on+dur*0.95)))
            inst.notes.append(note)
        pm.instruments.append(inst)

    return pm, metadata

# ---------------------------
# Defaults for role ranges & velocities (can be tuned)
# ---------------------------
def role_pitch_range_default(role):
    mapping = {
        "piano": (36,92),
        "violin": (60,84),
        "strings": (45,75),
        "pad": (40,78),
        "bass": (28,52),
        "choir": (50,76)
    }
    return mapping.get(role, (40,80))

def role_vel_range_default(role):
    mapping = {
        "piano": (60,120),
        "violin": (48,110),
        "strings": (40,100),
        "pad": (30,80),
        "bass": (50,110),
        "choir": (30,95)
    }
    return mapping.get(role, (45,100))

# ---------------------------
# Helper: simple API for end-to-end
# ---------------------------
# def compose_from_geometry_output(geometry_output, fingerprint, duration_sec, sf2_program_map=None):
#     """
#     geometry_output: dict from geometry_engine.generate_role_curves_for_all_roles
#     fingerprint: from EmotionCore.analyze_story
#     duration_sec: target duration
#     returns pretty_midi object and metadata
#     """
#     # collect original V_points/A_points from fingerprint if available; else compute means
#     V_points = fingerprint.get("V_points", None)
#     A_points = fingerprint.get("A_points", None)
#     if V_points is None or A_points is None:
#         # fallback: compute means from role curves
#         V_points = []
#         A_points = []
#         for r,d in geometry_output.items():
#             V_points.append(float(np.mean(d["V"])))
#             A_points.append(float(np.mean(d["A"])))
#     pm, meta = build_pretty_midi_from_role_curves(geometry_output, fingerprint, V_points, A_points, duration_sec, sf2_program_map=sf2_program_map)
#     return pm, meta
def compose_from_geometry_output(geometry_output, fingerprint, duration_sec, sf2_program_map=None):
    """
    geometry_output: dict from geometry_engine.generate_role_curves_for_all_roles
    fingerprint: from EmotionCore.analyze_story
    duration_sec: target duration
    returns pretty_midi object and metadata
    """
    # fallback: جمع‌آوری V_points / A_points
    V_points = fingerprint.get("V_points", None)
    A_points = fingerprint.get("A_points", None)
    if V_points is None or A_points is None:
        V_points = []
        A_points = []
        for r, d in geometry_output.items():
            V_points.append(float(np.mean(d["V"])))
            A_points.append(float(np.mean(d["A"])))

    pm, meta = build_pretty_midi_from_role_curves(
        geometry_output,
        fingerprint,
        V_points,
        A_points,
        duration_sec,
        sf2_program_map=sf2_program_map
    )
    return pm, meta


# ---------------------------
# Example quick test (if run standalone)
# ---------------------------
if __name__ == "__main__":
    # synthetic demo: requires geometry_engine demo to have produced 'roles'
    from geometry_engine import generate_role_curves_for_all_roles, build_splines_from_points, sample_splines, compute_frenet_quantities
    demo_V = [0.9, 0.4, -0.2, -0.8, 0.5]
    demo_A = [0.2, 0.6, 0.9, 0.3, 0.5]
    v_s, a_s = build_splines_from_points(demo_V, demo_A)
    T, Vc, Ac = sample_splines(v_s, a_s, duration_sec=30.0, samples_per_sec=60)
    K, Tau, Theta = compute_frenet_quantities(v_s, a_s, T)
    roles = generate_role_curves_for_all_roles(Vc, Ac, K, Tau, Theta)
    # fake fingerprint
    fingerprint = {"entropy": 1.2, "flow_complexity": 0.7, "V_points": demo_V, "A_points": demo_A}
    pm, meta = compose_from_geometry_output(roles, fingerprint, duration_sec=30.0)
    out_midi = "demo_compose.mid"
    pm.write(out_midi)
    print("Wrote demo MIDI:", out_midi)