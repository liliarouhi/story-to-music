
import os
import sys
import subprocess
from datetime import datetime
import tempfile
import numpy as np
import pretty_midi
import soundfile as sf
from PySide6.QtWidgets import (QApplication, QWidget, QLabel, QTextEdit,
                               QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
                               QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal

# ---------------------------
# پروژه ماژول‌ها
# ---------------------------
from core_emotion import EmotionExtractorLocal
from role_engine import build_role_phrases
from composer import compose_from_geometry_output
from geometry_engine import generate_role_curves_for_all_roles, build_splines_from_points, sample_splines, compute_frenet_quantities

# ---------------------------
# تنظیمات
# ---------------------------
PROJECT_DIR = os.path.expanduser("~/Desktop/music proj")
MODEL_PATH = os.path.join(PROJECT_DIR, "Models", "goemotions")
SOUNDFONT_DIR = os.path.join(PROJECT_DIR, "SoundFont")
SF2_FILES = [
    os.path.join(SOUNDFONT_DIR, "FluidR3_GM.sf2"),
    os.path.join(SOUNDFONT_DIR, "GeneralUser-GS.sf2"),
    os.path.join(SOUNDFONT_DIR, "MORCH.sf2"),
]
MIN_DURATION = 30
MAX_DURATION = 60
TMP_DIR = tempfile.gettempdir()

ROLE_PITCH_RANGE = {
    "piano": (48, 84),
    "violin": (60, 84),
    "strings": (48, 72),
    "pad": (40, 72),
    "bass": (28, 50),
    "choir": (50, 76),
}

INSTRUMENT_CANDIDATES = {
    "piano": ["Grand Piano","Piano","Acoustic Grand Piano"],
    "violin": ["Violin","Solo Violin"],
    "strings": ["Strings","String Ensemble","Orchestral Strings"],
    "pad": ["Pad","Warm Pad","Atmosphere"],
    "bass": ["Acoustic Bass","Fingered Bass","Synth Bass"],
    "choir": ["Choir","Ahh","Voice"]
}

# ---------------------------
# Render MIDI -> WAV
# ---------------------------
def render_midi_to_wav(midi_obj, sf2_path, out_wav_path, program_number=None):
    tmp_midi = os.path.join(TMP_DIR, f"tmp_{os.getpid()}_{int(datetime.now().timestamp()*1000)}.mid")
    if program_number is not None:
        for inst in midi_obj.instruments:
            inst.program = int(program_number)
    midi_obj.write(tmp_midi)
    cmd = ["fluidsynth", "-F", out_wav_path, "-ni", sf2_path, tmp_midi]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("fluidsynth error:", proc.stderr)
        raise RuntimeError(f"fluidsynth rendering failed for {out_wav_path}")
    try:
        os.remove(tmp_midi)
    except Exception:
        pass


MAJOR = [0,2,4,5,7,9,11]
MINOR = [0,2,3,5,7,8,10]

def map_to_midi_scale(base_note, normalized_values, mode="major", octave_span=2):
    """
    normalized_values: array in [0,1]
    base_note: MIDI root (e.g., 60 for C4)
    returns integer MIDI notes
    """
    scale = MAJOR if mode=="major" else MINOR
    n_steps = len(scale) * octave_span
    indices = (normalized_values * (n_steps-1)).astype(int)
    notes = []
    for idx in indices:
        octave = idx // len(scale)
        degree = idx % len(scale)
        midi = base_note + octave*12 + scale[degree]
        notes.append(int(midi))
    return np.array(notes, dtype=int)
# ---------------------------
# Mix WAVs
# ---------------------------
def mix_wavs(wav_paths, out_path):
    audios = []
    sr = None
    for p in wav_paths:
        y, s = sf.read(p)
        if sr is None:
            sr = s
        else:
            if s != sr:
                raise RuntimeError("Sample rate mismatch")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        audios.append(y.astype(np.float64))
    maxlen = max(len(a) for a in audios)
    mix = np.zeros(maxlen, dtype=np.float64)
    for a in audios:
        mix[:len(a)] += a
    peak = np.max(np.abs(mix)) + 1e-12
    mix = mix / peak * 0.95
    sf.write(out_path, mix, sr)
    return out_path

# ---------------------------
# SF2 Presets scanning
# ---------------------------
def get_sf2_presets(sf2_path, timeout=6):
    if not os.path.exists(sf2_path):
        return {}
    cmd = ["fluidsynth", sf2_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        p.stdin.write("inst 1\nquit\n")
        p.stdin.flush()
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
    presets = {}
    for ln in out.splitlines():
        ln = ln.strip()
        if len(ln) >= 11 and ln[0:3].isdigit() and ln[3] == '-':
            try:
                bank = int(ln[0:3])
                prog = int(ln[4:7])
                name = ln[8:].strip()
                presets[(bank, prog)] = name
            except Exception:
                continue
    return presets

def find_best_presets(sf2_files, instrument_candidates):
    sf2_presets = {sf2: get_sf2_presets(sf2) for sf2 in sf2_files}
    result = {}
    for role, patterns in instrument_candidates.items():
        found = None
        for pat in patterns:
            for sf2,presets in sf2_presets.items():
                for (b,p), nm in presets.items():
                    if pat.lower() in nm.lower():
                        found = {'sf2': sf2, 'bank': b, 'prog': p, 'name': nm}
                        break
                if found: break
            if found: break
        result[role] = found
    return result

# ---------------------------
# Pipeline Thread
# ---------------------------
def velocities_from_torsion(torsion_segment):
    t = np.array(torsion_segment)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-8)
    vel = 48 + (t_norm * (100-48))
    return np.round(vel).astype(int)

def build_midi_for_role(pitches, velocities, duration, program=0, role_name="role"):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=int(program), is_drum=False, name=role_name)
    N = len(pitches)
    if N == 0:
        return pm
    note_len = float(duration) / float(N)
    t = 0.0
    for i in range(N):
        start = t
        end = min(duration, t + note_len*0.9)
        note = pretty_midi.Note(velocity=int(velocities[i]), pitch=int(pitches[i]), start=start, end=end)
        inst.notes.append(note)
        t += note_len
    pm.instruments.append(inst)
    return pm

class PipelineWorker(QThread):
    progress = Signal(int)
    finished = Signal(str, str)  # wav, midi

    def __init__(self, story_text, duration_sec):
        super().__init__()
        self.story_text = story_text
        self.duration_sec = duration_sec

    def run(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_folder = os.path.join(PROJECT_DIR, f"run_{timestamp}")
            os.makedirs(out_folder, exist_ok=True)

            self.progress.emit(5)
            extractor = EmotionExtractorLocal(MODEL_PATH)

            # split sentences
            sentences = [s.strip() for s in self.story_text.replace("؟",".").replace("!","." ).replace("؛",".").split(".") if s.strip()]
            if len(sentences)<2:
                sentences = [self.story_text.strip(), self.story_text.strip()]

            V_points, A_points = [], []
            for s in sentences:
                _, v, a, _ = extractor.get_VA(s)
                V_points.append(v)
                A_points.append(a)

            self.progress.emit(15)
            v_spline, a_spline = build_splines_from_points(V_points, A_points)
            N_samples = max(800,int(self.duration_sec*40))
            sample_per_sec = int(N_samples/self.duration_sec)
            T, V, A = sample_splines(v_spline, a_spline, duration_sec=self.duration_sec, samples_per_sec=sample_per_sec)
            K, Tau, Theta = compute_frenet_quantities(v_spline, a_spline, T)

            self.progress.emit(25)
            role_curves = generate_role_curves_for_all_roles(V, A, K, Tau, Theta)

            self.progress.emit(35)
            presets_map = find_best_presets(SF2_FILES, INSTRUMENT_CANDIDATES)
            print("Found presets for roles:", presets_map)

            self.progress.emit(40)
            # ایجاد MIDI برای هر نقش
            role_midis = {}
            for role, curves in role_curves.items():
                pitches = map_to_midi_scale(
                    base_note=ROLE_PITCH_RANGE[role][0],
                    normalized_values=(curves["V"] - curves["V"].min()) / (np.ptp(curves["V"]) + 1e-8),
                    mode="major" if np.mean(curves["V"]) >= 0 else "minor",
                    octave_span=2
                )
                pitches = np.clip(pitches, ROLE_PITCH_RANGE[role][0], ROLE_PITCH_RANGE[role][1])
                velocities = velocities_from_torsion(curves["Tau"])
                midi_obj = build_midi_for_role(pitches, velocities, duration=self.duration_sec,
                                               program=presets_map[role]['prog'] if presets_map.get(role) else 0,
                                               role_name=role)
                role_midis[role] = midi_obj

            self.progress.emit(60)
            # رندر هر نقش جداگانه
            rendered_wavs = []
            for role, midi_obj in role_midis.items():
                preset = presets_map.get(role, {'sf2': SF2_FILES[0], 'prog': 0})
                out_wav = os.path.join(out_folder, f"{role}.wav")
                render_midi_to_wav(midi_obj, preset['sf2'], out_wav, program_number=preset['prog'])
                if role == "choir":
                    y, sr = sf.read(out_wav)
                    y = y * 0.15
                    sf.write(out_wav, y, sr)
                rendered_wavs.append(out_wav)

            # میکس نهایی WAV
            wav_path = os.path.join(out_folder, "story_output.wav")
            mix_wavs(rendered_wavs, wav_path)

            # میکس نهایی MIDI
            combined_midi = pretty_midi.PrettyMIDI()
            for midi_obj in role_midis.values():
                for inst in midi_obj.instruments:
                    combined_midi.instruments.append(inst)
            midi_path = os.path.join(out_folder, "story_output.mid")
            combined_midi.write(midi_path)

            self.progress.emit(100)
            self.finished.emit(wav_path, midi_path)

        except Exception as ex:
            print("Pipeline error:", ex)
            self.finished.emit("", "")

# ---------------------------
# GUI
# ---------------------------
class StoryMusicGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Story to Music — GUI Final")
        self.setGeometry(200,100,700,560)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.story_label = QLabel("Story / Emotions:")
        self.story_text = QTextEdit()
        layout.addWidget(self.story_label)
        layout.addWidget(self.story_text)

        dur_layout = QHBoxLayout()
        self.dur_label = QLabel("Duration (30–60 s):")
        self.dur_input = QLineEdit("40")
        dur_layout.addWidget(self.dur_label)
        dur_layout.addWidget(self.dur_input)
        layout.addLayout(dur_layout)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_pipeline)
        layout.addWidget(self.start_btn)
        self.setLayout(layout)

    def start_pipeline(self):
        text = self.story_text.toPlainText().strip()
        try:
            dur = float(self.dur_input.text())
        except:
            dur = 40.0
        dur = max(min(dur, MAX_DURATION), MIN_DURATION)
        self.progress.setValue(0)

        self.worker = PipelineWorker(text, dur)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.done)
        self.worker.start()
        self.start_btn.setEnabled(False)

    def done(self, wav_path, midi_path):
        self.start_btn.setEnabled(True)
        self.progress.setValue(100)
        if wav_path:
            print("Processing completed.")
            print("WAV:", wav_path)
            print("MIDI:", midi_path)
        else:
            print("Processing finished but no output. Check logs.")

# ---------------------------
# اجرا
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = StoryMusicGUI()
    gui.show()
    sys.exit(app.exec())