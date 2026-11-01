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

# ---------------------------
# render MIDI -> WAV
# ---------------------------
# def render_midi_to_wav(midi_obj, sf2_path, out_wav_path):
#     tmp_midi = os.path.join(TMP_DIR, f"tmp_{os.getpid()}_{int(datetime.now().timestamp()*1000)}.mid")
#     midi_obj.write(tmp_midi)
#     cmd = ["fluidsynth", "-F", out_wav_path, "-ni", sf2_path, tmp_midi]
#     proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     if proc.returncode != 0:
#         print("Fluidsynth error:", proc.stderr)
#         raise RuntimeError(f"Fluidsynth failed: {out_wav_path}")
#     try:
#         os.remove(tmp_midi)
#     except Exception:
#         pass

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

INSTRUMENT_CANDIDATES = {
    "piano": ["Grand Piano","Piano","Acoustic Grand Piano"],
    "violin": ["Violin","Solo Violin"],
    "strings": ["Strings","String Ensemble","Orchestral Strings"],
    "pad": ["Pad","Warm Pad","Atmosphere"],
    "bass": ["Acoustic Bass","Fingered Bass","Synth Bass"],
    "choir": ["Choir","Ahh","Voice"]}


# ---------------------------
# mix WAVs
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
# Thread اجرای pipeline
# ---------------------------
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
            T, V, A = sample_splines(v_spline, a_spline,duration_sec=self.duration_sec ,samples_per_sec = sample_per_sec )
            K, Tau,Theta = compute_frenet_quantities(v_spline, a_spline, T)

            self.progress.emit(25)
            role_curves = generate_role_curves_for_all_roles(V, A, K, Tau, T)

            self.progress.emit(40)
            pm, meta = compose_from_geometry_output(
                role_curves,
                fingerprint={"V_points": V_points, "A_points": A_points, "entropy":0.5,"flow_complexity":0.5},
                duration_sec=self.duration_sec
            )

            self.progress.emit(60)
            midi_path = os.path.join(out_folder, "story_output.mid")
            pm.write(midi_path)

            wav_path = os.path.join(out_folder, "story_output.wav")
            render_midi_to_wav(pm, SF2_FILES[0], wav_path)

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