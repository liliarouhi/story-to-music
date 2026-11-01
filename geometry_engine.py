# geometry_engine.py
"""
Geometry Engine for StoryToAdvancedMusic (geometry_engine.py)

وظایف:
- ساخت CubicSpline برای V(t) و A(t)
- نمونه‌برداری با رزولوشن بالا
- محاسبه مشتق‌های r', r'', r''' و بردارهای Frenet-like
- محاسبه curvature (K), torsion (Tau), theta
- smoothingِ قوی برای حذف پرش/نویز
- تولید منحنی‌های اختصاصی برای هر role (perturbations, phase shifts, envelopes)
- توابع کمکی برای ذخیره و رسم نمودارها (VA curves, K/Tau, role curves)

فرمول‌ها:
- r(t) = [1, V(t), A(t)]
- curvature: K = || r' × r'' || / ||r'||^3
- torsion: Tau = ( (r' × r'') · r''' ) / || r' × r'' ||^2
- theta: angle-like measure = atan2( || r' × r'' ||, r'·r'' )

توصیه‌ها:
- از نمونه‌برداری کافی (samples_per_sec ~ 40..120) استفاده کن
- sigma smoothing را تنظیم کن تا نمودارها خیلی تخت یا پرنوسان نشوند
"""
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os

# ---------------------------
# تنظیمات پیش‌فرض قابل تنظیم
# ---------------------------
DEFAULT_SAMPLES_PER_SEC = 60
DEFAULT_SIGMA_K = 5
DEFAULT_SIGMA_TAU = 4
DEFAULT_SIGMA_THETA = 4

# ---------------------------
# Helpers: spline build & sample
# ---------------------------
def build_splines_from_points(V_points, A_points, bc_type="natural"):
    """
    ورودی:
      V_points, A_points : لیست یا آرایه‌ای از مقادیر valence و arousal (طول n_sentences)
    خروجی:
      v_spline, a_spline : اشیاء scipy.interpolate.CubicSpline
    """
    n = max(2, len(V_points))
    t = np.linspace(0.0, 1.0, n)
    v_spline = CubicSpline(t, V_points, bc_type=bc_type)
    a_spline = CubicSpline(t, A_points, bc_type=bc_type)
    return v_spline, a_spline

def sample_splines(v_spline, a_spline, duration_sec, samples_per_sec=DEFAULT_SAMPLES_PER_SEC):
    """
    نمونه‌برداری پیوسته از V(t) و A(t).
    خروجی:
      T: نرمال‌شده زمان [0,1]
      V_curve, A_curve : آرایه‌های length N
    """
    N = max(200, int(duration_sec * samples_per_sec))
    T = np.linspace(0.0, 1.0, N)
    V = v_spline(T)
    A = a_spline(T)
    return T, V, A

# ---------------------------
# Derivatives & Frenet-like quantities
# ---------------------------
def compute_derivatives_arrays(v_spline, a_spline, T):
    """
    بازگرداندن r1, r2, r3 به صورت آرایه‌های شکل (3, N)
    r(t) = [1, V(t), A(t)]
    r1 = [1, V', A']
    r2 = [0, V'', A'']
    r3 = [0, V''', A''']
    """
    r1 = np.vstack([np.ones_like(T), v_spline(T, 1), a_spline(T, 1)])
    r2 = np.vstack([np.zeros_like(T), v_spline(T, 2), a_spline(T, 2)])
    r3 = np.vstack([np.zeros_like(T), v_spline(T, 3), a_spline(T, 3)])
    return r1, r2, r3

def compute_frenet_quantities(v_spline, a_spline, T,
                              sigma_K=DEFAULT_SIGMA_K,
                              sigma_Tau=DEFAULT_SIGMA_TAU,
                              sigma_Theta=DEFAULT_SIGMA_THETA):
    """
    محاسبه curvature (K), torsion (Tau) و theta برای هر نمونه در T.
    smoothing با gaussian_filter1d اعمال می‌شود تا نویز حذف شود.
    خروجی‌ها:
      K, Tau, Theta  (هر کدام آرایه length N)
    """
    r1, r2, r3 = compute_derivatives_arrays(v_spline, a_spline, T)
    # محاسبات برداری به صورت برداری‌شده
    # cross product r1 x r2 برای هر ستون
    cross = np.cross(r1.T, r2.T)  # shape (N,3)
    norm_r1 = np.linalg.norm(r1.T, axis=1) + 1e-12
    norm_cross = np.linalg.norm(cross, axis=1) + 1e-12

    # curvature
    K = norm_cross / (norm_r1**3 + 1e-12)

    # torsion: (cross · r3) / ||cross||^2
    dot_cross_r3 = np.einsum('ij,ij->i', cross, r3.T)
    Tau = dot_cross_r3 / (norm_cross**2 + 1e-12)

    # theta: atan2(||r' x r''||, r'·r'')
    dot_r1_r2 = np.einsum('ij,ij->i', r1.T, r2.T)
    Theta = np.arctan2(norm_cross, dot_r1_r2 + 1e-12)

    # smoothing
    K_s = gaussian_filter1d(K, sigma=sigma_K)
    Tau_s = gaussian_filter1d(Tau, sigma=sigma_Tau)
    Theta_s = gaussian_filter1d(Theta, sigma=sigma_Theta)

    return K_s, Tau_s, Theta_s

# ---------------------------
# Role-specific curve generation
# ---------------------------

def make_role_curves(V_curve, A_curve, K, Tau, Theta,
                     role="piano",
                     amplitude_noise=0.06,
                     phase_shift=0.0,
                     smoothing_sigma=3):
    """
    برای هر نقش (role) یک منحنی V_role و A_role تولید می‌کند که ترکیبی از سیگنال‌های پایه است.
    پارامترها قابل تنظیمند تا هر ساز منحنی منحصر به فرد بگیرد.

    ورودی‌ها:
      V_curve, A_curve, K, Tau, Theta : آرایه‌های هم‌اندازه
      role: یکی از ("piano","violin","strings","pad","bass","choir")
    خروجی:
      V_r, A_r (آرایه‌های هم‌اندازه)
    """
    N = len(V_curve)
    t = np.linspace(0, 1, N)

    # پایه‌ها: ترکیب وزن‌دار از V, A, K, Tau, Theta
    if role == "piano":
        base = 0.5 * normalize_array(V_curve) + 0.35 * normalize_array(K) + 0.15 * normalize_array(np.abs(np.gradient(V_curve)))
        amp = 0.06; ph = 0.0; sigma = smoothing_sigma
    elif role == "violin":
        base = 0.45 * normalize_array(np.gradient(V_curve)) + 0.35 * normalize_array(Tau) + 0.20 * normalize_array(Theta)
        amp = 0.08; ph = 0.4; sigma = smoothing_sigma
    elif role == "strings":
        base = 0.4 * normalize_array(K) + 0.45 * normalize_array(A_curve) + 0.15 * normalize_array(np.roll(K, int(0.03*N)))
        amp = 0.10; ph = 0.6; sigma = smoothing_sigma + 1
    elif role == "pad":
        base = 0.6 * normalize_array(A_curve) + 0.3 * normalize_array(V_curve)
        amp = 0.12; ph = 0.2; sigma = smoothing_sigma + 2
    elif role == "bass":
        base = 0.5 * normalize_array(K) + 0.3 * normalize_array(np.abs(np.gradient(K)))
        amp = 0.05; ph = 0.1; sigma = smoothing_sigma
    elif role == "choir":
        base = 0.7 * normalize_array(V_curve) + 0.2 * normalize_array(A_curve)
        amp = 0.08; ph = 0.15; sigma = smoothing_sigma + 3
    else:
        base = normalize_array(K)
        amp = amplitude_noise; ph = phase_shift; sigma = smoothing_sigma

    # اضافه کردن نویز فاز و سینوسی با امپلیتوود کم تا منحنی متفاوت شود اما smooth بماند
    perturb = amp * np.sin(2.0 * np.pi * (2.0 * t + ph)) + amp * 0.5 * np.random.normal(scale=1.0, size=N)
    V_r = base + perturb
    # A_role را معمولا بر اساس A_curve و بخش‌هایی از Theta/Tau می‌سازیم
    A_r = 0.6 * normalize_array(A_curve) + 0.3 * normalize_array(Tau) + 0.1 * normalize_array(Theta)
    A_r += 0.5 * perturb  # همبستگی ملایم با V_r

    # smoothing نهایی
    V_r = gaussian_filter1d(V_r, sigma=sigma)
    A_r = gaussian_filter1d(A_r, sigma=sigma)
    # بازگشت به بازه [-1,1] حفظ نسبی دامنه
    V_r = scale_to_range(V_r, -1.0, 1.0)
    A_r = scale_to_range(A_r, -1.0, 1.0)
    return V_r, A_r

def generate_role_curves_for_all_roles(V_curve, A_curve, K, Tau, Theta,
                                       roles=("piano","violin","strings","pad","bass","choir"),
                                       **kwargs):
    """
    تولید مجموعه منحنی‌ها برای همه نقش‌ها و برگرداندن یک دیکشنری.
    خروجی:
      { role: {"V":..., "A":..., "K":..., "Tau":..., "Theta":...}, ... }
    K/Tau/Theta نیز برای هر role می‌تواند sub-sampled یا filtered شود.
    """
    out = {}
    for i, role in enumerate(roles):
        # phase shift هر نقش را بر اساس اندیس ایجاد می‌کنیم تا align نشوند
        ph = (i * 0.37) % 1.0
        V_r, A_r = make_role_curves(V_curve, A_curve, K, Tau, Theta,
                                    role=role,
                                    amplitude_noise=kwargs.get("amplitude_noise", 0.06 + 0.02*i),
                                    phase_shift=ph,
                                    smoothing_sigma=kwargs.get("smoothing_sigma", 3))
        # محاسبه K/Tau/Theta محلی برای نقش (در صورت نیاز)
        try:
            v_spline_loc = CubicSpline(np.linspace(0,1,len(V_r)), V_r, bc_type="natural")
            a_spline_loc = CubicSpline(np.linspace(0,1,len(A_r)), A_r, bc_type="natural")
            T_loc = np.linspace(0,1,len(V_r))
            K_loc, Tau_loc, Theta_loc = compute_frenet_quantities(v_spline_loc, a_spline_loc, T_loc)
        except Exception:
            K_loc = gaussian_filter1d(K, sigma=2)
            Tau_loc = gaussian_filter1d(Tau, sigma=2)
            Theta_loc = gaussian_filter1d(Theta, sigma=2)
        out[role] = {"V": V_r, "A": A_r, "K": K_loc, "Tau": Tau_loc, "Theta": Theta_loc}
    return out

# ---------------------------
# Utilities: normalization & scaling & plotting
# ---------------------------
def normalize_array(x):
    x = np.array(x, dtype=float)
    mn = np.min(x); mx = np.max(x)
    if np.isclose(mx, mn):
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def scale_to_range(x, lo=-1.0, hi=1.0):
    x = np.array(x, dtype=float)
    if np.all(np.isclose(x, x.flat[0])):
        return np.clip(np.ones_like(x) * float(x.flat[0]), lo, hi)
    y = (x - x.min()) / (x.max() - x.min() + 1e-12)
    return y * (hi - lo) + lo

def plot_va_and_geometry(T, V_curve, A_curve, K, Tau, Theta, out_folder=".", prefix="va_geo"):
    """
    ذخیره ۳ شکل:
      - Valence & Arousal over time
      - Curvature & Torsion over time
      - Theta over time
    """
    os.makedirs(out_folder, exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(T, V_curve, label="Valence (V)")
    plt.plot(T, A_curve, label="Arousal (A)")
    plt.xlabel("Normalized time")
    plt.ylabel("V/A")
    plt.title("Valence & Arousal curves")
    plt.legend()
    p1 = os.path.join(out_folder, f"{prefix}_VA.png")
    plt.savefig(p1, dpi=300); plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(T, K, label="Curvature (K)")
    plt.plot(T, Tau, label="Torsion (Tau)")
    plt.xlabel("Normalized time")
    plt.ylabel("K / Tau")
    plt.title("Curvature & Torsion")
    plt.legend()
    p2 = os.path.join(out_folder, f"{prefix}_K_Tau.png")
    plt.savefig(p2, dpi=300); plt.close()

    plt.figure(figsize=(10,3))
    plt.plot(T, Theta, label="Theta")
    plt.xlabel("Normalized time")
    plt.ylabel("Theta")
    plt.title("Theta (curvature angle)")
    plt.legend()
    p3 = os.path.join(out_folder, f"{prefix}_Theta.png")
    plt.savefig(p3, dpi=300); plt.close()

    return p1, p2, p3

def plot_role_curves(role_curves, out_folder=".", prefix="role"):
    """
    role_curves: dict as returned from generate_role_curves_for_all_roles
    ذخیره نمودار pitch-like (V mapped) برای هر نقش
    """
    os.makedirs(out_folder, exist_ok=True)
    figs = []
    for role, d in role_curves.items():
        Tloc = np.linspace(0,1,len(d["V"]))
        plt.figure(figsize=(10,3))
        plt.plot(Tloc, d["V"], label=f"{role} V")
        plt.plot(Tloc, d["A"], label=f"{role} A", alpha=0.7)
        plt.title(f"Role curves: {role}")
        plt.legend()
        fpath = os.path.join(out_folder, f"{prefix}_{role}.png")
        plt.savefig(fpath, dpi=300); plt.close()
        figs.append(fpath)
    return figs

# ---------------------------
# Example / quick test (standalone)
# ---------------------------
if __name__ == "__main__":
    # quick demo with synthetic V/A points if run directly
    demo_V = [0.9, 0.7, -0.2, -0.8, -0.1, 0.5]   # example valence per sentence
    demo_A = [0.2, 0.5, 0.8, 0.6, 0.1, 0.4]     # example arousal per sentence
    duration = 40.0
    print("Building splines...")
    v_s, a_s = build_splines_from_points(demo_V, demo_A)
    T, Vc, Ac = sample_splines(v_s, a_s, duration, samples_per_sec=80)
    print("Computing Frenet quantities...")
    K, Tau, Theta = compute_frenet_quantities(v_s, a_s, T)
    print("Generating role curves...")
    roles = generate_role_curves_for_all_roles(Vc, Ac, K, Tau, Theta)
    out = "geometry_demo_output"
    print("Plotting shapes to folder:", out)
    p1,p2,p3 = plot_va_and_geometry(T, Vc, Ac, K, Tau, Theta, out_folder=out)
    figs = plot_role_curves(roles, out_folder=out)
    print("Saved:", p1, p2, p3)
    for f in figs:
        print("Saved:", f)
    print("Done demo.")