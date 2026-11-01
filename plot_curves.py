"""Saves the diagnostic plots of a run (V/A, curvature/torsion/theta, per-role curves)."""
from geometry_engine import plot_va_and_geometry, plot_role_curves


def save_curves(T, V, A, K, Tau, Theta, role_curves, out_folder):
    plot_va_and_geometry(T, V, A, K, Tau, Theta, out_folder=out_folder, prefix="VA_Geo")
    plot_role_curves(role_curves, out_folder=out_folder, prefix="RoleCurves")
