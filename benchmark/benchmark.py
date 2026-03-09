"""
FreeTTES Benchmark Test
=======================
Modelliert 48 Stunden mit Beladung (t=0..23h) und Entladung (t=24..47h).
Speichert Ergebnisse (CSV, JSON-Summary) und drei Plots im Unterordner results/.

Verwendung:
    python benchmark/benchmark.py          (aus dem Workspace-Root)
    python benchmark.py                    (aus dem benchmark/-Ordner)
"""

import sys
import os
import time
import csv
import json
from pathlib import Path

# ── Pfade ────────────────────────────────────────────────────────────────────
BENCHMARK_DIR = Path(__file__).resolve().parent
SRC_DIR       = BENCHMARK_DIR.parent / "src"

# Ausgabeordner: Standard = baseline/
# Für Optimierungsvergleiche z.B.: RUN_NAME=opt_v1 python benchmark.py
RUN_NAME    = os.environ.get("RUN_NAME", "baseline")
RESULTS_DIR = BENCHMARK_DIR / "results" / RUN_NAME

sys.path.insert(0, str(SRC_DIR))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import FreeTTES_model as model
import numpy as np
import matplotlib
matplotlib.use("Agg")   # kein GUI-Fenster nötig
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Stoffwert Hilfsfunktion (inline, kein Zugriff auf private Modulfunktionen) ─
def _rho(T: float) -> float:
    """Dichte von Wasser in kg/m³ (Polynom-Fit aus dem Modell)."""
    return -2.525726e-3 * T**2 - 2.123038e-1 * T + 1.005011e3

def m3h_to_kgs(m3h: float, T_ref: float) -> float:
    return _rho(T_ref) * m3h / 3600.0

def GJ_to_MWh(GJ: float) -> float:
    return GJ / 3.6

# ── Simulationsparameter ──────────────────────────────────────────────────────
DT_S         = 3600       # Zeitschritt in Sekunden
FLOW_M3H     = 300.0      # Volumenstrom [m³/h] für Beladung und Entladung
T_CHARGE_IN  = 90.0       # °C – Eintrittstemperatur beim Beladen
T_DISCH_IN   = 60.0       # °C – Eintrittstemperatur beim Entladen
T_AMB        = 10.0       # °C – Umgebungstemperatur

# Phasengrenzen [h] – exklusives Ende
T_END_CHARGE  = 24        # Beladung:  t=0   .. T_END_CHARGE-1
T_END_IDLE    = 36        # Standzeit: t=T_END_CHARGE .. T_END_IDLE-1
T_END_DISCH   = 60        # Entladung: t=T_END_IDLE   .. T_END_DISCH-1
N_HOURS = T_END_DISCH

# Zeitschritte, für die Temperaturprofile gespeichert werden
PROFILE_SNAPSHOTS = [0, 6, 12, 18, 23, 24, 30, 35, 36, 42, 48, 54, 59]

m_charge    = m3h_to_kgs(FLOW_M3H, T_CHARGE_IN)
m_discharge = m3h_to_kgs(FLOW_M3H, T_DISCH_IN)

# ── Startprofil: halbgeladener Speicher ──────────────────────────────────────
# Thermokline zwischen 18 m und 22 m (Speicher ≈ 40 m hoch)
# Unten kalt (60 °C), oben warm (90 °C)
start_profil = {
     2.0: 60.0,
     6.0: 60.0,
    10.0: 60.0,
    14.0: 60.0,
    18.0: 62.0,   # Beginn Übergangszone
    22.0: 85.0,   # Ende Übergangszone
    26.0: 90.0,
    30.0: 90.0,
    34.0: 90.0,
    38.0: 90.0,
}

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("FreeTTES – Benchmark Run")
print(f"  Zeitschritte : {N_HOURS} × {DT_S} s = {N_HOURS} h")
print(f"  Beladung     : t=0..{T_END_CHARGE-1} h    |  {FLOW_M3H} m³/h  |  T_zu={T_CHARGE_IN} °C")
print(f"  Standzeit    : t={T_END_CHARGE}..{T_END_IDLE-1} h   |  kein Durchfluss")
print(f"  Entladung    : t={T_END_IDLE}..{T_END_DISCH-1} h   |  {FLOW_M3H} m³/h  |  T_zu={T_DISCH_IN} °C")
print(f"  Ergebnisse   : {RESULTS_DIR}")
print("=" * 65)

records    = []   # eine Zeile pro Zeitschritt
profiles   = {}   # {t_h: Speicherzustand-dict}
step_times = []   # Berechnungszeit pro Zeitschritt [s]

total_start = time.perf_counter()

for t in range(N_HOURS):
    step_start = time.perf_counter()

    if t < T_END_CHARGE:
        # ── Beladen: warmes Wasser oben einleiten ────────────────────────────
        result = model.main(
            t=t,
            dt=DT_S,
            m_VL=m_charge,
            m_RL=-m_charge,
            T_Zustrom=T_CHARGE_IN,
            T_amb=T_AMB,
            eingabe_volumen=False,
            zustand_uebernehmen=(t == 0),
            zustand=start_profil.copy() if t == 0 else {},
        )
        phase = "beladen"
    elif t < T_END_IDLE:
        # ── Standzeit: kein Durchfluss ────────────────────────────────────────
        result = model.main(
            t=t,
            dt=DT_S,
            m_VL=0,
            m_RL=0,
            T_Zustrom=T_CHARGE_IN,   # irrelevant bei m=0
            T_amb=T_AMB,
            eingabe_volumen=False,
            zustand_uebernehmen=False,
            zustand={},
        )
        phase = "standzeit"
    else:
        # ── Entladen: kaltes Wasser unten einleiten ──────────────────────────
        result = model.main(
            t=t,
            dt=DT_S,
            m_VL=-m_discharge,
            m_RL=m_discharge,
            T_Zustrom=T_DISCH_IN,
            T_amb=T_AMB,
            eingabe_volumen=False,
            zustand_uebernehmen=False,
            zustand={},
        )
        phase = "entladen"

    step_dt = time.perf_counter() - step_start
    step_times.append(step_dt)

    records.append({
        "t_h":            t,
        "phase":          phase,
        "T_Austritt_C":   round(result["T_Austritt"], 4),
        "E_nutz_GJ":      round(result["E_nutz"], 6),
        "E_nutz_MWh":     round(GJ_to_MWh(result["E_nutz"]), 4),
        "E_ges_GJ":       round(result["E_ges"], 4),
        "m_nutz_t":       round(result["m_nutz"], 2),
        "Q_V_ges_W":      round(result["Q_V_ges"], 1),
        "T_Diff_O_C":     round(result["T_Diff_O"], 4),
        "T_Diff_U_C":     round(result["T_Diff_U"], 4),
        "H_WS_m":         round(result["H_WS"], 3),
        "calc_time_s":    round(step_dt, 4),
    })

    if t in PROFILE_SNAPSHOTS:
        profiles[t] = {k: list(v) for k, v in result["speicherzustand"].items()}

    print(
        f"t={t:3d}h [{phase:9s}]  "
        f"T_aus={result['T_Austritt']:6.2f}°C  "
        f"E_nutz={GJ_to_MWh(result['E_nutz']):8.2f} MWh  "
        f"calc={step_dt:.2f}s"
    )

total_time = time.perf_counter() - total_start

print("=" * 65)
print(f"Gesamtdauer         : {total_time:.2f} s")
print(f"Ø pro Zeitschritt   : {total_time / N_HOURS:.3f} s")
print(f"Min / Max           : {min(step_times):.3f} s / {max(step_times):.3f} s")
print("=" * 65)

# ── CSV-Ergebnisse ────────────────────────────────────────────────────────────
csv_path = RESULTS_DIR / "benchmark_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
print(f"CSV gespeichert     : {csv_path}")

# ── JSON-Summary ──────────────────────────────────────────────────────────────
summary = {
    "n_timesteps":           N_HOURS,
    "dt_s":                  DT_S,
    "flow_rate_m3h":         FLOW_M3H,
    "T_charge_in_C":         T_CHARGE_IN,
    "T_discharge_in_C":      T_DISCH_IN,
    "total_time_s":          round(total_time, 3),
    "avg_time_per_step_s":   round(total_time / N_HOURS, 4),
    "min_step_s":            round(min(step_times), 4),
    "max_step_s":            round(max(step_times), 4),
    "E_nutz_start_MWh":           round(GJ_to_MWh(records[0]["E_nutz_GJ"]), 2),
    "E_nutz_after_charge_MWh":    round(GJ_to_MWh(records[T_END_CHARGE - 1]["E_nutz_GJ"]), 2),
    "E_nutz_after_idle_MWh":      round(GJ_to_MWh(records[T_END_IDLE - 1]["E_nutz_GJ"]), 2),
    "E_nutz_end_MWh":             round(GJ_to_MWh(records[-1]["E_nutz_GJ"]), 2),
    "E_nutz_peak_MWh":            round(GJ_to_MWh(max(r["E_nutz_GJ"] for r in records)), 2),
    "E_loss_idle_MWh":            round(
        GJ_to_MWh(records[T_END_CHARGE - 1]["E_nutz_GJ"]
                  - records[T_END_IDLE - 1]["E_nutz_GJ"]), 2),
    "T_Austritt_charge_end_C":    round(records[T_END_CHARGE - 1]["T_Austritt_C"], 2),
    "T_Austritt_discharge_end_C": round(records[-1]["T_Austritt_C"], 2),
}
summary_path = RESULTS_DIR / "benchmark_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Summary gespeichert : {summary_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Zeitreihen (T_Austritt, E_nutz, Q_Verlust)
# ─────────────────────────────────────────────────────────────────────────────
t_axis = [r["t_h"] for r in records]
# T_Austritt getrennt nach Phase:
#   Beladung  → Rücklauf (kalt, unten)
#   Entladung → Vorlauf  (heiß, oben)
T_aus_charge_t = [r["t_h"]        for r in records if r["phase"] == "beladen"]
T_aus_charge   = [r["T_Austritt_C"] for r in records if r["phase"] == "beladen"]
T_aus_disch_t  = [r["t_h"]        for r in records if r["phase"] == "entladen"]
T_aus_disch    = [r["T_Austritt_C"] for r in records if r["phase"] == "entladen"]
E_ntz  = [r["E_nutz_MWh"] for r in records]
Q_v_kW = [r["Q_V_ges_W"] / 1000.0 for r in records]

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
fig.suptitle(
    f"FreeTTES Benchmark – {FLOW_M3H} m³/h | "
    f"Beladung 0–{T_END_CHARGE-1} h | Standzeit {T_END_CHARGE}–{T_END_IDLE-1} h | "
    f"Entladung {T_END_IDLE}–{T_END_DISCH-1} h",
    fontsize=11, y=0.98,
)

# Hintergrund-Shading pro Phase
for ax in axes:
    ax.axvspan(0,             T_END_CHARGE - 0.5, alpha=0.07, color="steelblue", label="_nolegend_")
    ax.axvspan(T_END_CHARGE - 0.5, T_END_IDLE - 0.5, alpha=0.07, color="gold",      label="_nolegend_")
    ax.axvspan(T_END_IDLE - 0.5,   T_END_DISCH,       alpha=0.07, color="tomato",    label="_nolegend_")
    ax.axvline(T_END_CHARGE, color="gray", linestyle="--", linewidth=1)
    ax.axvline(T_END_IDLE,   color="gray", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.35)

axes[0].plot(T_aus_charge_t, T_aus_charge, color="steelblue", linewidth=2,
             marker="o", markersize=3, label=f"Rücklauf (Beladung, unten)")
axes[0].plot(T_aus_disch_t,  T_aus_disch,  color="tomato",    linewidth=2,
             marker="o", markersize=3, label=f"Vorlauf (Entladung, oben)")
axes[0].set_ylabel("T Austritt [°C]")
axes[0].set_title("Austrittstemperatur  (Beladung → Rücklauf kalt | Entladung → Vorlauf heiß)", fontsize=10)

axes[1].plot(t_axis, E_ntz, color="steelblue", linewidth=2, marker="o", markersize=3)
axes[1].set_ylabel("E nutz [MWh]")
axes[1].set_title("Nutzbare Energie", fontsize=10)

axes[2].plot(t_axis, Q_v_kW, color="darkorange", linewidth=2, marker="o", markersize=3)
axes[2].set_ylabel("Q Verlust [kW]")
axes[2].set_xlabel("Zeit [h]")
axes[2].set_title("Wärmeverluste gesamt", fontsize=10)

# Legende: Phasenshading + Kurven
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elems = [
    Patch(facecolor="steelblue", alpha=0.3,  label="Phase: Beladung"),
    Patch(facecolor="gold",      alpha=0.3,  label="Phase: Standzeit"),
    Patch(facecolor="tomato",    alpha=0.3,  label="Phase: Entladung"),
    Line2D([0], [0], color="steelblue", linewidth=2, marker="o", markersize=4,
           label="Rücklauf T (Beladung, unten)"),
    Line2D([0], [0], color="tomato",    linewidth=2, marker="o", markersize=4,
           label="Vorlauf T (Entladung, oben)"),
]
axes[0].legend(handles=legend_elems, loc="center right", fontsize=8.5)

plt.tight_layout()
ts_path = RESULTS_DIR / "timeseries.png"
plt.savefig(ts_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot gespeichert    : {ts_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Temperaturprofile (Snapshots)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 10))

n_snap  = len(profiles)
n_bl = sum(1 for t in profiles if t < T_END_CHARGE)
n_id = sum(1 for t in profiles if T_END_CHARGE <= t < T_END_IDLE)
n_rd = sum(1 for t in profiles if t >= T_END_IDLE)
cmap_bl = plt.cm.Blues(np.linspace(0.35, 0.95, max(n_bl, 1)))
cmap_id = plt.cm.YlOrBr(np.linspace(0.35, 0.70, max(n_id, 1)))
cmap_rd = plt.cm.Reds( np.linspace(0.35, 0.95, max(n_rd, 1)))
colors  = list(cmap_bl[:n_bl]) + list(cmap_id[:n_id]) + list(cmap_rd[:n_rd])

for color, (t_snap, sz) in zip(colors, sorted(profiles.items())):
    h_vals = np.array(sorted(sz.keys()), dtype=float)
    T_vals = np.array([sz[h][0] for h in h_vals], dtype=float)

    f_ip   = interp1d(h_vals, T_vals, kind="linear",
                      bounds_error=False, fill_value="extrapolate")
    h_fine = np.linspace(h_vals[0], h_vals[-1], 600)
    if t_snap < T_END_CHARGE:
        phase_label = "Bel."
    elif t_snap < T_END_IDLE:
        phase_label = "Stand."
    else:
        phase_label = "Entl."
    ax.plot(f_ip(h_fine), h_fine, color=color, linewidth=1.8,
            label=f"t={t_snap:2d}h ({phase_label})")

ax.set_xlabel("Temperatur [°C]", fontsize=11)
ax.set_ylabel("Höhe [m]",         fontsize=11)
ax.set_title("Temperaturprofile – Snapshots", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.35)
plt.tight_layout()
prof_path = RESULTS_DIR / "temperature_profiles.png"
plt.savefig(prof_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot gespeichert    : {prof_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Berechnungszeit pro Zeitschritt
# ─────────────────────────────────────────────────────────────────────────────
bar_colors = [
    "steelblue" if r["phase"] == "beladen"
    else ("gold" if r["phase"] == "standzeit" else "tomato")
    for r in records
]

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(t_axis, step_times, color=bar_colors, width=0.8, alpha=0.85)
ax.axhline(
    np.mean(step_times), color="black", linestyle="--", linewidth=1.5,
    label=f"Ø {np.mean(step_times):.3f} s"
)
ax.set_xlabel("Zeit [h]", fontsize=11)
ax.set_ylabel("Berechnungszeit [s]", fontsize=11)
ax.set_title("Berechnungszeit pro Zeitschritt (blau=Beladen, gelb=Standzeit, rot=Entladen)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

# Gesamtzeit als Annotation
ax.text(
    0.98, 0.95,
    f"Gesamt: {total_time:.1f} s  |  Ø: {total_time/N_HOURS:.3f} s/Schritt",
    transform=ax.transAxes, ha="right", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)
plt.tight_layout()
timing_path = RESULTS_DIR / "timing_per_step.png"
plt.savefig(timing_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot gespeichert    : {timing_path}")

# ─────────────────────────────────────────────────────────────────────────────
print()
print("Benchmark abgeschlossen.")
print(f"  Gesamtdauer       : {total_time:.2f} s")
print(f"  Ø pro Zeitschritt : {np.mean(step_times):.3f} s")
print(f"  Min / Max         : {min(step_times):.3f} s / {max(step_times):.3f} s")
print(f"  E_nutz (Start)    : {GJ_to_MWh(records[0]['E_nutz_GJ']):.1f} MWh")
print(f"  E_nutz (Peak)     : {GJ_to_MWh(max(r['E_nutz_GJ'] for r in records)):.1f} MWh")
print(f"  E_nutz (Ende)     : {GJ_to_MWh(records[-1]['E_nutz_GJ']):.1f} MWh")
