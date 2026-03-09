"""I/O layer for the TES tank model.

All filesystem read/write is placed here to keep the model/physics module pure.
The functions keep the legacy file formats and names so existing postprocessing
continues to work.
"""

from __future__ import annotations

import csv
import os
import logging
from scipy import interpolate

import numpy as np

import FreeTTES_config as cfg

logger = logging.getLogger(__name__)


def ausgabe_zeitschritt(
    last: int,
    Ausgabezeit: int,
    alle_Temperaturprofile: dict,
    Fundamentzustand: dict,
    Speicherzustand: dict,
    Kapazitaeten: dict,
) -> dict:
    """Legacy output writer.

    - When ``last == 1`` (final sub-step only): interpolates the stratified
      profile onto a 5 cm grid, stores it in
      ``alle_Temperaturprofile[Ausgabezeit]``, and writes:
        * temp_profil_<t>.dat
        * last_profile_*.csv snapshots (state persistence for the next call)
        * sz/sz<t>.dat
    - When ``last == 0`` (intermediate sub-steps): returns immediately without
      doing any work.

    **Why the early-return is safe for intermediate sub-steps** (last == 0):

    1. ``alle_Temperaturprofile`` is a *local* variable inside
       ``FreeTTES_model.main()`` (initialised at line 154 of
       ``FreeTTES_model.py`` as ``{}``).  It is updated only inside the
       sub-step loop (lines 476-479) and is **never** included in the
       ``outputs`` dict that is returned to the caller (lines 536-576 list
       every key; ``alle_Temperaturprofile`` is absent).  The caller therefore
       never sees the intermediate profiles, making them computationally
       worthless.

    2. The CubicSpline construction (``scipy.interpolate.CubicSpline`` over
       ~800 nodes) and the subsequent 5 cm-grid evaluation loop (~820
       iterations) are the two most expensive operations in the function.
       Skipping them on the 59 out of 60 sub-steps where ``last == 0``
       eliminates ~98 % of the spline work per main time-step.

    3. The only side-effects that matter – the filesystem writes
       (``temp_profil_*.dat``, ``last_profile_*.csv``, ``sz*.dat``) – are
       already guarded by ``if last == 1:``.  Returning early does not affect
       any output file.

    4. ``letzter_zustand()`` (the state-reload function called at the start of
       every subsequent ``main()`` invocation) reads exclusively from the three
       ``last_profile_*.csv`` snapshot files.  It never touches
       ``alle_Temperaturprofile``.  Physics continuity is therefore fully
       preserved.

    Numerical behaviour for ``last == 1`` is completely unchanged.
    """

    # Optimization: nothing useful happens on intermediate sub-steps.
    # See docstring for the full evidence chain.
    if last != 1:
        return alle_Temperaturprofile

    # Use the globally loaded storage parameters (legacy behavior)
    speicher_param = cfg.speicher_param

    all_h_pos = sorted(list(Speicherzustand))
    all_theta = [v[0] for k, v in sorted(Speicherzustand.items())]

    theta_spline = interpolate.CubicSpline(all_h_pos, all_theta, bc_type="natural")

    alle_Temperaturprofile[Ausgabezeit] = {}
    temperaturDatei = cfg.abs_folder_temperaturProfile + "temp_profil_" + str(Ausgabezeit) + ".dat"

    hPos = 0.0
    while hPos < speicher_param["H_Mantel"]:
        if hPos < all_h_pos[0]:
            grad = (all_theta[1] - all_theta[0]) / (all_h_pos[1] - all_h_pos[0])
            alle_Temperaturprofile[Ausgabezeit][hPos] = all_theta[0] + grad * (hPos - all_h_pos[0])
        elif hPos > all_h_pos[-1]:
            alle_Temperaturprofile[Ausgabezeit][hPos] = all_theta[-1]
        else:
            alle_Temperaturprofile[Ausgabezeit][hPos] = float(theta_spline(hPos))
        hPos = round(hPos + 0.05, 10)

    with open(temperaturDatei, "w") as f:
        for hPos in sorted(list(alle_Temperaturprofile[Ausgabezeit])):
            f.write("%.2f;%.5f;\n" % (hPos, alle_Temperaturprofile[Ausgabezeit][hPos]))

    # Keep legacy Windows-style paths (original code used script_dir + "\\datei\\..." )
    script_dir = cfg.SCRIPT_DIR

    with open(script_dir + "\\datei\\last_profile_kap.csv", mode="w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        for k, v in Kapazitaeten.items():
            w.writerow((k, v))

    with open(script_dir + "\\datei\\last_profile_fund.csv", mode="w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        for k, v in Fundamentzustand.items():
            w.writerow((k, v))

    with open(script_dir + "\\datei\\last_profile_zus.csv", mode="w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        for k, v in Speicherzustand.items():
            w.writerow((k, v))

    with open(script_dir + "\\datei\\sz\\sz" + str(Ausgabezeit) + ".dat", mode="w", newline="") as f:
        for k, v in Speicherzustand.items():
            f.write("%f;%f;\n" % (k, v[0]))

    return alle_Temperaturprofile


def letzter_zustand():
    """Load last stored model state from legacy CSV snapshot files."""

    Speicherzustand: dict = {}
    Fundamentzustand: dict = {}
    Kapazitaeten: dict = {}

    script_dir = cfg.SCRIPT_DIR

    with open(script_dir + "\\datei\\last_profile_fund.csv", mode="r") as f:
        reader = csv.reader(f)
        dc_ = dict(reader)
        for k, v in dc_.items():
            Fundamentzustand[float(k)] = [float(x) for x in (v[1:-1].split(","))]

    with open(script_dir + "\\datei\\last_profile_kap.csv", mode="r") as f:
        reader = csv.reader(f)
        dc_ = dict(reader)
        for k, v in dc_.items():
            Kapazitaeten[float(k)] = [float(x) for x in (v[1:-1].split(","))]

    with open(script_dir + "\\datei\\last_profile_zus.csv", mode="r") as f:
        reader = csv.reader(f)
        dc_ = dict(reader)
        for k, v in dc_.items():
            Speicherzustand[float(k)] = [float(x) for x in (v[1:-1].split(","))]

    return Speicherzustand, Fundamentzustand, Kapazitaeten
