# Optimization B: `__Modell_Waermeleitung` – NumPy-Vektorisierung

## Problem

`__Modell_Waermeleitung` wird **60×/Zeitschritt** aufgerufen und ist mit ~14 %
der Gesamtlaufzeit der drittgrößte Hotspot.

Die Funktion:
1. Baut 7 Python-Listen mit `[None for h in all_h_pos]` auf (~1 300 Elemente)
2. Füllt sie in **drei separaten Python-`for`-Schleifen** zellenweise
3. Ruft dabei `_sw_rho`, `_sw_cp`, `_sw_lambda` einzeln pro Zelle auf
   → bei 1 300 Zellen × 60 Sub-Schritten = **78 000 Python-Funktionsaufrufe**
   allein für diese Funktion
4. Übergibt die fertigen Listen an `__Modell_TDMASolve` (weiterer Python-Loop)

**Alle `_sw_*`-Funktionen sind einfache Polynome** – sie sind trivial vektorisierbar:

```python
def _sw_rho(T):    return -2.525726E-03*T**2 - 2.123038E-01*T + 1.005011E+03
def _sw_lambda(T): return 3.097195E-08*T**3 - 1.565775E-05*T**2 + ...
```

## Lösung

1. `thetaWL`, `dx` als NumPy-Arrays anlegen statt Python-Listen
2. `_sw_rho`, `_sw_cp`, `_sw_lambda` vektorisiert auf das gesamte Array aufrufen
3. `tlf_mod`, `lambda_`, `a_WL`, `b_WL`, `c_WL`, `d_WL` als NumPy-Arrays berechnen
4. `__Modell_TDMASolve` durch NumPy-TDMA (vektorisierte Forward/Backward-Substitution) ersetzen

### Beispiel – Vektorisierung `tlf_mod`:
```python
# Vorher (Python-Loop):
rho = np.where(all_h_pos > 0, rho_arr[j], _RHO_FUNDAMENT)
tlf_mod = dt / (rho * cp * 2 * dx)

# Nachher (NumPy):
rho_arr = np.where(h_arr > 0,
                   _sw_rho_vec(theta_arr),
                   _RHO_FUNDAMENT)
cp_arr  = np.where(h_arr[1:] > 0, ...)
tlf_mod = dt_sub / (rho_arr * cp_arr * 2 * dx_arr)
```

### Vektorisierte Polynomial-Hilfsfunktionen:
Neue `_sw_rho_v`, `_sw_cp_v`, `_sw_lambda_v` die NumPy-Arrays akzeptieren.

## Erwarteter Gewinn
~8–12 % Gesamtlaufzeit (NumPy-Vektorisierung typisch 10–50× schneller als Python-Loops).

## Risiko: mittel
Logik ist komplex (Fundament vs. Speicher-Index Unterscheidung, Randbedingungen).
Tests gegen Baseline-Ergebnisse sind zwingend.
