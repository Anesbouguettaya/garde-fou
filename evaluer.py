#!/usr/bin/env python3
"""Banc d'essai : chaque pilote, avec et sans garde-fou, sur 1 000 scènes jamais vues.

    python evaluer.py      # une à deux minutes

Affiche le tableau des résultats et écrit docs/resultats.js, que la démo web
affiche. Code de sortie 1 si le garde-fou laisse passer une seule collision.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from garde_fou import DISTANCE_SECURITE, DT, scenes, simuler
from pilotes import glouton, kamikaze, pilote_reseau

N_SCENES = 1000
GRAINE = 2026          # différente des graines d'entraînement (123) et de validation (10 000)
PAS_MAX = 600          # 30 s par scène

PILOTES = [
    ("reseau", "Réseau de neurones", "lidar 24 faisceaux, appris par imitation", pilote_reseau()),
    ("naif", "Pilote naïf", "tout droit vers le but", glouton),
    ("hostile", "Pilote hostile", "vise le pilier le plus proche", kamikaze),
]


def bilan(r: dict) -> dict:
    atteint = r["atteint"]
    return {
        "atteint": float(atteint.mean()),
        "collision": float(r["collision"].mean()),
        "corriges": float(r["corriges"].sum() / r["pas"].sum()),
        "duree_s": float(r["duree"][atteint].mean() * DT) if atteint.any() else None,
        "marge_min_cm": float(r["marge_min"].min() * 100),
        "pas": int(r["pas"].sum()),
    }


def main() -> int:
    piliers, departs, buts = scenes(N_SCENES, GRAINE)
    lignes = []
    for cle, nom, description, pilote in PILOTES:
        sans = bilan(simuler(piliers, departs, buts, pilote, garde_fou=False, pas_max=PAS_MAX))
        avec = bilan(simuler(piliers, departs, buts, pilote, garde_fou=True, pas_max=PAS_MAX))
        lignes.append({"cle": cle, "nom": nom, "description": description, "sans": sans, "avec": avec})

    def pc(x):
        return f"{x:6.1%}"

    print(f"Banc d'essai : {N_SCENES} scènes, {PAS_MAX * DT:.0f} s au plus par scène\n")
    print(f"{'':20} {'sans garde-fou':>21}   {'avec garde-fou':>36}")
    print(f"{'pilote':20} {'but':>10} {'collision':>10}   {'but':>10} {'collision':>10} {'corrigées':>14}")
    for l in lignes:
        print(f"{l['nom']:20} {pc(l['sans']['atteint']):>10} {pc(l['sans']['collision']):>10}   "
              f"{pc(l['avec']['atteint']):>10} {pc(l['avec']['collision']):>10} {pc(l['avec']['corriges']):>14}")
    marge_min = min(l["avec"]["marge_min_cm"] for l in lignes)
    pas = sum(l["avec"]["pas"] for l in lignes)
    collisions = sum(l["avec"]["collision"] for l in lignes)
    milliers = f"{pas:,}".replace(",", " ")
    print(f"\nAvec garde-fou : {milliers} pas simulés, marge minimale {marge_min:.2f} cm "
          f"(distance de sécurité : {DISTANCE_SECURITE * 100:.0f} cm)")

    resultats = {"date": date.today().isoformat(), "scenes": N_SCENES, "duree_max_s": PAS_MAX * DT,
                 "distance_securite_cm": DISTANCE_SECURITE * 100, "marge_min_cm": round(marge_min, 4),
                 "pas_avec_garde_fou": pas, "pilotes": lignes}
    sortie = Path(__file__).resolve().parent / "docs" / "resultats.js"
    sortie.write_text("// Résultats du banc d'essai, écrits par evaluer.py. Ne pas modifier à la main.\n"
                      f"globalThis.RESULTATS = {json.dumps(resultats, ensure_ascii=False, indent=1)};\n", encoding="utf-8")
    print("Résultats écrits dans docs/resultats.js")
    if collisions > 0:
        print("ÉCHEC : le garde-fou a laissé passer une collision.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
