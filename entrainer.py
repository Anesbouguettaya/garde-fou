#!/usr/bin/env python3
"""Entraîne le pilote neuronal par imitation (DAgger), sans garde-fou.

    python entrainer.py            # quelques minutes sur un portable

Le professeur connaît la carte. Il calcule, sur une grille de 5 cm, le coût du
chemin le plus court vers le but (algorithme de Dijkstra), en pénalisant la
proximité des piliers comme la couche d'inflation de ROS Navigation, puis il
descend ce coût. Il atteint toujours son but sans rien toucher.

L'élève, le réseau de neurones, ne voit que son lidar et la direction du but.
Il apprend à reproduire la vitesse du professeur dans chaque situation. Avec
DAgger (Ross et al., 2011), c'est l'élève qui conduit : on demande au
professeur ce qu'il aurait fait dans les situations où l'élève s'est mis, on
ajoute ces exemples et on réentraîne. L'élève apprend ainsi à rattraper ses
propres erreurs.

Le réseau apprend bien, mais imiter un professeur sûr ne rend pas sûr : c'est le
rôle du garde-fou.
"""

from __future__ import annotations

import json
import time
from datetime import date

import numpy as np

from garde_fou import (DISTANCE_SECURITE, DT, HAUTEUR, LARGEUR, RAYON_BUT, V_MAX,
                       marge, observation, scenes, simuler)
from pilotes import POLITIQUE, pilote_reseau, reseau

ENTREES, CACHEES, SORTIES = 27, 32, 2
RESOLUTION = 0.05                      # m : pas de la grille du professeur
ITERATIONS = 10                        # tours de DAgger
SCENES_ENTRAINEMENT = 400
PAS_MAX = 400                          # 20 s par épisode

_xs = np.arange(round(LARGEUR / RESOLUTION) + 1) * RESOLUTION
_ys = np.arange(round(HAUTEUR / RESOLUTION) + 1) * RESOLUTION
_VOISINS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


# --------------------------------------------------------------- professeur ---

def carte_de_cout(piliers: np.ndarray, but: np.ndarray) -> np.ndarray:
    """Coût du plus court chemin vers le but depuis chaque point de la grille (Dijkstra par relaxations)."""
    gx, gy = np.meshgrid(_xs, _ys)
    points = np.stack([gx.ravel(), gy.ravel()], axis=1)
    m = marge(np.broadcast_to(piliers, (len(points),) + piliers.shape), points).reshape(gx.shape)
    libre = m >= DISTANCE_SECURITE
    poids = 1 + 6 * np.exp(-np.maximum(m, 0) / 0.12)          # inflation : passer près d'un pilier coûte cher
    cout = np.full(gx.shape, np.inf)
    cout[round(but[1] / RESOLUTION), round(but[0] / RESOLUTION)] = 0.0
    while True:
        avant = cout.copy()
        for dx, dy in _VOISINS:
            decale = np.full(gx.shape, np.inf)
            poids_voisin = np.ones(gx.shape)
            cible = (slice(max(dy, 0), gx.shape[0] + min(dy, 0)), slice(max(dx, 0), gx.shape[1] + min(dx, 0)))
            source = (slice(max(-dy, 0), gx.shape[0] + min(-dy, 0)), slice(max(-dx, 0), gx.shape[1] + min(-dx, 0)))
            decale[cible] = cout[source]
            poids_voisin[cible] = poids[source]
            pas = RESOLUTION * np.hypot(dx, dy) * (poids + poids_voisin) / 2
            cout = np.where(libre, np.minimum(cout, decale + pas), np.inf)
        if np.array_equal(avant, cout):
            break
    plafond = cout[np.isfinite(cout)].max() + 10
    return np.where(np.isfinite(cout), cout, plafond)          # obstacles : coût élevé mais fini


def _interpoler(cartes: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Valeur bilinéaire de chaque carte (N, NY, NX) au point p (N, 2)."""
    x = np.clip(p[:, 0] / RESOLUTION, 0, len(_xs) - 1.000001)
    y = np.clip(p[:, 1] / RESOLUTION, 0, len(_ys) - 1.000001)
    i, j = x.astype(int), y.astype(int)
    fx, fy = x - i, y - j
    n = np.arange(len(p))
    return ((1 - fx) * (1 - fy) * cartes[n, j, i] + fx * (1 - fy) * cartes[n, j, i + 1]
            + (1 - fx) * fy * cartes[n, j + 1, i] + fx * fy * cartes[n, j + 1, i + 1])


def professeur(cartes: np.ndarray, p: np.ndarray, buts: np.ndarray) -> np.ndarray:
    """Descend la carte de coût ; en approche finale, file droit vers le but."""
    e = 0.03
    gradient = np.stack([_interpoler(cartes, p + [e, 0]) - _interpoler(cartes, p - [e, 0]),
                         _interpoler(cartes, p + [0, e]) - _interpoler(cartes, p - [0, e])], axis=1)
    direction = -gradient / np.maximum(np.linalg.norm(gradient, axis=1, keepdims=True), 1e-9)
    vers_but = buts - p
    distance = np.linalg.norm(vers_but, axis=1, keepdims=True)
    direction = np.where(distance < 0.3, vers_but / np.maximum(distance, 1e-9), direction)
    return V_MAX * np.minimum(1.0, distance / 0.5) * direction


# -------------------------------------------------------------------- élève ---

def initialiser(rng) -> dict:
    return {"W1": rng.normal(0, 1 / np.sqrt(ENTREES), (ENTREES, CACHEES)), "b1": np.zeros(CACHEES),
            "W2": rng.normal(0, 1 / np.sqrt(CACHEES), (CACHEES, SORTIES)), "b2": np.zeros(SORTIES)}


def ajuster(poids: dict, X: np.ndarray, Y: np.ndarray, rng, epoques: int = 6, lot: int = 256,
            pas_apprentissage: float = 2e-3) -> dict:
    """Régression des vitesses du professeur : erreur quadratique, rétropropagation et Adam, écrits à la main."""
    poids = {cle: valeur.copy() for cle, valeur in poids.items()}
    moments = {cle: np.zeros_like(valeur) for cle, valeur in poids.items()}
    carres = {cle: np.zeros_like(valeur) for cle, valeur in poids.items()}
    t = 0
    for _ in range(epoques):
        ordre = rng.permutation(len(X))
        for debut in range(0, len(X), lot):
            k = ordre[debut:debut + lot]
            x, y = X[k], Y[k]
            cachee = np.tanh(x @ poids["W1"] + poids["b1"])
            sortie = np.tanh(cachee @ poids["W2"] + poids["b2"])
            d_z2 = 2 * (V_MAX * sortie - y) * V_MAX * (1 - sortie ** 2) / len(k)
            d_z1 = (d_z2 @ poids["W2"].T) * (1 - cachee ** 2)
            gradients = {"W2": cachee.T @ d_z2, "b2": d_z2.sum(axis=0), "W1": x.T @ d_z1, "b1": d_z1.sum(axis=0)}
            t += 1
            for cle, g in gradients.items():
                moments[cle] = 0.9 * moments[cle] + 0.1 * g
                carres[cle] = 0.999 * carres[cle] + 0.001 * g ** 2
                poids[cle] -= pas_apprentissage * (moments[cle] / (1 - 0.9 ** t)) / (
                    np.sqrt(carres[cle] / (1 - 0.999 ** t)) + 1e-8)
    return poids


def collecter(piliers, departs, buts, cartes, poids: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Fait rouler l'élève (ou le professeur si poids est None), sans garde-fou, et note
    pour chaque situation rencontrée la vitesse que le professeur aurait choisie."""
    p = departs.copy()
    actifs = np.arange(len(p))
    X, Y = [], []
    for _ in range(PAS_MAX):
        obs = observation(piliers[actifs], p[actifs], buts[actifs])
        conseil = professeur(cartes[actifs], p[actifs], buts[actifs])
        X.append(obs)
        Y.append(conseil)
        u = conseil if poids is None else reseau(poids, obs)
        p[actifs] += DT * u
        fini = (marge(piliers[actifs], p[actifs]) < 0) | (np.linalg.norm(buts[actifs] - p[actifs], axis=1) < RAYON_BUT)
        actifs = actifs[~fini]
        if len(actifs) == 0:
            break
    return np.concatenate(X), np.concatenate(Y)


def entrainer(graine: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(graine)
    debut = time.perf_counter()
    piliers, departs, buts = scenes(SCENES_ENTRAINEMENT, graine=123)
    cartes = np.stack([carte_de_cout(piliers[s], buts[s]) for s in range(SCENES_ENTRAINEMENT)])
    validation = scenes(200, graine=10_000)
    cartes_validation = np.stack([carte_de_cout(validation[0][s], validation[2][s]) for s in range(200)])
    r = simuler(*validation, lambda _, p, b: professeur(cartes_validation, p, b), garde_fou=False)
    print(f"professeur : but atteint {r['atteint'].mean():.1%}, collision {r['collision'].mean():.1%}, "
          f"marge minimale {r['marge_min'].min() * 100:.1f} cm · {time.perf_counter() - debut:.0f} s", flush=True)

    poids, X_tous, Y_tous = initialiser(rng), [], []
    meilleur, meilleur_bilan = poids, {"note": -np.inf}
    for iteration in range(ITERATIONS):
        X, Y = collecter(piliers, departs, buts, cartes, None if iteration == 0 else poids)
        X_tous.append(X)
        Y_tous.append(Y)
        poids = ajuster(poids, np.concatenate(X_tous), np.concatenate(Y_tous), rng)
        r = simuler(*validation, pilote_reseau(poids), garde_fou=False)
        bilan = {"iteration": iteration, "exemples": int(sum(len(x) for x in X_tous)),
                 "atteint": float(r["atteint"].mean()), "collision": float(r["collision"].mean())}
        bilan["note"] = bilan["atteint"] - bilan["collision"]
        if bilan["note"] > meilleur_bilan["note"]:
            meilleur, meilleur_bilan = poids, bilan
        print(f"tour {iteration:2d} · {bilan['exemples']:7d} exemples · but atteint {bilan['atteint']:6.1%} · "
              f"collision {bilan['collision']:6.1%} · {time.perf_counter() - debut:4.0f} s", flush=True)
    return meilleur, meilleur_bilan


def enregistrer(poids: dict, bilan: dict) -> None:
    """Écrit les poids dans docs/politique.js, lu à la fois par Python et par la démo web."""
    donnees = {
        "description": "Pilote neuronal de Garde-fou : 27 entrées, 32 neurones tanh, 2 sorties. "
                       "Appris par imitation (DAgger) d'un planificateur qui connaît la carte, sans garde-fou.",
        "entraine_le": date.today().isoformat(),
        "tour_dagger": bilan["iteration"],
        "exemples": bilan["exemples"],
        "validation_sans_garde_fou": {"atteint": bilan["atteint"], "collision": bilan["collision"]},
        **{cle: valeur.tolist() for cle, valeur in poids.items()},
    }
    POLITIQUE.write_text("// Poids du pilote neuronal, écrits par entrainer.py. Ne pas modifier à la main.\n"
                         f"globalThis.POLITIQUE = {json.dumps(donnees, ensure_ascii=False)};\n", encoding="utf-8")


if __name__ == "__main__":
    poids, bilan = entrainer()
    enregistrer(poids, bilan)
    print(f"Meilleur réseau : tour {bilan['iteration']}, but atteint {bilan['atteint']:.1%}, "
          f"collision {bilan['collision']:.1%} (200 scènes de validation, sans garde-fou)")
    print(f"Poids enregistrés dans {POLITIQUE.relative_to(POLITIQUE.parents[1])}")
