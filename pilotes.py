"""Les pilotes : ils proposent une vitesse, sans se soucier de la sécurité.

Chaque pilote est une fonction pilote(piliers, p, buts) -> vitesses (N, 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from garde_fou import V_MAX, observation

POLITIQUE = Path(__file__).resolve().parent / "docs" / "politique.js"


def borner(u: np.ndarray) -> np.ndarray:
    """Ramène chaque vitesse à V_MAX au plus, sans changer sa direction."""
    norme = np.linalg.norm(u, axis=-1, keepdims=True)
    return u * np.minimum(1.0, V_MAX / np.maximum(norme, 1e-12))


# ------------------------------------------------------ réseau de neurones ---

def reseau(poids: dict, obs: np.ndarray) -> np.ndarray:
    """Perceptron à une couche cachée : 27 entrées → 32 neurones tanh → 2 vitesses.

    Les poids peuvent porter une dimension de lot (un jeu de poids par robot),
    ce qui sert à l'entraînement.
    """
    cachee = np.tanh(np.einsum("...i,...ij->...j", obs, poids["W1"]) + poids["b1"])
    sortie = np.tanh(np.einsum("...i,...ij->...j", cachee, poids["W2"]) + poids["b2"])
    return borner(V_MAX * sortie)


def charger_poids(chemin: Path = POLITIQUE) -> dict:
    """Lit les poids dans docs/politique.js, le fichier que charge aussi la démo web."""
    texte = chemin.read_text(encoding="utf-8")
    donnees = json.loads(texte[texte.index("{"): texte.rindex("}") + 1])
    return {cle: np.array(donnees[cle]) for cle in ("W1", "b1", "W2", "b2")}


def pilote_reseau(poids: dict | None = None):
    poids = charger_poids() if poids is None else poids
    return lambda piliers, p, buts: reseau(poids, observation(piliers, p, buts))


# ------------------------------------------------------- pilotes simples ---

def glouton(piliers, p, buts):
    """Fonce tout droit vers le but et ralentit dans les 50 derniers centimètres."""
    vers_but = buts - p
    distance = np.linalg.norm(vers_but, axis=1, keepdims=True)
    return V_MAX * vers_but / np.maximum(distance, 1e-9) * np.minimum(1.0, distance / 0.5)


def kamikaze(piliers, p, buts):
    """Hostile : vise à pleine vitesse le centre du pilier le plus proche."""
    ecart = piliers[..., :2] - p[:, None, :]
    distances = np.linalg.norm(ecart, axis=-1) - piliers[..., 2]
    cible = ecart[np.arange(len(p)), distances.argmin(axis=1)]
    return V_MAX * cible / np.linalg.norm(cible, axis=1, keepdims=True)


def aleatoire(graine: int, tenue: int = 10):
    """Erratique : une vitesse au hasard, conservée `tenue` pas de suite."""
    rng = np.random.default_rng(graine)
    etat = {"pas": 0, "u": None}

    def pilote(piliers, p, buts):
        if etat["pas"] % tenue == 0:
            angle = rng.uniform(0, 2 * np.pi, len(p))
            vitesse = V_MAX * np.sqrt(rng.uniform(0, 1, len(p)))
            etat["u"] = vitesse[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        etat["pas"] += 1
        return etat["u"]

    return pilote
