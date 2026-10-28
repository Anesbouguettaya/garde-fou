"""Garde-fou : un filtre de sécurité qui empêche toute IA de faire percuter un obstacle au robot.

Le monde : une arène rectangulaire, des piliers circulaires et un robot mobile
omnidirectionnel (roues mecanum), commandé en vitesse à 20 Hz. Un pilote, réseau
de neurones ou humain, propose une vitesse ; le garde-fou la remplace par la
vitesse sûre la plus proche.

La garantie (voir README) : pour chaque obstacle, on note h(p) ≥ 0 une mesure de
la place libre. On impose h(p + Δt·u) ≥ (1 − γ)·h(p) à chaque pas : h ne peut
donc jamais devenir négatif, et le robot reste toujours à 2 cm au moins de tout
obstacle.

Tout est vectorisé avec numpy : chaque fonction traite N robots à la fois,
chacun dans sa propre scène. Notations : p position (m), u vitesse (m/s).
"""

from __future__ import annotations

import numpy as np

LARGEUR, HAUTEUR = 8.0, 5.0     # arène, en mètres
RAYON_ROBOT = 0.2               # m
V_MAX = 1.0                     # m/s
DT = 0.05                       # s : boucle de commande à 20 Hz
GAMMA = 0.2                     # part de la marge que le robot peut consommer à chaque pas
DISTANCE_SECURITE = 0.02        # m : le robot reste toujours à 2 cm au moins de tout obstacle
TOLERANCE = 1e-9                # tolérance du test « cette commande est-elle sûre ? »
N_PILIERS = 8
N_RAYONS, PORTEE = 24, 3.0      # lidar : 24 faisceaux, 3 m de portée
RAYON_BUT = 0.25                # m : objectif atteint

_angles = 2 * np.pi * np.arange(N_RAYONS) / N_RAYONS
DIRECTIONS = np.stack([np.cos(_angles), np.sin(_angles)], axis=1)          # (R, 2)
_NORMALES_MURS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])  # gauche, droite, bas, haut


# ------------------------------------------------------------------ scènes ---

def scenes(n: int, graine: int):
    """Tire n scènes : 8 piliers, un départ et un but par scène.

    Entre deux piliers, et entre un pilier et un mur, il reste toujours au moins
    50 cm : le robot (40 cm de diamètre) peut passer partout, chaque but est atteignable.
    Renvoie piliers (n, 8, 3) en [x, y, rayon], departs (n, 2) et buts (n, 2).
    """
    rng = np.random.default_rng(graine)
    passage = 2 * RAYON_ROBOT + 0.1
    piliers = np.empty((n, N_PILIERS, 3))
    departs, buts = np.empty((n, 2)), np.empty((n, 2))
    for s in range(n):
        while True:
            pris = []
            for _ in range(500):
                r = rng.uniform(0.2, 0.5)
                x = rng.uniform(r + passage, LARGEUR - r - passage)
                y = rng.uniform(r + passage, HAUTEUR - r - passage)
                if all(np.hypot(x - a, y - b) >= r + c + passage for a, b, c in pris):
                    pris.append((x, y, r))
                    if len(pris) == N_PILIERS:
                        break
            if len(pris) == N_PILIERS:
                break
        piliers[s] = pris

        def point_libre():
            while True:
                q = rng.uniform([RAYON_ROBOT + 0.1] * 2, [LARGEUR - RAYON_ROBOT - 0.1, HAUTEUR - RAYON_ROBOT - 0.1])
                if all(np.hypot(*(q - o[:2])) >= o[2] + RAYON_ROBOT + 0.1 for o in piliers[s]):
                    return q

        departs[s] = point_libre()
        while True:
            buts[s] = point_libre()
            if np.hypot(*(buts[s] - departs[s])) >= 3.0:
                break
    return piliers, departs, buts


def marge(piliers: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distance entre le bord du robot et l'obstacle le plus proche, en mètres. Négative = contact."""
    d_piliers = np.linalg.norm(p[:, None, :] - piliers[..., :2], axis=-1) - piliers[..., 2] - RAYON_ROBOT
    x, y = p[:, 0], p[:, 1]
    d_murs = np.stack([x, LARGEUR - x, y, HAUTEUR - y], axis=1) - RAYON_ROBOT
    return np.minimum(d_piliers.min(axis=1), d_murs.min(axis=1))


# --------------------------------------------------------------- garde-fou ---

def contraintes(piliers: np.ndarray, p: np.ndarray, dt: float = DT, gamma: float = GAMMA,
                distance: float = DISTANCE_SECURITE):
    """Les contraintes a·u ≥ b qu'une commande u doit respecter pour être sûre.

    Pilier de centre o et de rayon r, avec ρ = r + rayon du robot + distance de sécurité :
        h(p) = ‖p − o‖² − ρ²,   a = 2(p − o),   b = −(γ/Δt)·h(p).
    Mur, par exemple le mur de gauche : h(p) = x − (rayon du robot + distance de sécurité), a = (1, 0).
    Renvoie a (N, M, 2) et b (N, M), avec M = 8 piliers + 4 murs.
    """
    ecart = p[:, None, :] - piliers[..., :2]
    rho = piliers[..., 2] + RAYON_ROBOT + distance
    h_piliers = (ecart ** 2).sum(axis=-1) - rho ** 2
    x, y = p[:, 0], p[:, 1]
    bord = RAYON_ROBOT + distance
    h_murs = np.stack([x - bord, LARGEUR - bord - x, y - bord, HAUTEUR - bord - y], axis=1)
    a = np.concatenate([2 * ecart, np.broadcast_to(_NORMALES_MURS, (len(p), 4, 2))], axis=1)
    h = np.concatenate([h_piliers, h_murs], axis=1)
    return a, -(gamma / dt) * h


def filtrer(piliers: np.ndarray, p: np.ndarray, u: np.ndarray, dt: float = DT, gamma: float = GAMMA,
            distance: float = DISTANCE_SECURITE) -> np.ndarray:
    """La commande sûre la plus proche de u : projection exacte sur {v : a·v ≥ b}.

    En dimension 2, la projection sur un polygone convexe est soit u lui-même
    (s'il est sûr), soit sa projection sur l'une des droites a·v = b, soit un
    coin où deux droites se croisent. On essaie donc tous ces candidats et on
    garde le plus proche de u parmi ceux qui sont sûrs. La commande nulle est
    toujours sûre (b ≤ 0) : il existe toujours une solution.
    """
    a, b = contraintes(piliers, p, dt, gamma, distance)
    n, m, _ = a.shape
    i, j = np.triu_indices(m, k=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # projection de u sur chaque droite a·v = b
        t = (b - np.einsum("nmd,nd->nm", a, u)) / (a ** 2).sum(axis=-1)
        sur_droites = u[:, None, :] + t[..., None] * a
        # intersection de chaque paire de droites (règle de Cramer)
        ai, aj, bi, bj = a[:, i], a[:, j], b[:, i], b[:, j]
        det = ai[..., 0] * aj[..., 1] - ai[..., 1] * aj[..., 0]
        coins = np.stack([(bi * aj[..., 1] - bj * ai[..., 1]) / det,
                          (ai[..., 0] * bj - aj[..., 0] * bi) / det], axis=-1)
    candidats = np.concatenate([u[:, None, :], sur_droites, coins, np.zeros((n, 1, 2))], axis=1)

    finis = np.isfinite(candidats).all(axis=-1)
    candidats[~finis] = 0.0                   # droites parallèles : pas de coin
    surs = finis & (np.einsum("nmd,ncd->ncm", a, candidats) >= b[:, None, :] - TOLERANCE).all(axis=-1)
    surs[:, -1] = True                        # rester sur place ne réduit jamais la marge
    ecarts = np.where(surs, ((candidats - u[:, None, :]) ** 2).sum(axis=-1), np.inf)
    return candidats[np.arange(n), ecarts.argmin(axis=1)]


# ------------------------------------------------------------------- lidar ---

def lidar(piliers: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distances mesurées par les 24 faisceaux du lidar, depuis le centre du robot (N, 24)."""
    dx, dy = DIRECTIONS[:, 0], DIRECTIONS[:, 1]
    x, y = p[:, :1], p[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_x = np.where(dx > 0, (LARGEUR - x) / dx, np.where(dx < 0, -x / dx, np.inf))
        t_y = np.where(dy > 0, (HAUTEUR - y) / dy, np.where(dy < 0, -y / dy, np.inf))
        # rayon p + t·d contre cercle (o, r) : t² + 2t·d·(p − o) + ‖p − o‖² − r² = 0
        ecart = p[:, None, :] - piliers[..., :2]                         # (N, K, 2)
        demi_b = np.einsum("rd,nkd->nrk", DIRECTIONS, ecart)              # (N, R, K)
        c = (ecart ** 2).sum(axis=-1)[:, None, :] - piliers[:, None, :, 2] ** 2
        delta = demi_b ** 2 - c
        t = -demi_b - np.sqrt(delta)
        t = np.where(c <= 0, 0.0, np.where((delta >= 0) & (t >= 0), t, np.inf))
    return np.minimum(np.minimum(t_x, t_y), np.minimum(t.min(axis=-1), PORTEE))


def observation(piliers: np.ndarray, p: np.ndarray, buts: np.ndarray) -> np.ndarray:
    """Ce que voit le pilote neuronal (N, 27) : direction du but, distance du but (bornée à 3 m)
    et, pour chaque faisceau du lidar, la proximité de l'obstacle : 0 au-delà de 3 m, 1 au contact."""
    vers_but = buts - p
    distance = np.linalg.norm(vers_but, axis=1, keepdims=True)
    direction = vers_but / np.maximum(distance, 1e-9)
    proximite = 1 - lidar(piliers, p) / PORTEE
    return np.concatenate([direction, np.minimum(distance, PORTEE) / PORTEE, proximite], axis=1)


# -------------------------------------------------------------- simulation ---

def simuler(piliers, departs, buts, pilote, garde_fou: bool = True, pas_max: int = 600) -> dict:
    """Joue N épisodes en parallèle. Un épisode s'arrête au but, au premier contact ou après pas_max pas.

    pilote(piliers, p, buts) renvoie les vitesses proposées (N, 2).
    """
    p = departs.astype(float).copy()
    n = len(p)
    en_cours = np.ones(n, bool)
    atteint, collision = np.zeros(n, bool), np.zeros(n, bool)
    duree = np.full(n, pas_max)
    pas_joues, corriges = np.zeros(n, int), np.zeros(n, int)
    marge_min = marge(piliers, p)
    for k in range(pas_max):
        proposee = pilote(piliers, p, buts)
        appliquee = filtrer(piliers, p, proposee) if garde_fou else proposee
        corriges += en_cours & (np.linalg.norm(appliquee - proposee, axis=1) > 1e-9)
        pas_joues += en_cours
        p = np.where(en_cours[:, None], p + DT * appliquee, p)
        m = marge(piliers, p)
        marge_min = np.where(en_cours, np.minimum(marge_min, m), marge_min)
        choc = en_cours & (m < 0)
        arrivee = en_cours & ~choc & (np.linalg.norm(buts - p, axis=1) < RAYON_BUT)
        collision |= choc
        atteint |= arrivee
        duree[choc | arrivee] = k + 1
        en_cours &= ~(choc | arrivee)
        if not en_cours.any():
            break
    return {"atteint": atteint, "collision": collision, "duree": duree,
            "pas": pas_joues, "corriges": corriges, "marge_min": marge_min, "positions": p}
