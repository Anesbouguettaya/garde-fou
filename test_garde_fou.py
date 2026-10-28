"""Tests de Garde-fou. Lancer : python -m unittest

Les vérifications ne réutilisent pas le code testé : la garantie est recalculée
à la main, l'optimalité est comparée à un autre algorithme (Dykstra), le lidar à
un lancer de rayon pas à pas, et la démo JavaScript au code Python.
"""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np

from garde_fou import (DISTANCE_SECURITE, DT, GAMMA, HAUTEUR, LARGEUR, N_RAYONS, PORTEE, RAYON_ROBOT, V_MAX,
                       contraintes, filtrer, lidar, marge, observation, scenes, simuler)
from pilotes import aleatoire, charger_poids, glouton, kamikaze, pilote_reseau, reseau

DOCS = Path(__file__).resolve().parent / "docs"


def etats_pres_du_contact(n, graine):
    """Des robots à moins de 30 cm de la limite de sécurité d'un pilier ou d'un mur, avec une commande au hasard."""
    rng = np.random.default_rng(graine)
    piliers, _, _ = scenes(n, graine)
    p = np.empty((n, 2))
    for s in range(n):
        while True:
            if rng.random() < 0.7:      # près d'un pilier
                x, y, r = piliers[s, rng.integers(len(piliers[s]))]
                angle, ecart = rng.uniform(0, 2 * np.pi), r + RAYON_ROBOT + DISTANCE_SECURITE + rng.uniform(0, 0.3)
                q = np.array([x + ecart * np.cos(angle), y + ecart * np.sin(angle)])
            else:                       # près d'un mur ou d'un coin
                q = rng.uniform(0, 0.3, 2) + RAYON_ROBOT + DISTANCE_SECURITE
                q = q * rng.choice([1, -1], 2)
                q = np.where(q > 0, q, [LARGEUR, HAUTEUR] + q)
            if marge(piliers[s:s + 1], q[None])[0] >= DISTANCE_SECURITE:
                p[s] = q
                break
    angle = rng.uniform(0, 2 * np.pi, n)
    u = V_MAX * np.sqrt(rng.uniform(0, 1, n))[:, None] * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    return piliers, p, u


def hauteurs(piliers, p):
    """h pour chaque pilier et chaque mur, recalculé indépendamment de garde_fou.py."""
    limite = RAYON_ROBOT + DISTANCE_SECURITE
    h = [((p[:, 0, None] - piliers[..., 0]) ** 2 + (p[:, 1, None] - piliers[..., 1]) ** 2)
         - (piliers[..., 2] + limite) ** 2]
    for valeur in (p[:, 0], LARGEUR - p[:, 0], p[:, 1], HAUTEUR - p[:, 1]):
        h.append((valeur - limite)[:, None])
    return np.concatenate(h, axis=1)


class Garantie(unittest.TestCase):

    def test_aucun_contact_quel_que_soit_le_pilote(self):
        """300 scènes, 30 s chacune : avec le garde-fou, aucun pilote ne s'approche à moins de 2 cm, même hostile."""
        piliers, departs, buts = scenes(300, graine=1)
        for nom, pilote in [("réseau de neurones", pilote_reseau()), ("glouton", glouton),
                            ("kamikaze", kamikaze), ("aléatoire", aleatoire(graine=7))]:
            with self.subTest(pilote=nom):
                r = simuler(piliers, departs, buts, pilote, garde_fou=True, pas_max=600)
                self.assertEqual(int(r["collision"].sum()), 0)
                self.assertGreaterEqual(r["marge_min"].min(), DISTANCE_SECURITE - 1e-9)

    def test_temoin_sans_garde_fou(self):
        """Témoin : sans garde-fou, le kamikaze touche à tous les coups. Le test précédent sait échouer."""
        piliers, departs, buts = scenes(100, graine=2)
        r = simuler(piliers, departs, buts, kamikaze, garde_fou=False, pas_max=600)
        self.assertTrue(r["collision"].all())

    def test_la_marge_ne_fond_jamais_trop_vite(self):
        """Pas à pas : h(p + Δt·u) ≥ (1 − γ)·h(p) pour chaque obstacle, près du contact."""
        piliers, p, u = etats_pres_du_contact(2000, graine=3)
        avant = hauteurs(piliers, p)
        apres = hauteurs(piliers, p + DT * filtrer(piliers, p, u))
        self.assertTrue(np.all(apres >= (1 - GAMMA) * avant - 1e-9))
        self.assertTrue(np.all(apres >= 0))


class Filtre(unittest.TestCase):

    def setUp(self):
        self.piliers, self.p, self.u = etats_pres_du_contact(2000, graine=4)
        self.a, self.b = contraintes(self.piliers, self.p)
        self.sortie = filtrer(self.piliers, self.p, self.u)

    def test_commande_sure_inchangee(self):
        """Une commande déjà sûre passe telle quelle, au bit près : le garde-fou n'agit que si besoin."""
        deja_sure = (np.einsum("nmd,nd->nm", self.a, self.u) >= self.b).all(axis=1)
        self.assertGreater(deja_sure.sum(), 100)
        self.assertTrue(np.array_equal(self.sortie[deja_sure], self.u[deja_sure]))
        self.assertTrue((np.linalg.norm(self.sortie[~deja_sure] - self.u[~deja_sure], axis=1) > 0).all())

    def test_toujours_une_commande_sure(self):
        """Le résultat respecte toutes les contraintes : le problème n'est jamais sans solution."""
        marges = np.einsum("nmd,nd->nm", self.a, self.sortie) - self.b
        self.assertGreaterEqual(marges.min(), -1e-9)

    def test_jamais_plus_rapide(self):
        """Le garde-fou ne fait jamais accélérer le robot."""
        self.assertTrue(np.all(np.linalg.norm(self.sortie, axis=1) <= np.linalg.norm(self.u, axis=1) + 1e-12))

    def test_correction_minimale(self):
        """La commande rendue est la commande sûre la plus proche : comparée à l'algorithme de Dykstra."""
        a, b, x = self.a, self.b, self.u.copy()
        corrections = np.zeros_like(a)
        for _ in range(3000):                  # projections alternées sur chaque contrainte, corrigées (Dykstra)
            for i in range(a.shape[1]):
                y = x + corrections[:, i]
                exces = b[:, i] - np.einsum("nd,nd->n", a[:, i], y)
                x = y + (np.maximum(exces, 0) / (a[:, i] ** 2).sum(axis=1))[:, None] * a[:, i]
                corrections[:, i] = y - x
        ecarts = np.linalg.norm(self.sortie - x, axis=1)
        self.assertLess(ecarts.max(), 1e-6, f"pire cas : n° {ecarts.argmax()}")


class Lidar(unittest.TestCase):

    def test_murs(self):
        """Au centre d'une arène vide : 4 m à l'est et à l'ouest, 2,5 m au nord et au sud."""
        vide = np.array([[[100.0, 100.0, 0.0]] * 8])         # piliers rejetés hors de l'arène
        mesure = lidar(vide, np.array([[4.0, 2.5]]))[0]
        quart = N_RAYONS // 4
        np.testing.assert_allclose(mesure[[0, quart, 2 * quart, 3 * quart]], [PORTEE, 2.5, PORTEE, 2.5])
        np.testing.assert_allclose(lidar(vide, np.array([[1.0, 2.5]]))[0][2 * quart], 1.0)

    def test_pilier_devant(self):
        """Un pilier de 50 cm de rayon, centré 1 m devant : le faisceau s'arrête à 50 cm."""
        piliers = np.array([[[5.0, 2.5, 0.5]] + [[100.0, 100.0, 0.0]] * 7])
        self.assertAlmostEqual(lidar(piliers, np.array([[4.0, 2.5]]))[0][0], 0.5)

    def test_comparaison_lancer_pas_a_pas(self):
        """Sur 200 positions, chaque faisceau est comparé à une marche pas à pas de 0,1 mm."""
        piliers, p, _ = etats_pres_du_contact(200, graine=5)
        mesure = lidar(piliers, p)
        t = np.arange(0, PORTEE + 1e-4, 1e-4)
        for k in range(200):
            points = p[k] + t[:, None, None] * np.array([[np.cos(a), np.sin(a)] for a in
                                                         2 * np.pi * np.arange(N_RAYONS) / N_RAYONS])
            dedans = ((points[..., 0] <= 0) | (points[..., 0] >= LARGEUR) |
                      (points[..., 1] <= 0) | (points[..., 1] >= HAUTEUR))
            for x, y, r in piliers[k]:
                dedans |= (points[..., 0] - x) ** 2 + (points[..., 1] - y) ** 2 <= r ** 2
            premier = np.where(dedans.any(axis=0), t[dedans.argmax(axis=0)], PORTEE)
            with self.subTest(cas=k):
                np.testing.assert_allclose(mesure[k], premier, atol=2e-4)


class Pilotes(unittest.TestCase):

    def test_reseau(self):
        """Le réseau se charge depuis docs/politique.js et ne dépasse jamais la vitesse maximale."""
        poids = charger_poids()
        self.assertEqual((poids["W1"].shape, poids["W2"].shape), ((27, 32), (32, 2)))
        piliers, p, _ = etats_pres_du_contact(500, graine=6)
        u = reseau(poids, observation(piliers, p, np.full((500, 2), [4.0, 2.5])))
        self.assertTrue(np.all(np.linalg.norm(u, axis=1) <= V_MAX + 1e-12))

    def test_le_reseau_seul_n_est_pas_sur(self):
        """Le constat de départ : sans garde-fou, le réseau entraîné percute parfois un obstacle."""
        r = simuler(*scenes(300, graine=8), pilote_reseau(), garde_fou=False)
        self.assertGreater(r["collision"].mean(), 0)
        self.assertGreater(r["atteint"].mean(), 0.5)


class Parite(unittest.TestCase):

    @unittest.skipIf(shutil.which("node") is None, "Node.js absent : parité Python / JavaScript non vérifiée")
    def test_la_demo_web_calcule_comme_python(self):
        """docs/garde_fou.js donne les mêmes commandes, mesures lidar et sorties du réseau que Python."""
        piliers, p, u = etats_pres_du_contact(400, graine=9)
        buts = np.random.default_rng(9).uniform([0.5, 0.5], [LARGEUR - 0.5, HAUTEUR - 0.5], (400, 2))
        cas = [{"piliers": piliers[k].tolist(), "p": p[k].tolist(), "u": u[k].tolist(), "but": buts[k].tolist()}
               for k in range(400)]
        script = """
            const gf = require(process.argv[1]); require(process.argv[2]);
            let entree = "";
            process.stdin.on("data", (morceau) => (entree += morceau));
            process.stdin.on("end", () => process.stdout.write(JSON.stringify(JSON.parse(entree).map((c) => ({
              filtre: gf.filtrer(c.piliers, c.p, c.u),
              lidar: gf.lidar(c.piliers, c.p),
              reseau: gf.reseau(globalThis.POLITIQUE, gf.observation(c.piliers, c.p, c.but)),
            })))));
        """
        resultat = subprocess.run(["node", "-e", script, str(DOCS / "garde_fou.js"), str(DOCS / "politique.js")],
                                  input=json.dumps(cas), capture_output=True, text=True, check=True, timeout=60)
        js = json.loads(resultat.stdout)
        poids = charger_poids()
        attendu = {"filtre": filtrer(piliers, p, u), "lidar": lidar(piliers, p),
                   "reseau": reseau(poids, observation(piliers, p, buts))}
        for cle, valeurs in attendu.items():
            with self.subTest(sortie=cle):
                np.testing.assert_allclose(np.array([c[cle] for c in js]), valeurs, rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
