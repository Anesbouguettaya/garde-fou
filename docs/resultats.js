// Résultats du banc d'essai, écrits par evaluer.py. Ne pas modifier à la main.
globalThis.RESULTATS = {
 "date": "2026-10-27",
 "scenes": 1000,
 "duree_max_s": 30.0,
 "distance_securite_cm": 2.0,
 "marge_min_cm": 2.0,
 "pas_avec_garde_fou": 890107,
 "pilotes": [
  {
   "cle": "reseau",
   "nom": "Réseau de neurones",
   "description": "lidar 24 faisceaux, appris par imitation",
   "sans": {
    "atteint": 0.796,
    "collision": 0.086,
    "corriges": 0.0,
    "duree_s": 6.315577889447237,
    "marge_min_cm": -2.614155115774736,
    "pas": 179968
   },
   "avec": {
    "atteint": 0.876,
    "collision": 0.0,
    "corriges": 0.017976425932540424,
    "duree_s": 6.459988584474885,
    "marge_min_cm": 2.0032716751889112,
    "pas": 187579
   }
  },
  {
   "cle": "naif",
   "nom": "Pilote naïf",
   "description": "tout droit vers le but",
   "sans": {
    "atteint": 0.183,
    "collision": 0.817,
    "corriges": 0.0,
    "duree_s": 3.7912568306010934,
    "marge_min_cm": -4.943469635447223,
    "pas": 35955
   },
   "avec": {
    "atteint": 1.0,
    "collision": 0.0,
    "corriges": 0.3819444444444444,
    "duree_s": 5.1264,
    "marge_min_cm": 2.0001194965631077,
    "pas": 102528
   }
  },
  {
   "cle": "hostile",
   "nom": "Pilote hostile",
   "description": "vise le pilier le plus proche",
   "sans": {
    "atteint": 0.0,
    "collision": 1.0,
    "corriges": 0.0,
    "duree_s": null,
    "marge_min_cm": -4.996227117748764,
    "pas": 11890
   },
   "avec": {
    "atteint": 0.0,
    "collision": 0.0,
    "corriges": 0.9901,
    "duree_s": null,
    "marge_min_cm": 2.000000000000007,
    "pas": 600000
   }
  }
 ]
};
