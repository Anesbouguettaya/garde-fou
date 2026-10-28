/* Garde-fou : le même cœur que garde_fou.py, en JavaScript, pour la démo web.
   Un seul robot à la fois. test_garde_fou.py vérifie que les deux versions
   donnent les mêmes résultats, au milliardième près. */
(function (racine) {
  "use strict";

  const LARGEUR = 8, HAUTEUR = 5;      // arène, en mètres
  const RAYON_ROBOT = 0.2;             // m
  const V_MAX = 1;                     // m/s
  const DT = 0.05;                     // s : boucle de commande à 20 Hz
  const GAMMA = 0.2;                   // part de la marge consommable à chaque pas
  const DISTANCE_SECURITE = 0.02;      // m : toujours 2 cm au moins entre le robot et un obstacle
  const TOLERANCE = 1e-9;
  const N_PILIERS = 8;
  const N_RAYONS = 24, PORTEE = 3;     // lidar
  const RAYON_BUT = 0.25;              // m

  const DIRECTIONS = [];
  for (let k = 0; k < N_RAYONS; k++) {
    const angle = 2 * Math.PI * k / N_RAYONS;
    DIRECTIONS.push([Math.cos(angle), Math.sin(angle)]);
  }
  const NORMALES_MURS = [[1, 0], [-1, 0], [0, 1], [0, -1]];

  function norme(x, y) { return Math.sqrt(x * x + y * y); }

  /* Distance entre le bord du robot et l'obstacle le plus proche (négative = contact). */
  function marge(piliers, p) {
    let m = Math.min(p[0], LARGEUR - p[0], p[1], HAUTEUR - p[1]) - RAYON_ROBOT;
    for (const [ox, oy, r] of piliers) m = Math.min(m, norme(p[0] - ox, p[1] - oy) - r - RAYON_ROBOT);
    return m;
  }

  /* Contraintes a·u ≥ b : h(p + Δt·u) ≥ (1 − γ)·h(p) pour chaque pilier et chaque mur. */
  function contraintes(piliers, p, dt = DT, gamma = GAMMA) {
    const a = [], h = [];
    for (const [ox, oy, r] of piliers) {
      const ex = p[0] - ox, ey = p[1] - oy, rho = r + RAYON_ROBOT + DISTANCE_SECURITE;
      a.push([2 * ex, 2 * ey]);
      h.push((ex * ex + ey * ey) - rho * rho);
    }
    const bord = RAYON_ROBOT + DISTANCE_SECURITE;
    h.push(p[0] - bord, LARGEUR - bord - p[0], p[1] - bord, HAUTEUR - bord - p[1]);
    a.push(...NORMALES_MURS);
    return { a, b: h.map((valeur) => -(gamma / dt) * valeur), h };
  }

  /* La commande sûre la plus proche de u : u lui-même, sa projection sur une
     droite a·v = b, ou un coin entre deux droites. La commande nulle est
     toujours sûre, donc il existe toujours une solution. */
  function filtrer(piliers, p, u, dt = DT, gamma = GAMMA) {
    const { a, b } = contraintes(piliers, p, dt, gamma);
    const m = b.length;
    const sure = (vx, vy) => {
      if (!Number.isFinite(vx) || !Number.isFinite(vy)) return false;
      for (let i = 0; i < m; i++) if (a[i][0] * vx + a[i][1] * vy < b[i] - TOLERANCE) return false;
      return true;
    };
    if (sure(u[0], u[1])) return [u[0], u[1]];
    let meilleur = [0, 0], ecart = u[0] * u[0] + u[1] * u[1];
    const essayer = (vx, vy) => {
      if (!sure(vx, vy)) return;
      const d = (vx - u[0]) ** 2 + (vy - u[1]) ** 2;
      if (d < ecart) { ecart = d; meilleur = [vx, vy]; }
    };
    for (let i = 0; i < m; i++) {
      const t = (b[i] - (a[i][0] * u[0] + a[i][1] * u[1])) / (a[i][0] * a[i][0] + a[i][1] * a[i][1]);
      essayer(u[0] + t * a[i][0], u[1] + t * a[i][1]);
    }
    for (let i = 0; i < m; i++) {
      for (let j = i + 1; j < m; j++) {
        const det = a[i][0] * a[j][1] - a[i][1] * a[j][0];
        essayer((b[i] * a[j][1] - b[j] * a[i][1]) / det, (a[i][0] * b[j] - a[j][0] * b[i]) / det);
      }
    }
    return meilleur;
  }

  /* Distances mesurées par les 24 faisceaux du lidar, depuis le centre du robot. */
  function lidar(piliers, p) {
    return DIRECTIONS.map(([dx, dy]) => {
      const tx = dx > 0 ? (LARGEUR - p[0]) / dx : dx < 0 ? -p[0] / dx : Infinity;
      const ty = dy > 0 ? (HAUTEUR - p[1]) / dy : dy < 0 ? -p[1] / dy : Infinity;
      let t = Math.min(tx, ty, PORTEE);
      for (const [ox, oy, r] of piliers) {
        const ex = p[0] - ox, ey = p[1] - oy;
        const demiB = dx * ex + dy * ey;
        const c = (ex * ex + ey * ey) - r * r;
        if (c <= 0) { t = 0; continue; }
        const delta = demiB * demiB - c;
        if (delta < 0) continue;
        const contact = -demiB - Math.sqrt(delta);
        if (contact >= 0) t = Math.min(t, contact);
      }
      return t;
    });
  }

  /* Ce que voit le pilote neuronal : direction du but, distance du but (bornée à 3 m) et,
     pour chaque faisceau, la proximité de l'obstacle : 0 au-delà de 3 m, 1 au contact. */
  function observation(piliers, p, but) {
    const vx = but[0] - p[0], vy = but[1] - p[1];
    const distance = norme(vx, vy), diviseur = Math.max(distance, 1e-9);
    return [vx / diviseur, vy / diviseur, Math.min(distance, PORTEE) / PORTEE,
      ...lidar(piliers, p).map((t) => 1 - t / PORTEE)];
  }

  function borner(u) {
    const n = norme(u[0], u[1]), facteur = Math.min(1, V_MAX / Math.max(n, 1e-12));
    return [u[0] * facteur, u[1] * facteur];
  }

  /* Perceptron 27 → 32 (tanh) → 2 (tanh), poids lus dans politique.js. */
  function reseau(poids, obs) {
    const cachee = poids.b1.map((biais, j) => {
      let s = 0;
      for (let i = 0; i < obs.length; i++) s += obs[i] * poids.W1[i][j];
      return Math.tanh(s + biais);
    });
    const sortie = poids.b2.map((biais, j) => {
      let s = 0;
      for (let i = 0; i < cachee.length; i++) s += cachee[i] * poids.W2[i][j];
      return V_MAX * Math.tanh(s + biais);
    });
    return borner(sortie);
  }

  /* Pilote naïf : tout droit vers le but, en ralentissant dans les 50 derniers centimètres. */
  function glouton(p, but) {
    const vx = but[0] - p[0], vy = but[1] - p[1], d = norme(vx, vy);
    const facteur = V_MAX * Math.min(1, d / 0.5) / Math.max(d, 1e-9);
    return [vx * facteur, vy * facteur];
  }

  /* Générateur pseudo-aléatoire à graine (mulberry32) : une graine donne toujours la même scène. */
  function alea(graine) {
    let etat = graine >>> 0;
    return function () {
      etat = (etat + 0x6D2B79F5) >>> 0;
      let t = etat;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Mêmes règles que scenes() en Python : 8 piliers, 50 cm de passage partout. */
  function scene(graine) {
    const hasard = alea(graine), entre = (a, b) => a + (b - a) * hasard();
    const passage = 2 * RAYON_ROBOT + 0.1;
    let piliers = [];
    while (piliers.length < N_PILIERS) {
      piliers = [];
      for (let essai = 0; essai < 500 && piliers.length < N_PILIERS; essai++) {
        const r = entre(0.2, 0.5);
        const x = entre(r + passage, LARGEUR - r - passage), y = entre(r + passage, HAUTEUR - r - passage);
        if (piliers.every(([a, b, c]) => norme(x - a, y - b) >= r + c + passage)) piliers.push([x, y, r]);
      }
    }
    const depart = pointLibre(piliers, hasard);
    let but;
    do { but = pointLibre(piliers, hasard); } while (norme(but[0] - depart[0], but[1] - depart[1]) < 3);
    return { piliers, depart, but, hasard };
  }

  function pointLibre(piliers, hasard, loinDe = null, ecartMin = 0) {
    const bord = RAYON_ROBOT + 0.1;
    for (;;) {
      const q = [bord + (LARGEUR - 2 * bord) * hasard(), bord + (HAUTEUR - 2 * bord) * hasard()];
      if (piliers.some(([x, y, r]) => norme(q[0] - x, q[1] - y) < r + RAYON_ROBOT + 0.1)) continue;
      if (loinDe && norme(q[0] - loinDe[0], q[1] - loinDe[1]) < ecartMin) continue;
      return q;
    }
  }

  const GardeFou = {
    LARGEUR, HAUTEUR, RAYON_ROBOT, V_MAX, DT, GAMMA, DISTANCE_SECURITE, N_PILIERS, N_RAYONS, PORTEE, RAYON_BUT, DIRECTIONS,
    marge, contraintes, filtrer, lidar, observation, reseau, borner, glouton, alea, scene, pointLibre,
  };
  if (typeof module === "object" && module.exports) module.exports = GardeFou;
  else racine.GardeFou = GardeFou;
})(typeof globalThis !== "undefined" ? globalThis : this);
