#!/usr/bin/env python3
"""
Calvano et al. (2020) Q-Learning Simulation — Bachelorarbeit
=============================================================

Interaktive GUI zur Simulation algorithmischer Kollusion auf
Bertrand-Oligopol-Maerkten mit Logit-Nachfrage und Q-Learning-Agenten.

Unterstuetzt:
- Alle Modellparameter variabel (n, m, alpha, beta, delta, mu, xi, a0)
- Firmenspezifische Parameter (a_i, c_i) zur Einfuehrung von Asymmetrie
- Seed-Management ueber Konsole (/setseed, /getseed)
- Live-Plots: Preise, Gewinne, Kollusionsindex Delta
- Impulse-Response-Analyse nach abgeschlossenem Training

Abhaengigkeiten: numpy, scipy, matplotlib (tkinter ist Standardbibliothek)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
from scipy.optimize import fsolve
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
import threading
import time
import csv
import os
import queue as queue_module
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# ==========================================================
# OEKONOMISCHES MODELL (Calvano et al. 2020)
# ==========================================================

class CalvanoEnvironment:
    """Bertrand-Oligopol mit multinomialer Logit-Nachfrage."""

    def __init__(self, n, a, a0, c, mu):
        self.n = int(n)
        self.a = np.array(a, dtype=float)
        self.a0 = float(a0)
        self.c = np.array(c, dtype=float)
        self.mu = float(mu)
        if len(self.a) != self.n or len(self.c) != self.n:
            raise ValueError("Laenge von a und c muss n entsprechen.")
        if self.mu <= 0:
            raise ValueError("mu muss positiv sein.")

    def demand(self, prices):
        """Logit-Nachfrage q_i = exp((a_i-p_i)/mu) / (sum + exp(a0/mu))."""
        prices = np.asarray(prices, dtype=float)
        u = (self.a - prices) / self.mu
        u0 = self.a0 / self.mu
        m = max(u.max(), u0)
        exps = np.exp(u - m)
        denom = exps.sum() + np.exp(u0 - m)
        return exps / denom

    def profits(self, prices):
        q = self.demand(prices)
        return (np.asarray(prices, dtype=float) - self.c) * q

    def nash_prices(self):
        """Bertrand-Nash-Gleichgewicht via FOC: p_i - c_i = mu / (1 - q_i)."""
        def foc(p):
            q = self.demand(p)
            return p - self.c - self.mu / (1.0 - q)

        p0 = self.c + self.mu
        sol, info, ier, msg = fsolve(foc, p0, full_output=True)
        if ier != 1:
            # Fallback: Best-Response-Iteration mit Daempfung
            p = p0.copy()
            for _ in range(5000):
                q = self.demand(p)
                p_new = self.c + self.mu / (1.0 - q)
                if np.max(np.abs(p_new - p)) < 1e-10:
                    return p_new
                p = 0.5 * p + 0.5 * p_new
            return p
        return sol

    def monopoly_prices(self):
        """Joint-Profit-Maximum: direkte Maximierung (robuster als FOC-Loesung)."""
        from scipy.optimize import minimize

        def neg_joint_profit(p):
            return -float(np.sum(self.profits(p)))

        # Startwert deutlich oberhalb Nash-Niveau
        p_nash = self.nash_prices()
        p0 = p_nash + self.mu

        # Schranken: ueber Grenzkosten, mit grosszuegigem Oberwert
        lower = self.c + 1e-4
        upper = self.c + 30.0 * self.mu
        bounds = list(zip(lower, upper))

        result = minimize(
            neg_joint_profit, p0, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-10},
        )
        if result.success:
            return result.x
        # Fallback: ein paar Startpunkte ausprobieren
        best_p = p0
        best_obj = neg_joint_profit(p0)
        for mult in (1.5, 2.0, 3.0, 5.0):
            p_try = p_nash * mult
            r = minimize(
                neg_joint_profit, p_try, method="L-BFGS-B", bounds=bounds,
                options={"ftol": 1e-12, "gtol": 1e-10},
            )
            if r.success and r.fun < best_obj:
                best_obj = r.fun
                best_p = r.x
        return best_p


# ==========================================================
# Q-LEARNING-SIMULATION
# ==========================================================

class QLearningSimulation:
    # Drei waehlbare Initialisierungsstrategien fuer die Q-Tabelle
    # (siehe Ideen.docx, Abschnitt 1.1):
    #   "best_response": Erwartungswert des abdiskontierten Gewinns bei
    #                    gleichverteilten Gegnerstrategien. DAS ENTSPRICHT DER
    #                    CALVANO-BASELINE (Calvano et al. 2020, S. 3276): die
    #                    Firma rechnet vorab den durchschnittlichen Gewinn aus,
    #                    den sie pro Aktion gegen uniform zufaellig spielende
    #                    Gegner erzielen wuerde.
    #   "zeros":         alle Q-Werte = 0. NICHT Calvano. Diese Wahl wirkt bei
    #                    strikt positiven Belohnungen optimistisch und erzeugt
    #                    eine zusaetzliche Explorations-Tendenz.
    #   "random":        uniform in [0, max moeglicher abdiskontierter Gewinn].
    #                    Bricht systematische Symmetrien der best_response-Wahl auf
    #                    und ist die uebliche Robustheitspruefung gegen Pfadabhaengigkeit.
    INIT_STRATEGIES = ("best_response", "zeros", "random")

    def __init__(self, env, m, alpha, beta, delta, xi, rng,
                 init_strategy: str = "best_response",
                 track_entropy: bool = False,
                 entropy_window: int = 1000):
        self.env = env
        self.n = env.n
        self.m = int(m)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.xi = float(xi)
        self.rng = rng

        if init_strategy not in self.INIT_STRATEGIES:
            raise ValueError(
                f"init_strategy '{init_strategy}' unbekannt. "
                f"Erlaubt: {self.INIT_STRATEGIES}")
        self.init_strategy = init_strategy

        # Gleichgewichtspreise als Referenz
        self.p_nash = env.nash_prices()
        self.p_mono = env.monopoly_prices()
        self.pi_nash = env.profits(self.p_nash)
        self.pi_mono = env.profits(self.p_mono)

        # Aktionsraum (einheitlich, basierend auf Mittelwerten der
        # Gleichgewichtspreise, Erweiterung um xi)
        p_n_mean = float(np.mean(self.p_nash))
        p_m_mean = float(np.mean(self.p_mono))
        self.p_min = p_n_mean - xi * (p_m_mean - p_n_mean)
        self.p_max = p_m_mean + xi * (p_m_mean - p_n_mean)
        self.prices = np.linspace(self.p_min, self.p_max, self.m)

        # Q-Tabellen: eine pro Firma, Shape (m^n, m)
        self.n_states = self.m ** self.n
        self.Q = np.zeros((self.n, self.n_states, self.m), dtype=np.float64)

        # Initialisierung gemaess gewaehlter Strategie.
        # best_response ist der Default und entspricht der Calvano-Baseline.
        if self.init_strategy == "best_response":
            self._init_Q_best_response()
        elif self.init_strategy == "zeros":
            pass  # Q bleibt mit Nullen gefuellt (alternative, NICHT Calvano)
        elif self.init_strategy == "random":
            self._init_Q_random()

        # ----- Diagnostik-Tracking (siehe Ideen.docx, Abschnitte 1.3 und 1.4) -----
        # 1) Exploration vs Exploitation: zaehle, wie oft Aktionen zufaellig
        #    gewaehlt wurden (Exploration) und wie oft argmax (Exploitation).
        #    Sowohl global als auch pro Firma, um asymmetrische Exploration zu erkennen.
        self.explore_count = 0
        self.exploit_count = 0
        self.explore_per_firm = np.zeros(self.n, dtype=np.int64)
        self.exploit_per_firm = np.zeros(self.n, dtype=np.int64)
        # 2) Visits pro (Firma, State, Aktion): wie oft wurde dieser Eintrag
        #    aktualisiert? Wichtig, weil seltene Updates eine unzuverlaessige
        #    Q-Schaetzung bedeuten. Die Visits-Heatmap ist eine zentrale
        #    Diagnostik-Abbildung fuer die Bachelorarbeit.
        self.visits = np.zeros((self.n, self.n_states, self.m), dtype=np.int64)
        # 3) Optional Aktionsentropie: rollierendes Fenster der letzten Aktionen,
        #    aus dem H_t = -sum p_a log p_a berechnet wird. Kostet RAM, daher per Flag.
        self.track_entropy = bool(track_entropy)
        self.entropy_window = int(entropy_window)
        if self.track_entropy and self.entropy_window > 0:
            self._entropy_buffer = np.full(
                (self.n, self.entropy_window), -1, dtype=np.int16)
        else:
            self._entropy_buffer = None
        self._entropy_buffer_idx = 0

        # 4) Konvergenz-Tracking (Calvano-Konvention: greedy-Politik stabil ueber
        #    N aufeinanderfolgende Perioden). Wir pruefen alle 1000 Perioden, ob
        #    die argmax-Aktionen aller Firmen im aktuellen Zustand sich gegenueber
        #    der letzten Pruefung geaendert haben. Bleibt die greedy-Politik
        #    convergence_threshold Perioden lang stabil, gilt der Lauf als konvergiert.
        self.convergence_threshold = 100_000
        self._convergence_check_interval = 1000
        self._last_greedy = None       # zuletzt geprueftes greedy-Tupel
        self._last_greedy_change_t = 0 # Zeitpunkt der letzten Aenderung

        self.t = 0
        self.current_state = int(self.rng.integers(0, self.n_states))

    # ----------------- Initialisierungsstrategien -----------------
    def _init_Q_best_response(self):
        """Erwartungswert bei Gleichverteilung der Gegner.

        Inhaltlich: Vor dem ersten Schritt rechnet jede Firma den durchschnittlichen
        Periodengewinn aus, den sie mit Aktion a erzielen wuerde, wenn die Gegner
        gleichverteilt aus dem Aktionsraum ziehen. Dieser Erwartungswert wird durch
        (1 - delta) geteilt, damit er den Gegenwartswert einer unendlichen Konsumrente
        approximiert. Oekonomische Lesart: die Firma macht eine grobe Marktanalyse,
        bevor sie den Algorithmus startet.
        """
        # Bei kleinem Aktionsraum exakt enumerieren, sonst Monte-Carlo-Sampling
        full_size = self.m ** (self.n - 1) if self.n > 1 else 1
        use_enum = full_size <= 20000
        n_samples = 2000

        for firm in range(self.n):
            other_idx = [i for i in range(self.n) if i != firm]
            for a in range(self.m):
                p = np.zeros(self.n)
                p[firm] = self.prices[a]
                total = 0.0
                if use_enum and self.n > 1:
                    count = 0
                    for combo in np.ndindex(*([self.m] * (self.n - 1))):
                        for j, act in enumerate(combo):
                            p[other_idx[j]] = self.prices[act]
                        total += self.env.profits(p)[firm]
                        count += 1
                    avg = total / count
                elif self.n == 1:
                    avg = self.env.profits(p)[firm]
                else:
                    for _ in range(n_samples):
                        for j in other_idx:
                            p[j] = self.prices[self.rng.integers(0, self.m)]
                        total += self.env.profits(p)[firm]
                    avg = total / n_samples
                self.Q[firm, :, a] = avg / (1.0 - self.delta)

    def _init_Q_random(self):
        """Zufallsinitialisierung in [0, Q_max].

        Q_max ist eine obere Schranke fuer den abdiskontierten Gewinn pro Firma.
        Wir verwenden den maximal moeglichen Periodengewinn (Monopolfall)
        geteilt durch (1 - delta). Diese Wahl ist optimistisch, vermeidet aber
        die exakt symmetrischen Startwerte der best_response-Strategie und
        bricht so systematische Pfadabhaengigkeiten in der Konvergenz.
        """
        pi_max = float(np.max(self.pi_mono))
        q_max = pi_max / max(1.0 - self.delta, 1e-9)
        self.Q[:] = self.rng.uniform(0.0, q_max, size=self.Q.shape)

    def _state_from_actions(self, actions):
        """Kodiere Aktionstupel (a_1, ..., a_n) als flachen State-Index."""
        idx = 0
        for a in actions:
            idx = idx * self.m + int(a)
        return idx

    def step(self):
        # Aktuelle Explorationswahrscheinlichkeit gemaess Calvano-Spezifikation
        epsilon = float(np.exp(-self.beta * self.t))

        # ---------- Aktionswahl: epsilon-greedy pro Firma ----------
        # Wir loggen pro Firma, ob die Aktion zufaellig (Exploration)
        # oder argmax (Exploitation) gewaehlt wurde. Daraus lassen sich
        # spaeter Diagnose-Plots zum Exploration-Exploitation-Verhaeltnis bauen.
        actions = np.empty(self.n, dtype=np.int64)
        for i in range(self.n):
            if self.rng.random() < epsilon:
                actions[i] = self.rng.integers(0, self.m)
                self.explore_count += 1
                self.explore_per_firm[i] += 1
            else:
                actions[i] = int(np.argmax(self.Q[i, self.current_state]))
                self.exploit_count += 1
                self.exploit_per_firm[i] += 1

        # ---------- Optional: Aktionsentropie-Tracking ----------
        # Wenn aktiviert, schreiben wir die zuletzt gewaehlte Aktion jeder Firma
        # in einen Ringpuffer der Laenge entropy_window. Aus diesem Puffer
        # berechnet action_entropy() spaeter H_t pro Firma.
        if self._entropy_buffer is not None:
            self._entropy_buffer[:, self._entropy_buffer_idx] = actions.astype(np.int16)
            self._entropy_buffer_idx = (
                (self._entropy_buffer_idx + 1) % self.entropy_window)

        prices = self.prices[actions]
        profits = self.env.profits(prices)
        next_state = self._state_from_actions(actions)

        # ---------- Q-Update (Bellman-Rekursion) ----------
        # Gleichzeitig zaehlen wir, wie oft jeder (state, action)-Eintrag aktualisiert wurde.
        for i in range(self.n):
            best_next = float(np.max(self.Q[i, next_state]))
            old = self.Q[i, self.current_state, actions[i]]
            self.Q[i, self.current_state, actions[i]] = (
                (1.0 - self.alpha) * old
                + self.alpha * (profits[i] + self.delta * best_next)
            )
            self.visits[i, self.current_state, actions[i]] += 1

        self.current_state = next_state
        self.t += 1

        # ---------- Konvergenzpruefung (alle N Perioden) ----------
        # Vergleich: hat sich die greedy-Aktion einer Firma im (neuen) aktuellen
        # Zustand gegenueber der letzten Pruefung veraendert? Wenn ja, setzen
        # wir den Stabilitaetszaehler zurueck. Wenn nein, akkumuliert er.
        if self.t % self._convergence_check_interval == 0:
            greedy = np.array([int(np.argmax(self.Q[i, self.current_state]))
                                for i in range(self.n)])
            if self._last_greedy is None or not np.array_equal(greedy, self._last_greedy):
                self._last_greedy_change_t = self.t
                self._last_greedy = greedy

        return prices, profits, epsilon

    def convergence_status(self):
        """Liefert ein Tupel (status, periods_stable, threshold) zurueck.

        status ist eine der drei Phasen:
          - 'lernt'         : weniger als 25% des Schwellenwerts ohne Aenderung
          - 'stabilisiert'  : 25% bis 100% des Schwellenwerts ohne Aenderung
          - 'konvergiert'   : >= 100% des Schwellenwerts ohne Aenderung
        periods_stable ist die Anzahl Perioden seit der letzten beobachteten
        Aenderung der greedy-Politik im aktuellen Zustand.
        """
        periods_stable = self.t - self._last_greedy_change_t
        if periods_stable >= self.convergence_threshold:
            status = 'konvergiert'
        elif periods_stable >= self.convergence_threshold // 4:
            status = 'stabilisiert'
        else:
            status = 'lernt'
        return status, periods_stable, self.convergence_threshold

    def collusion_index(self, profits):
        """Delta = (pi_avg - pi_Nash) / (pi_Mono - pi_Nash)."""
        num = float(np.mean(profits) - np.mean(self.pi_nash))
        den = float(np.mean(self.pi_mono) - np.mean(self.pi_nash))
        if abs(den) < 1e-12:
            return 0.0
        return num / den

    # ---------- Diagnostik-Hilfsfunktionen ----------
    def explore_share(self) -> float:
        """Gesamtanteil exploratorischer Aktionen ueber alle bisherigen Schritte.

        Wert in [0, 1]. Ein hoher Wert bedeutet, dass der Algorithmus
        bisher viel zufaellig gewaehlt hat; ein niedriger Wert, dass er
        schon weitgehend greedy spielt.
        """
        total = self.explore_count + self.exploit_count
        return 0.0 if total == 0 else self.explore_count / total

    def action_entropy(self) -> np.ndarray:
        """Aktionsentropie H_t pro Firma ueber das letzte entropy_window.

        Misst die Streuung der zuletzt gewaehlten Aktionen. Konvergiert die
        Politik gegen eine deterministische Strategie, faellt H gegen 0.
        Rueckgabe ist ein Vektor der Laenge n; gibt np.nan zurueck, wenn
        Tracking nicht aktiv ist oder noch zu wenige Daten vorliegen.
        """
        if self._entropy_buffer is None:
            return np.full(self.n, np.nan)
        H = np.zeros(self.n)
        for i in range(self.n):
            window = self._entropy_buffer[i]
            valid = window[window >= 0]
            if len(valid) < 2:
                H[i] = np.nan
                continue
            counts = np.bincount(valid, minlength=self.m).astype(float)
            probs = counts / counts.sum()
            # Klassische Shannon-Entropie; 0-Logarithmen ausschliessen
            nz = probs[probs > 0]
            H[i] = float(-(nz * np.log(nz)).sum())
        return H

    def epsilon_at(self, t: int) -> float:
        """Explorationswahrscheinlichkeit zur Iteration t (deterministische Kurve)."""
        return float(np.exp(-self.beta * t))

    def suggest_episodes(self, target_updates_per_entry: int = 100) -> int:
        """Vorschlag fuer eine faire Iterationsanzahl T.

        Damit jeder Q-Tabellen-Eintrag im Erwartungswert mindestens
        `target_updates_per_entry` Updates erhaelt, muss
            T >= target_updates_per_entry * m^n
        gelten (vereinfachtes Argument: jede Periode aktualisiert genau
        n Eintraege, davon ist nur einer pro Firma der aktuelle State-Aktion).
        Diese Heuristik liefert die untere Schranke fuer T(n).
        """
        return int(target_updates_per_entry * self.n_states)


# ==========================================================
# BATCH-SIMULATION: Replikationen ueber mehrere Seeds
# ==========================================================

@dataclass
class BatchConfig:
    """Konfiguration eines Batch-Laufs (alle Replikationen identisch ausser Seed)."""
    n: int
    m: int
    alpha: float
    beta: float
    delta: float
    mu: float
    a0: float
    xi: float
    a: List[float]
    c: List[float]
    episodes: int
    seeds: List[int]
    # Aufzeichnung der letzten Periode-Statistiken (gemittelt)
    avg_window: int = 1000  # Mittelung von Delta ueber die letzten N Iterationen
    save_q_tables: bool = False  # Q-Tabellen pro Replikation zurueckgeben?
    # Neu (siehe Ideen.docx):
    # 1.1: waehlbare Q-Initialisierungsstrategie
    init_strategy: str = "best_response"
    # 1.3: zusaetzliche Diagnostik einsammeln (Visits-Heatmap, Explore-Statistik)
    save_visits: bool = False
    # 1.4: damit Replikationen pro Eintrag fair vergleichbar bleiben, koennen
    # T und avg_window automatisch proportional zu m^n skaliert werden.
    # Diese Option wird in der GUI gesetzt, hier bleibt sie informativ.


@dataclass
class RunResult:
    """Ergebnis einer einzelnen Replikation."""
    seed: int
    final_prices: List[float]
    final_profits: List[float]
    final_delta: float          # Mittelwert ueber letzte avg_window Iterationen
    p_nash: List[float]
    p_mono: List[float]
    converged: bool             # Optimal action konstant ueber letzte 10% der Iterationen?
    elapsed_s: float
    error: str = ""
    q_table: Optional[np.ndarray] = None    # Shape (n, m^n, m), nur wenn angefordert
    action_prices: Optional[np.ndarray] = None  # m-Vektor der diskretisierten Preise
    # Neu: aggregierte Exploration-Statistik (siehe Ideen.docx, 1.3).
    # Wird auch ohne save_q_tables zurueckgegeben, weil die Werte sehr klein sind.
    explore_count: int = 0
    exploit_count: int = 0
    explore_per_firm: Optional[List[int]] = None
    exploit_per_firm: Optional[List[int]] = None
    # Optional: Visits-Heatmap pro Firma (Shape n, m^n, m). Nur wenn save_visits aktiv.
    visits: Optional[np.ndarray] = None


def _run_single_replication(args) -> RunResult:
    """Worker-Funktion fuer einen einzelnen Lauf — wird in Subprozess ausgefuehrt.

    Wichtig: Diese Funktion muss auf Modulebene definiert sein, damit
    multiprocessing sie picklen kann.

    args = (cfg, seed, progress_queue)
        progress_queue: optional eine multiprocessing.Queue, in die periodisch
        Fortschritts-Updates geschrieben werden. Format pro Update:
            ("progress", seed, t, total, current_delta, current_eps)
        Wenn None, kein Reporting.
    """
    cfg, seed, progress_queue = args
    t0 = time.time()

    def report(kind: str, **kwargs):
        if progress_queue is not None:
            try:
                progress_queue.put_nowait((kind, seed, kwargs))
            except Exception:
                pass  # Queue voll oder geschlossen — Updates verlieren ist okay

    try:
        env = CalvanoEnvironment(n=cfg.n, a=cfg.a, a0=cfg.a0, c=cfg.c, mu=cfg.mu)
        rng = np.random.default_rng(seed)
        # init_strategy aus cfg uebernehmen, damit der Batch-Lauf alle
        # Replikationen mit gleicher Initialisierung startet (saubere Methodik).
        sim = QLearningSimulation(
            env=env, m=cfg.m, alpha=cfg.alpha, beta=cfg.beta,
            delta=cfg.delta, xi=cfg.xi, rng=rng,
            init_strategy=getattr(cfg, "init_strategy", "best_response"),
        )
        report("started", episodes=cfg.episodes)

        # Konvergenz-Tracking
        convergence_window = max(10000, cfg.episodes // 10)
        last_change = 0
        prev_greedy = np.array([np.argmax(sim.Q[i, sim.current_state])
                                 for i in range(cfg.n)])

        # Aufzeichnung der letzten avg_window Werte
        avg_w = min(cfg.avg_window, cfg.episodes)
        recent_profits = np.zeros((avg_w, cfg.n))
        recent_prices = np.zeros((avg_w, cfg.n))
        recent_deltas = np.zeros(avg_w)

        # Progress-Reporting: ca. 100 Updates pro Lauf, mind. alle 50k
        progress_interval = max(50000, cfg.episodes // 100)

        for t in range(cfg.episodes):
            prices, profits, eps = sim.step()
            idx = t % avg_w
            recent_prices[idx] = prices
            recent_profits[idx] = profits
            recent_deltas[idx] = sim.collusion_index(profits)

            if t % 1000 == 0:
                greedy = np.array([np.argmax(sim.Q[i, sim.current_state])
                                   for i in range(cfg.n)])
                if not np.array_equal(greedy, prev_greedy):
                    last_change = t
                    prev_greedy = greedy

            # Progress-Update an Queue
            if t > 0 and t % progress_interval == 0:
                # Aktuelles Delta als Mittel ueber bisher gefuellte recent_deltas
                fill = min(t + 1, avg_w)
                cur_delta = float(np.mean(recent_deltas[:fill]) if fill < avg_w
                                  else recent_deltas.mean())
                report("progress", t=t, eps=float(eps), delta=cur_delta)

        converged = (cfg.episodes - last_change) >= convergence_window

        # Visits-Heatmap nur kopieren, wenn explizit angefordert: spart bei
        # grossen Konfigurationen viel Speicher und Inter-Prozess-Kommunikation.
        save_visits = getattr(cfg, "save_visits", False)
        result = RunResult(
            seed=seed,
            final_prices=recent_prices.mean(axis=0).tolist(),
            final_profits=recent_profits.mean(axis=0).tolist(),
            final_delta=float(recent_deltas.mean()),
            p_nash=sim.p_nash.tolist(),
            p_mono=sim.p_mono.tolist(),
            converged=converged,
            elapsed_s=time.time() - t0,
            q_table=sim.Q.copy() if cfg.save_q_tables else None,
            action_prices=sim.prices.copy() if cfg.save_q_tables else None,
            # Exploration-Statistik (klein, daher immer mitgegeben)
            explore_count=int(sim.explore_count),
            exploit_count=int(sim.exploit_count),
            explore_per_firm=sim.explore_per_firm.tolist(),
            exploit_per_firm=sim.exploit_per_firm.tolist(),
            # Visits-Heatmap (gross, daher nur auf Wunsch)
            visits=sim.visits.copy() if save_visits else None,
        )
        report("done", delta=result.final_delta, converged=converged,
               elapsed=result.elapsed_s)
        return result
    except Exception as e:
        report("error", message=f"{type(e).__name__}: {e}")
        return RunResult(
            seed=seed, final_prices=[], final_profits=[],
            final_delta=float("nan"), p_nash=[], p_mono=[],
            converged=False, elapsed_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )


class BatchRunner:
    """Orchestriert parallel ausgefuehrte Replikationen."""

    def __init__(self, cfg: BatchConfig, n_workers: Optional[int] = None):
        self.cfg = cfg
        self.n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
        self.results: List[RunResult] = []
        self._executor: Optional[ProcessPoolExecutor] = None
        self._futures = []
        self._cancelled = False

    def run(self, on_progress=None, on_done=None, on_live_update=None):
        """Startet alle Replikationen parallel. Callbacks aus Thread aufgerufen.

        on_progress(result, done, total): wird aufgerufen wenn ein Lauf fertig ist
        on_done(results, cancelled): nach Abschluss aller Laeufe
        on_live_update(seed, kind, info): Echtzeit-Updates aus Workern (started,
            progress, done, error). Wird aus dem Polling-Thread heraus aufgerufen.
        """
        self.results = []
        self._cancelled = False

        # Manager-Queue fuer Live-Updates aus den Subprozessen
        self._manager = mp.Manager()
        self._progress_queue = self._manager.Queue()

        self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        self._futures = [
            self._executor.submit(_run_single_replication,
                                   (self.cfg, s, self._progress_queue))
            for s in self.cfg.seeds
        ]

        # Polling-Thread, der die Queue auslaesst
        self._poll_stop = False

        def poll_loop():
            while not self._poll_stop:
                try:
                    msg = self._progress_queue.get(timeout=0.2)
                except queue_module.Empty:
                    continue
                except Exception:
                    break
                if on_live_update:
                    try:
                        kind, seed, info = msg
                        on_live_update(seed, kind, info)
                    except Exception:
                        pass

        self._poll_thread = threading.Thread(target=poll_loop, daemon=True)
        self._poll_thread.start()

        try:
            for fut in as_completed(self._futures):
                if self._cancelled:
                    break
                res = fut.result()
                self.results.append(res)
                if on_progress:
                    on_progress(res, len(self.results), len(self.cfg.seeds))
        finally:
            self._poll_stop = True
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None
            self._progress_queue = None
            if on_done:
                on_done(self.results, self._cancelled)

    def cancel(self):
        self._cancelled = True
        if self._executor is not None:
            for f in self._futures:
                f.cancel()


def aggregate_results(results: List[RunResult]) -> dict:
    """Aggregiere finale Delta-Werte aus erfolgreichen Replikationen."""
    ok = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    if not ok:
        return dict(
            n_runs=0, n_failed=len(failed),
            mean_delta=float("nan"), std_delta=float("nan"),
            min_delta=float("nan"), max_delta=float("nan"),
            median_delta=float("nan"),
            n_converged=0, deltas=[],
        )
    deltas = np.array([r.final_delta for r in ok])
    return dict(
        n_runs=len(ok),
        n_failed=len(failed),
        mean_delta=float(deltas.mean()),
        std_delta=float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        min_delta=float(deltas.min()),
        max_delta=float(deltas.max()),
        median_delta=float(np.median(deltas)),
        n_converged=sum(1 for r in ok if r.converged),
        deltas=deltas.tolist(),
    )


def estimate_q_tables_size_bytes(cfg: BatchConfig) -> int:
    """Schaetze Speichergroesse aller Q-Tabellen in Bytes (vor Komprimierung).

    Pro Replikation: n * m^n * m float64-Werte = 8 Byte pro Eintrag.
    Komprimiertes .npz ist meist deutlich kleiner (Faktor 2-10), aber wir
    schaetzen konservativ.
    """
    per_run = cfg.n * (cfg.m ** cfg.n) * cfg.m * 8
    return per_run * len(cfg.seeds)


def export_q_tables_npz(cfg: BatchConfig, results: List[RunResult],
                         path: str) -> None:
    """Speichere alle Q-Tabellen in einer komprimierten .npz-Datei.

    Aufbau der Datei:
      - 'config_n', 'config_m', 'config_alpha', ... : Modellparameter (Skalare)
      - 'config_a', 'config_c' : firmenspezifische Vektoren
      - 'config_episodes' : Skalar
      - 'seeds' : Array der Seeds (sortiert)
      - 'action_prices' : der diskretisierte Preisraum (laenge m)
                          identisch fuer alle Laeufe da Konfiguration gleich
      - 'p_nash' : Nash-Preise pro Firma (laenge n)
      - 'p_mono' : Monopol-Preise pro Firma (laenge n)
      - 'Q_seed_<S>' : Q-Tabelle fuer Seed S, Shape (n, m^n, m)
    """
    ok = [r for r in results if not r.error and r.q_table is not None]
    if not ok:
        raise ValueError("Keine Q-Tabellen verfuegbar (save_q_tables nicht aktiv "
                         "oder alle Laeufe fehlgeschlagen).")
    ok_sorted = sorted(ok, key=lambda r: r.seed)

    payload = {
        # Konfiguration
        "config_n": np.int64(cfg.n),
        "config_m": np.int64(cfg.m),
        "config_alpha": np.float64(cfg.alpha),
        "config_beta": np.float64(cfg.beta),
        "config_delta": np.float64(cfg.delta),
        "config_mu": np.float64(cfg.mu),
        "config_a0": np.float64(cfg.a0),
        "config_xi": np.float64(cfg.xi),
        "config_a": np.array(cfg.a, dtype=np.float64),
        "config_c": np.array(cfg.c, dtype=np.float64),
        "config_episodes": np.int64(cfg.episodes),
        "config_avg_window": np.int64(cfg.avg_window),
        # Referenzen
        "seeds": np.array([r.seed for r in ok_sorted], dtype=np.int64),
        "action_prices": ok_sorted[0].action_prices.astype(np.float64),
        "p_nash": np.array(ok_sorted[0].p_nash, dtype=np.float64),
        "p_mono": np.array(ok_sorted[0].p_mono, dtype=np.float64),
    }
    # Eine Q-Tabelle pro Lauf
    for r in ok_sorted:
        payload[f"Q_seed_{r.seed}"] = r.q_table.astype(np.float64)
    np.savez_compressed(path, **payload)


def export_q_tables_csv(cfg: BatchConfig, results: List[RunResult],
                          out_dir: str) -> List[str]:
    """Schreibe jede Q-Tabelle als CSV-Datei in out_dir.

    Dateischema: q_table_seed{S}_agent{i+1}.csv

    Die Q-Tabelle wird zweidimensional ausgeschrieben:
      - eine Zeile pro State (Indexspalte "state_index" und zusaetzlich die
        Aktionstupel-Codierung "state_actions" zur Lesbarkeit)
      - eine Spalte pro Aktion (price_1, price_2, ..., wobei der Spaltenkopf
        den diskretisierten Preiswert enthaelt)

    Diese Form ist die im Betreuergespraech ausdruecklich angefragte Form
    (siehe Ideen.docx, Abschnitt 1.2). Sie ist gross, aber unmittelbar in
    Excel oder pandas einlesbar.
    """
    ok = [r for r in results if not r.error and r.q_table is not None]
    if not ok:
        raise ValueError("Keine Q-Tabellen verfuegbar.")
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    n, m = cfg.n, cfg.m
    n_states = m ** n

    # Hilfsfunktion: State-Index -> menschenlesbares Aktionstupel
    def decode_state(idx: int) -> str:
        actions = []
        rest = idx
        for _ in range(n):
            actions.append(rest % m)
            rest //= m
        actions.reverse()
        return "(" + ",".join(str(a) for a in actions) + ")"

    state_labels = [decode_state(s) for s in range(n_states)]

    for r in sorted(ok, key=lambda x: x.seed):
        for i in range(n):
            path_i = os.path.join(
                out_dir, f"q_table_seed{r.seed}_agent{i+1}.csv")
            with open(path_i, "w", newline="", encoding="utf-8") as f:
                # Kopfkommentar mit der Konfiguration
                f.write(f"# Q-Tabelle Firma {i+1}, Seed {r.seed}\n")
                f.write(f"# n={cfg.n}, m={cfg.m}, alpha={cfg.alpha}, "
                        f"beta={cfg.beta}, delta={cfg.delta}, mu={cfg.mu}\n")
                f.write(f"# init_strategy={getattr(cfg, 'init_strategy', 'best_response')}\n")
                f.write(f"# episodes={cfg.episodes}\n")
                writer = csv.writer(f)
                # Header: state_index | state_actions | preis_1 | ... | preis_m
                header = ["state_index", "state_actions"]
                for a in range(m):
                    header.append(f"a{a+1}_p={r.action_prices[a]:.4f}")
                writer.writerow(header)
                Qi = r.q_table[i]  # Shape (n_states, m)
                for s in range(n_states):
                    row = [s, state_labels[s]]
                    row.extend(f"{Qi[s, a]:.6f}" for a in range(m))
                    writer.writerow(row)
            written.append(path_i)
    return written


def export_visits_csv(cfg: BatchConfig, results: List[RunResult],
                        out_dir: str) -> List[str]:
    """Schreibe die Visits-Matrix (wie oft jeder Q-Eintrag besucht wurde) als CSV.

    Dateischema: visits_seed{S}_agent{i+1}.csv

    Das ist die diagnostische Tabelle aus Ideen.docx, Abschnitt 1.4.
    Sie zeigt, ob die Q-Schaetzungen flaechig verteilt sind oder ob das
    Verfahren in wenigen Zustaenden festhaengt.
    """
    ok = [r for r in results if not r.error and r.visits is not None]
    if not ok:
        raise ValueError("Keine Visits-Daten verfuegbar (save_visits nicht aktiv?).")
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    n, m = cfg.n, cfg.m
    for r in sorted(ok, key=lambda x: x.seed):
        for i in range(n):
            path_i = os.path.join(
                out_dir, f"visits_seed{r.seed}_agent{i+1}.csv")
            np.savetxt(path_i, r.visits[i], fmt="%d", delimiter=",",
                       header=f"Visits Firma {i+1}, Seed {r.seed}; "
                              f"Zeilen=State (m^n), Spalten=Aktion",
                       comments="# ")
            written.append(path_i)
    return written


def export_results_csv(cfg: BatchConfig, results: List[RunResult],
                        path: str) -> None:
    """Schreibe Header mit Konfiguration + eine Zeile pro Replikation."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        # Konfigurations-Header als Kommentarzeilen
        f.write(f"# Calvano Q-Learning Batch Results\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# n={cfg.n}, m={cfg.m}, alpha={cfg.alpha}, beta={cfg.beta}\n")
        f.write(f"# delta={cfg.delta}, mu={cfg.mu}, a0={cfg.a0}, xi={cfg.xi}\n")
        f.write(f"# a={cfg.a}, c={cfg.c}\n")
        f.write(f"# episodes={cfg.episodes}, avg_window={cfg.avg_window}\n")
        f.write(f"# n_replications={len(results)}\n")
        f.write(f"# init_strategy={getattr(cfg, 'init_strategy', 'best_response')}\n")
        writer = csv.writer(f)
        n = cfg.n
        header = ["seed", "converged", "elapsed_s", "final_delta"]
        header += [f"final_price_{i+1}" for i in range(n)]
        header += [f"final_profit_{i+1}" for i in range(n)]
        header += [f"p_nash_{i+1}" for i in range(n)]
        header += [f"p_mono_{i+1}" for i in range(n)]
        # Exploration-Anteil als zusaetzliche Spalte: nuetzlich, um auf einen
        # Blick zu sehen, ob ein Lauf vorzeitig in die Exploitation gerutscht ist.
        header += ["explore_share"]
        header += [f"explore_count_{i+1}" for i in range(n)]
        header += [f"exploit_count_{i+1}" for i in range(n)]
        header += ["error"]
        writer.writerow(header)
        for r in sorted(results, key=lambda x: x.seed):
            row = [r.seed, int(r.converged), f"{r.elapsed_s:.2f}",
                   f"{r.final_delta:.6f}"]
            row += [f"{p:.6f}" for p in (r.final_prices or [float("nan")] * n)]
            row += [f"{p:.6f}" for p in (r.final_profits or [float("nan")] * n)]
            row += [f"{p:.6f}" for p in (r.p_nash or [float("nan")] * n)]
            row += [f"{p:.6f}" for p in (r.p_mono or [float("nan")] * n)]
            total = r.explore_count + r.exploit_count
            share = (r.explore_count / total) if total > 0 else 0.0
            row.append(f"{share:.4f}")
            exp_pf = r.explore_per_firm or [0] * n
            exl_pf = r.exploit_per_firm or [0] * n
            row += [str(int(x)) for x in exp_pf]
            row += [str(int(x)) for x in exl_pf]
            row.append(r.error)
            writer.writerow(row)


# ==========================================================
# GUI
# ==========================================================

class CalvanoGUI:
    MAX_FIRMS = 10

    def __init__(self, root):
        self.root = root
        self.root.title("Calvano Q-Learning Simulation — Bachelorarbeit")
        self.root.geometry("1500x900")

        self.seed = 42
        self.rng = np.random.default_rng(self.seed)
        self.simulation = None
        self.running = False
        self.worker_thread = None

        # Verlaufsdaten fuer die Live-Plots
        self.history_t = []
        self.history_prices = []  # Liste von Listen: pro Firma
        self.history_profits = []
        self.history_delta = []
        # Neu (Ideen.docx 1.3): Exploration-Verlauf fuer Diagnose-Plot.
        # Wir speichern pro Sampling-Punkt die Explorationswahrscheinlichkeit
        # epsilon und den kumulativen Anteil exploratorischer Aktionen,
        # damit die Phasen Exploration -> Uebergang -> Exploitation
        # spaeter sauber visualisiert werden koennen.
        self.history_epsilon = []
        self.history_explore_share = []

        self._init_firm_vars()
        self._build_ui()
        self._refresh_firm_params()

    def _init_firm_vars(self):
        self.n_firms_var = tk.IntVar(value=2)
        self.a_vars = [tk.DoubleVar(value=2.0) for _ in range(self.MAX_FIRMS)]
        self.c_vars = [tk.DoubleVar(value=1.0) for _ in range(self.MAX_FIRMS)]

    def _build_ui(self):
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=4)
        main.add(left, weight=0)
        main.add(right, weight=1)

        # ---- Modellparameter ----
        params_frame = ttk.LabelFrame(left, text="Modellparameter", padding=8)
        params_frame.pack(fill=tk.X, pady=4)

        self.m_var = tk.IntVar(value=15)
        self.alpha_var = tk.DoubleVar(value=0.15)
        self.beta_var = tk.DoubleVar(value=4e-6)
        self.delta_var = tk.DoubleVar(value=0.95)
        self.mu_var = tk.DoubleVar(value=0.25)
        self.a0_var = tk.DoubleVar(value=0.0)
        self.xi_var = tk.DoubleVar(value=0.1)
        self.episodes_var = tk.IntVar(value=500000)
        # Neu (Ideen.docx 1.1): Q-Initialisierungsstrategie als Combobox
        self.init_strategy_var = tk.StringVar(value="best_response")

        def add_row(r, label, var, width=12):
            ttk.Label(params_frame, text=label).grid(row=r, column=0, sticky=tk.W, pady=1)
            ttk.Entry(params_frame, textvariable=var, width=width).grid(
                row=r, column=1, sticky=tk.W, padx=4
            )

        ttk.Label(params_frame, text="Anzahl Firmen n:").grid(row=0, column=0, sticky=tk.W, pady=1)
        ttk.Spinbox(
            params_frame, from_=2, to=self.MAX_FIRMS, textvariable=self.n_firms_var,
            width=10, command=self._refresh_firm_params,
        ).grid(row=0, column=1, sticky=tk.W, padx=4)

        add_row(1, "Preispunkte m:", self.m_var)
        add_row(2, "Lernrate \u03b1:", self.alpha_var)
        add_row(3, "Explorationsrate \u03b2:", self.beta_var)
        add_row(4, "Diskontfaktor \u03b4:", self.delta_var)
        add_row(5, "Differenzierung \u03bc:", self.mu_var)
        add_row(6, "Outside Option a\u2080:", self.a0_var)
        add_row(7, "Preisspanne \u03be:", self.xi_var)
        add_row(8, "Iterationen:", self.episodes_var)

        # Q-Initialisierung als Combobox (siehe Ideen.docx 1.1)
        ttk.Label(params_frame, text="Q-Init-Strategie:").grid(
            row=9, column=0, sticky=tk.W, pady=1)
        ttk.Combobox(
            params_frame, textvariable=self.init_strategy_var,
            values=list(QLearningSimulation.INIT_STRATEGIES),
            state="readonly", width=14,
        ).grid(row=9, column=1, sticky=tk.W, padx=4)

        # Hilfsbutton: Iterationen anhand der Q-Tabellengroesse vorschlagen
        # (siehe Ideen.docx 1.4 \u2014 gleicher Erwartungswert an Updates pro Eintrag,
        # damit Replikationen ueber verschiedene n vergleichbar sind).
        ttk.Button(
            params_frame, text="Iterationen vorschlagen (\u22483 \u00d7 m^n)",
            command=self._suggest_episodes,
        ).grid(row=10, column=0, columnspan=2, sticky=tk.EW, pady=(4, 0))

        # ---- Firmenspezifisch ----
        self.firm_frame = ttk.LabelFrame(
            left, text="Firmenspezifisch (Asymmetrie)", padding=8
        )
        self.firm_frame.pack(fill=tk.X, pady=4)

        # ---- Steuerung ----
        ctrl = ttk.LabelFrame(left, text="Steuerung", padding=8)
        ctrl.pack(fill=tk.X, pady=4)

        self.start_btn = ttk.Button(ctrl, text="Start", command=self.start_simulation)
        self.start_btn.pack(fill=tk.X, pady=1)
        self.stop_btn = ttk.Button(
            ctrl, text="Stop", command=self.stop_simulation, state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=1)
        self.reset_btn = ttk.Button(ctrl, text="Reset", command=self.reset_simulation)
        self.reset_btn.pack(fill=tk.X, pady=1)
        self.impulse_btn = ttk.Button(
            ctrl, text="Impulse Response", command=self.impulse_response, state=tk.DISABLED
        )
        self.impulse_btn.pack(fill=tk.X, pady=1)

        # Neu: Diagnose-Fenster nach Trainingsende (siehe Ideen.docx 1.3 und 1.4).
        # Zeigt epsilon-Verlauf, Visits-Heatmap, Aktionsentropie und die kumulative
        # Exploration vs Exploitation-Zaehlung. Die Auswertungen helfen, Calvanos
        # Spezifikation kritisch zu wuerdigen.
        self.diag_btn = ttk.Button(
            ctrl, text="Diagnose-Plots...", command=self.show_diagnostics,
            state=tk.DISABLED,
        )
        self.diag_btn.pack(fill=tk.X, pady=1)
        # Neu: Q-Tabellen-Export aus dem Single-Run (CSV pro Firma, Ideen.docx 1.2).
        self.export_q_btn = ttk.Button(
            ctrl, text="Q-Tabellen als CSV speichern...",
            command=self.export_single_q_tables, state=tk.DISABLED,
        )
        self.export_q_btn.pack(fill=tk.X, pady=1)

        # Trennlinie zwischen Single-Run-Aktionen und Batch-Modus
        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        self.batch_btn = ttk.Button(
            ctrl, text="Batch Run (mehrere Seeds)...", command=self.open_batch_window
        )
        self.batch_btn.pack(fill=tk.X, pady=1)
        # Vergleichsfenster: zwei Konfigurationen gleichzeitig laufen lassen
        # und Side-by-side auswerten. Nutzt fuer die Bachelorarbeit z. B. um
        # zwei μ-Werte oder zwei Spielerzahlen direkt gegenueberzustellen.
        self.compare_btn = ttk.Button(
            ctrl, text="Vergleichsfenster (A vs. B)...", command=self.open_compare_window
        )
        self.compare_btn.pack(fill=tk.X, pady=1)

        # ---- Status ----
        status_frame = ttk.LabelFrame(left, text="Status", padding=8)
        status_frame.pack(fill=tk.X, pady=4)
        self.status_label = ttk.Label(
            status_frame, text="Bereit.", wraplength=280, justify=tk.LEFT, font=("Courier", 9)
        )
        self.status_label.pack(fill=tk.X)

        # ---- Plots ----
        plot_frame = ttk.Frame(right)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax_price = self.fig.add_subplot(3, 1, 1)
        self.ax_profit = self.fig.add_subplot(3, 1, 2)
        self.ax_delta = self.fig.add_subplot(3, 1, 3)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()

        # ---- Konsole ----
        cons = ttk.LabelFrame(
            right, text="Konsole — Befehle: /setseed <int>, /getseed, /help",
            padding=4,
        )
        cons.pack(fill=tk.X, pady=4)
        self.console_output = scrolledtext.ScrolledText(cons, height=8, font=("Courier", 9))
        self.console_output.pack(fill=tk.X)
        self.console_output.configure(state=tk.DISABLED)
        inp = ttk.Frame(cons)
        inp.pack(fill=tk.X, pady=2)
        ttk.Label(inp, text=">").pack(side=tk.LEFT)
        self.console_input = ttk.Entry(inp)
        self.console_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.console_input.bind("<Return>", self._handle_command)

        self._print(f"Willkommen. Aktueller Seed = {self.seed}. Mit /help werden Befehle angezeigt.")

    def _refresh_firm_params(self):
        for w in self.firm_frame.winfo_children():
            w.destroy()
        n = max(2, min(self.MAX_FIRMS, int(self.n_firms_var.get())))
        ttk.Label(self.firm_frame, text="Firma").grid(row=0, column=0, padx=4)
        ttk.Label(self.firm_frame, text="a\u1d62 (Qualitaet)").grid(row=0, column=1, padx=4)
        ttk.Label(self.firm_frame, text="c\u1d62 (Grenzkosten)").grid(row=0, column=2, padx=4)
        for i in range(n):
            ttk.Label(self.firm_frame, text=f"{i+1}").grid(row=i + 1, column=0, padx=4)
            ttk.Entry(self.firm_frame, textvariable=self.a_vars[i], width=9).grid(
                row=i + 1, column=1, padx=4, pady=1
            )
            ttk.Entry(self.firm_frame, textvariable=self.c_vars[i], width=9).grid(
                row=i + 1, column=2, padx=4, pady=1
            )

    # ---- Konsole ----
    def _print(self, msg):
        self.console_output.configure(state=tk.NORMAL)
        self.console_output.insert(tk.END, msg + "\n")
        self.console_output.see(tk.END)
        self.console_output.configure(state=tk.DISABLED)

    def _handle_command(self, event):
        cmd = self.console_input.get().strip()
        self.console_input.delete(0, tk.END)
        if not cmd:
            return
        self._print("> " + cmd)
        parts = cmd.split()
        head = parts[0].lower()
        if head == "/setseed" and len(parts) == 2:
            try:
                self.seed = int(parts[1])
                self.rng = np.random.default_rng(self.seed)
                self._print(f"Seed gesetzt auf {self.seed}. Wirksam ab naechstem Start/Reset.")
            except ValueError:
                self._print("Fehler: Seed muss eine ganze Zahl sein.")
        elif head == "/getseed":
            self._print(f"Aktueller Seed: {self.seed}")
        elif head == "/help":
            self._print("Verfuegbare Befehle:")
            self._print("  /setseed <int>  — Seed setzen (wirkt ab naechstem Start)")
            self._print("  /getseed        — aktuellen Seed anzeigen")
            self._print("  /help           — diese Hilfe")
        else:
            self._print(f"Unbekannter Befehl: {parts[0]}")

    # ---- Simulation ----
    def _collect_params(self):
        n = int(self.n_firms_var.get())
        if not (2 <= n <= self.MAX_FIRMS):
            raise ValueError(f"n muss in [2, {self.MAX_FIRMS}] liegen.")
        a = [float(self.a_vars[i].get()) for i in range(n)]
        c = [float(self.c_vars[i].get()) for i in range(n)]
        m = int(self.m_var.get())
        if m < 2:
            raise ValueError("m muss mindestens 2 sein.")
        # Speicherwarnung
        q_entries = n * (m ** n) * m
        if q_entries > 5_000_000:
            if not messagebox.askyesno(
                "Warnung",
                f"Die Q-Tabellen enthalten {q_entries:,} Eintraege "
                f"(~{q_entries * 8 / 1e6:.0f} MB). Fortfahren?",
            ):
                raise RuntimeError("Abgebrochen durch Nutzer.")
        return dict(
            n=n, m=m,
            alpha=float(self.alpha_var.get()),
            beta=float(self.beta_var.get()),
            delta=float(self.delta_var.get()),
            mu=float(self.mu_var.get()),
            a0=float(self.a0_var.get()),
            xi=float(self.xi_var.get()),
            a=a, c=c,
            episodes=int(self.episodes_var.get()),
            init_strategy=str(self.init_strategy_var.get()),
        )

    def _suggest_episodes(self):
        """Setzt das Iterationsfeld auf einen Vorschlagswert (3 · m^n).

        Hintergrund (siehe Ideen.docx, Abschnitt 1.4): Damit jeder Q-Eintrag im
        Erwartungswert die gleiche Anzahl Updates erhaelt, sollten die Iterationen
        proportional zur Q-Tabellengroesse m^n skaliert werden. Mit dem Faktor 3
        erhaelt jeder Eintrag im Mittel rund 3 Updates aus der Exploitation-Phase,
        was fuer eine stabile Q-Schaetzung in Calvanos Baseline empirisch ausreicht.
        Bei sehr kleinem m^n setzen wir mindestens 200000, damit auch das Duopol
        eine sinnvolle Konvergenz erreicht.
        """
        try:
            m = int(self.m_var.get())
            n = int(self.n_firms_var.get())
        except (tk.TclError, ValueError):
            messagebox.showinfo(
                "Hinweis", "Bitte zuerst gueltiges n und m eingeben.")
            return
        suggestion = max(200_000, 3 * (m ** n))
        self.episodes_var.set(suggestion)
        self._print(
            f"Vorschlag: {suggestion:,} Iterationen (= max(200k, 3 · m^n) "
            f"mit m={m}, n={n}).")

    def start_simulation(self):
        if self.running:
            return
        try:
            p = self._collect_params()
            env = CalvanoEnvironment(n=p["n"], a=p["a"], a0=p["a0"], c=p["c"], mu=p["mu"])
            self.rng = np.random.default_rng(self.seed)
            # Aktionsentropie nur fuer den Single-Run aktivieren, im Batch
            # waere der zusaetzliche Speicherbedarf zu gross.
            self.simulation = QLearningSimulation(
                env=env, m=p["m"], alpha=p["alpha"], beta=p["beta"],
                delta=p["delta"], xi=p["xi"], rng=self.rng,
                init_strategy=p["init_strategy"],
                track_entropy=True,
                entropy_window=min(2000, max(500, p["episodes"] // 100)),
            )
            self.n_episodes_target = p["episodes"]

            self._print(f"Simulation gestartet mit Seed={self.seed}.")
            self._print(f"  Q-Init-Strategie: {p['init_strategy']}")
            self._print(f"  Nash-Preise:     {np.round(self.simulation.p_nash, 4).tolist()}")
            self._print(f"  Monopol-Preise:  {np.round(self.simulation.p_mono, 4).tolist()}")
            self._print(f"  Aktionsraum:     [{self.simulation.p_min:.4f}, {self.simulation.p_max:.4f}]")
            self._print(f"  Q-Tabellen:      {p['n']} x {self.simulation.n_states} x {p['m']}")

            self.history_t = []
            self.history_prices = [[] for _ in range(p["n"])]
            self.history_profits = [[] for _ in range(p["n"])]
            self.history_delta = []
            self.history_epsilon = []
            self.history_explore_share = []
            # Konvergenz-Alert-Flag fuer diesen Lauf zuruecksetzen
            self._alerted_converged = False

            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.impulse_btn.config(state=tk.DISABLED)
            self.diag_btn.config(state=tk.DISABLED)
            self.export_q_btn.config(state=tk.DISABLED)

            self.worker_thread = threading.Thread(target=self._training_loop, daemon=True)
            self.worker_thread.start()
        except Exception as e:
            messagebox.showerror("Fehler beim Start", str(e))
            self._print(f"FEHLER: {e}")

    def stop_simulation(self):
        if self.running:
            self.running = False
            self._print("Stopp angefordert...")

    def reset_simulation(self):
        self.running = False
        self.simulation = None
        self.history_t = []
        self.history_prices = []
        self.history_profits = []
        self.history_delta = []
        self.history_epsilon = []
        self.history_explore_share = []
        for ax in (self.ax_price, self.ax_profit, self.ax_delta):
            ax.clear()
        self.canvas.draw()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.impulse_btn.config(state=tk.DISABLED)
        self.diag_btn.config(state=tk.DISABLED)
        self.export_q_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Bereit.")
        self._print("Simulation zurueckgesetzt.")

    def _training_loop(self):
        sim = self.simulation
        n = sim.n
        target = self.n_episodes_target
        log_interval = max(1, target // 500)
        last_gui = time.time()

        try:
            while self.running and sim.t < target:
                prices, profits, eps = sim.step()

                if sim.t % log_interval == 0 or sim.t == target:
                    delta_val = sim.collusion_index(profits)
                    self.history_t.append(sim.t)
                    for i in range(n):
                        self.history_prices[i].append(float(prices[i]))
                        self.history_profits[i].append(float(profits[i]))
                    self.history_delta.append(delta_val)
                    # epsilon und kumulativer Explorationsanteil mitloggen
                    # (siehe Ideen.docx 1.3, fuer Phasenvisualisierung).
                    self.history_epsilon.append(float(eps))
                    self.history_explore_share.append(sim.explore_share())

                if time.time() - last_gui > 0.5:
                    cur_eps = eps
                    cur_delta = sim.collusion_index(profits)
                    cur_t = sim.t
                    # Konvergenzstatus mitlesen (siehe Hinweise: dem Nutzer
                    # mitteilen, ob und wann der Lauf stabil ist).
                    conv_status, conv_stable, conv_thr = sim.convergence_status()
                    # Einmaliger Konsolen-Alert beim ersten Erreichen der Konvergenz
                    if conv_status == 'konvergiert' and not getattr(self, '_alerted_converged', False):
                        self._alerted_converged = True
                        self.root.after(0, lambda t=cur_t: self._print(
                            f"[Konvergenz] greedy-Politik seit {conv_thr:,} Perioden stabil (bei t = {t:,})."))
                    self.root.after(0, self._update_plots)
                    self.root.after(
                        0,
                        lambda t=cur_t, e=cur_eps, d=cur_delta,
                               cs=conv_status, ps=conv_stable, ct=conv_thr:
                            self._update_status(t, e, d, cs, ps, ct),
                    )
                    last_gui = time.time()
        except Exception as e:
            self.root.after(0, lambda err=str(e): self._print(f"FEHLER im Training: {err}"))
        finally:
            self.root.after(0, self._update_plots)
            done = sim.t >= target
            msg = (
                f"Training abgeschlossen nach {sim.t:,} Iterationen."
                if done
                else f"Training gestoppt bei {sim.t:,} Iterationen."
            )
            self.root.after(0, lambda m=msg: self._print(m))
            if self.history_delta:
                final_delta = self.history_delta[-1]
                self.root.after(0, lambda d=final_delta:
                                self._print(f"Finaler Kollusionsindex: Delta = {d:.4f}"))
            self.running = False
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.impulse_btn.config(state=tk.NORMAL))
            # Diagnose- und CSV-Export erst nach Trainingsende verfuegbar
            self.root.after(0, lambda: self.diag_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.export_q_btn.config(state=tk.NORMAL))

    def _update_status(self, t, eps, delta, conv_status='lernt',
                       periods_stable=0, conv_thr=100_000):
        # Konvergenz-Indikator als ASCII-Symbol vor dem Status
        indicator = {
            'lernt': '...',           # noch im Lernprozess
            'stabilisiert': '~',      # auf dem Weg zur Konvergenz
            'konvergiert': '+',       # greedy-Politik stabil ueber Schwellenwert
        }.get(conv_status, '?')
        self.status_label.config(
            text=(
                f"t     = {t:>10,}\n"
                f"\u03b5     = {eps:.4e}\n"
                f"\u0394     = {delta:>+.4f}\n"
                f"Seed  = {self.seed}\n"
                f"Konv. = {indicator} {conv_status} "
                f"({periods_stable:>8,} / {conv_thr:,} P. stabil)"
            )
        )

    def _update_plots(self):
        if not self.history_t or self.simulation is None:
            return
        sim = self.simulation
        n = sim.n

        self.ax_price.clear()
        self.ax_profit.clear()
        self.ax_delta.clear()

        colors = matplotlib.cm.tab10(np.linspace(0, 1, max(n, 2)))
        for i in range(n):
            self.ax_price.plot(
                self.history_t, self.history_prices[i],
                label=f"Firma {i+1}", linewidth=0.7, color=colors[i],
            )
            self.ax_profit.plot(
                self.history_t, self.history_profits[i],
                label=f"Firma {i+1}", linewidth=0.7, color=colors[i],
            )

        for i in range(n):
            self.ax_price.axhline(
                sim.p_nash[i], color=colors[i], linestyle=":", alpha=0.5, linewidth=0.8,
            )
            self.ax_price.axhline(
                sim.p_mono[i], color=colors[i], linestyle="--", alpha=0.5, linewidth=0.8,
            )

        self.ax_delta.plot(
            self.history_t, self.history_delta, color="purple", linewidth=0.8,
        )
        self.ax_delta.axhline(0, color="red", linestyle="--", alpha=0.5, label="Nash (0)")
        self.ax_delta.axhline(1, color="green", linestyle="--", alpha=0.5, label="Monopol (1)")

        self.ax_price.set_ylabel("Preis")
        self.ax_price.set_title(
            "Preisentwicklung (gepunktet: Nash, gestrichelt: Monopol)"
        )
        self.ax_price.legend(fontsize=7, loc="upper right", ncol=min(n, 4))
        self.ax_price.grid(True, alpha=0.3)

        self.ax_profit.set_ylabel("Gewinn \u03c0")
        self.ax_profit.set_title("Gewinnentwicklung")
        self.ax_profit.legend(fontsize=7, loc="upper right", ncol=min(n, 4))
        self.ax_profit.grid(True, alpha=0.3)

        self.ax_delta.set_ylabel("Kollusionsindex \u0394")
        self.ax_delta.set_xlabel("Iteration t")
        self.ax_delta.set_title("Kollusionsindex (0 = Nash, 1 = Monopol)")
        self.ax_delta.legend(fontsize=7, loc="upper right")
        self.ax_delta.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ---- Impulse Response ----
    def impulse_response(self):
        if self.simulation is None or self.running:
            messagebox.showinfo("Hinweis", "Bitte zuerst Training abschliessen.")
            return
        sim = self.simulation
        pre, post = 20, 30
        trace_p, trace_pi, trace_d = [], [], []

        state = sim.current_state
        # Vor dem Schock: greedy play
        for _ in range(pre):
            actions = np.array([np.argmax(sim.Q[i, state]) for i in range(sim.n)])
            prices = sim.prices[actions]
            profits = sim.env.profits(prices)
            trace_p.append(prices.copy())
            trace_pi.append(profits.copy())
            trace_d.append(sim.collusion_index(profits))
            state = sim._state_from_actions(actions)

        # Schock: Firma 1 weicht auf Nash-Preisniveau ab
        actions = np.array([np.argmax(sim.Q[i, state]) for i in range(sim.n)])
        nash_action = int(np.argmin(np.abs(sim.prices - float(np.mean(sim.p_nash)))))
        actions[0] = nash_action
        prices = sim.prices[actions]
        profits = sim.env.profits(prices)
        trace_p.append(prices.copy())
        trace_pi.append(profits.copy())
        trace_d.append(sim.collusion_index(profits))
        state = sim._state_from_actions(actions)

        # Nach Schock: wieder greedy
        for _ in range(post):
            actions = np.array([np.argmax(sim.Q[i, state]) for i in range(sim.n)])
            prices = sim.prices[actions]
            profits = sim.env.profits(prices)
            trace_p.append(prices.copy())
            trace_pi.append(profits.copy())
            trace_d.append(sim.collusion_index(profits))
            state = sim._state_from_actions(actions)

        self._show_impulse_window(pre, trace_p, trace_pi, trace_d, sim)

    def _show_impulse_window(self, pre, trace_p, trace_pi, trace_d, sim):
        win = tk.Toplevel(self.root)
        win.title("Impulse Response Analysis")
        win.geometry("950x700")
        fig = Figure(figsize=(9.5, 7), dpi=100)
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        prices = np.array(trace_p)
        profits = np.array(trace_pi)
        deltas = np.array(trace_d)
        t_axis = np.arange(len(prices)) - pre

        colors = matplotlib.cm.tab10(np.linspace(0, 1, max(sim.n, 2)))
        for i in range(sim.n):
            ax1.plot(t_axis, prices[:, i], marker="o", markersize=3,
                     label=f"Firma {i+1}", color=colors[i])
            ax1.axhline(sim.p_nash[i], color=colors[i], linestyle=":", alpha=0.4)
            ax1.axhline(sim.p_mono[i], color=colors[i], linestyle="--", alpha=0.4)
            ax2.plot(t_axis, profits[:, i], marker="o", markersize=3,
                     label=f"Firma {i+1}", color=colors[i])

        for ax in (ax1, ax2, ax3):
            ax.axvline(0, color="black", linestyle=":", alpha=0.6)
            ax.grid(True, alpha=0.3)

        ax3.plot(t_axis, deltas, color="purple", marker="o", markersize=3)
        ax3.axhline(0, color="red", linestyle="--", alpha=0.4)
        ax3.axhline(1, color="green", linestyle="--", alpha=0.4)

        ax1.set_ylabel("Preis")
        ax1.set_title("Impulse Response: Schock bei t = 0 (Firma 1 weicht auf Nash-Niveau ab)")
        ax1.legend(fontsize=8, loc="best")

        ax2.set_ylabel("Gewinn \u03c0")
        ax2.legend(fontsize=8, loc="best")

        ax3.set_ylabel("\u0394")
        ax3.set_xlabel("Perioden relativ zum Schock")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, win).update()
        canvas.draw()

    # ---- Diagnose-Fenster (Ideen.docx 1.3, 1.4) ----
    def show_diagnostics(self):
        """Oeffnet ein Fenster mit vier Diagnose-Plots zur fertigen Simulation:

          1) Epsilon-Verlauf mit Phasenmarkierung (Exploration, Uebergang, Exploitation)
          2) Kumulative Aktionszaehlung (Exploration vs Exploitation)
          3) Aktionsentropie pro Firma ueber das rollierende Fenster
          4) Visits-Heatmap (Firma 1): wie oft wurde jeder Q-Eintrag aktualisiert

        Diese vier Plots sind die im Betreuergespraech ausdruecklich diskutierte
        Diagnostik und gehoeren spaeter in den Methodenanhang der Bachelorarbeit.
        """
        if self.simulation is None or self.running:
            messagebox.showinfo(
                "Hinweis", "Bitte zuerst Training abschliessen.")
            return
        sim = self.simulation
        win = tk.Toplevel(self.root)
        win.title("Diagnose: Exploration, Entropie, Visits")
        win.geometry("1100x800")
        fig = Figure(figsize=(11, 8), dpi=100)

        # --- 1) Epsilon-Verlauf mit Phasenmarkierung ---
        # epsilon(t) ist deterministisch gemaess exp(-beta*t), die Phasen
        # ergeben sich aus festen Schwellenwerten (0.1 und 0.01).
        ax1 = fig.add_subplot(2, 2, 1)
        if self.history_t:
            t_arr = np.array(self.history_t)
            eps_arr = np.array(self.history_epsilon)
            ax1.semilogy(t_arr, eps_arr, color="navy", linewidth=1.2,
                         label=r"$\varepsilon_t = \exp(-\beta t)$")
            # Phasenbaender markieren
            ax1.axhspan(0.1, 1.0, alpha=0.08, color="orange",
                        label="Exploration ε > 0.1")
            ax1.axhspan(0.01, 0.1, alpha=0.08, color="yellow",
                        label="Uebergang 0.01 < ε < 0.1")
            ax1.axhspan(1e-10, 0.01, alpha=0.08, color="green",
                        label="Exploitation ε < 0.01")
            ax1.set_ylim(max(1e-8, eps_arr.min() / 10), 1.2)
        ax1.set_xlabel("Iteration t")
        ax1.set_ylabel(r"$\varepsilon$ (log-Skala)")
        ax1.set_title("Explorationsrate ueber Zeit")
        ax1.legend(fontsize=7, loc="lower left")
        ax1.grid(True, alpha=0.3, which="both")

        # --- 2) Kumulative Aktionszaehlung ---
        # Direkt aus dem Tracking in sim.explore_count / sim.exploit_count.
        # Wir koennen den Endwert exakt anzeigen und die kumulative Kurve
        # aus history_explore_share rekonstruieren.
        ax2 = fig.add_subplot(2, 2, 2)
        if self.history_t:
            shares = np.array(self.history_explore_share)
            totals = np.array(self.history_t) * sim.n
            # Anzahl exploratorischer Aktionen bis zu Zeitpunkt t
            n_explore = shares * totals
            n_exploit = totals - n_explore
            ax2.plot(t_arr, n_explore, color="orange", linewidth=1.2,
                     label="Exploration (kumulativ)")
            ax2.plot(t_arr, n_exploit, color="green", linewidth=1.2,
                     label="Exploitation (kumulativ)")
        ax2.set_xlabel("Iteration t")
        ax2.set_ylabel("Anzahl Aktionen")
        ax2.set_title(
            f"Kumulative Aktionen — Endwert: {sim.explore_count:,} explor., "
            f"{sim.exploit_count:,} expl.")
        ax2.legend(fontsize=8, loc="best")
        ax2.grid(True, alpha=0.3)

        # --- 3) Aktionsentropie pro Firma ---
        # Berechnen wir nur fuer den aktuellen Stand (Endwert), weil das
        # Tracking ueber Zeit zu teuer waere. H = 0 bedeutet deterministische
        # Politik; H = log(m) ist uniforme Streuung.
        ax3 = fig.add_subplot(2, 2, 3)
        H = sim.action_entropy()
        if not np.any(np.isnan(H)):
            colors = matplotlib.cm.tab10(np.linspace(0, 1, max(sim.n, 2)))
            ax3.bar(range(1, sim.n + 1), H, color=colors[:sim.n],
                    edgecolor="black", linewidth=0.6)
            ax3.axhline(np.log(sim.m), color="red", linestyle="--",
                        alpha=0.5, label=f"log(m) = {np.log(sim.m):.2f}")
            ax3.set_ylim(0, np.log(sim.m) * 1.1)
            ax3.set_xticks(range(1, sim.n + 1))
            ax3.legend(fontsize=8)
        else:
            ax3.text(0.5, 0.5, "Entropie-Tracking nicht aktiviert",
                     ha="center", va="center", transform=ax3.transAxes)
        ax3.set_xlabel("Firma")
        ax3.set_ylabel("Aktionsentropie H")
        ax3.set_title(f"Aktionsentropie (Fenster {sim.entropy_window})")
        ax3.grid(True, alpha=0.3, axis="y")

        # --- 4) Visits-Heatmap Firma 1 ---
        # Zeigt, wie oft jede (State, Action)-Kombination aktualisiert wurde.
        # Bei n=2, m=15 ist die Matrix 225 x 15. Bei groesseren Konfigurationen
        # ist die Anzeige der Zeilen unleserlich; wir reduzieren die Y-Achse dann
        # auf 50 Beispielzeilen (gleichverteilt).
        ax4 = fig.add_subplot(2, 2, 4)
        V = sim.visits[0].astype(float)
        if V.shape[0] > 50:
            sample_idx = np.linspace(0, V.shape[0] - 1, 50).astype(int)
            V_display = V[sample_idx]
            y_label = "State-Index (Stichprobe von 50)"
        else:
            V_display = V
            y_label = "State-Index"
        im = ax4.imshow(V_display, aspect="auto", cmap="viridis", origin="lower")
        ax4.set_xlabel("Aktion (Preisniveau)")
        ax4.set_ylabel(y_label)
        ax4.set_title("Visits-Heatmap Firma 1 (Updates pro Eintrag)")
        fig.colorbar(im, ax=ax4, label="Anzahl Updates")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, win).update()
        canvas.draw()

    # ---- Q-Tabellen-Export aus Single-Run als CSV (Ideen.docx 1.2) ----
    def export_single_q_tables(self):
        """Speichert die Q-Tabellen der aktuell trainierten Simulation als CSV.

        Der Nutzer waehlt ein Verzeichnis; pro Firma wird eine CSV-Datei mit
        Kopfkommentaren (Konfiguration, Seed) und dem Q-Inhalt geschrieben.
        Diese Form ist Excel- und pandas-kompatibel und entspricht der vom
        Betreuer gewuenschten Speicherung.
        """
        if self.simulation is None or self.running:
            messagebox.showinfo(
                "Hinweis", "Bitte zuerst Training abschliessen.")
            return
        from tkinter import filedialog
        out_dir = filedialog.askdirectory(
            parent=self.root, title="Verzeichnis fuer Q-Tabellen-CSV waehlen")
        if not out_dir:
            return
        sim = self.simulation
        p = self._collect_params()
        cfg = BatchConfig(
            n=p["n"], m=p["m"], alpha=p["alpha"], beta=p["beta"],
            delta=p["delta"], mu=p["mu"], a0=p["a0"], xi=p["xi"],
            a=list(p["a"]), c=list(p["c"]),
            episodes=int(self.n_episodes_target or p["episodes"]),
            seeds=[self.seed], avg_window=1000,
            init_strategy=p["init_strategy"],
        )
        # Letzte erlernten Preise als Mittelwert ueber die zuletzt geloggte
        # Phase ermitteln (fuer den Header der CSV).
        n_log = len(self.history_prices[0]) if self.history_prices else 0
        win_size = min(200, n_log)
        final_prices = []
        for i in range(sim.n):
            arr = self.history_prices[i][-win_size:] if win_size > 0 else []
            final_prices.append(float(np.mean(arr)) if arr else float("nan"))
        rr = RunResult(
            seed=self.seed,
            final_prices=final_prices,
            final_profits=[],
            final_delta=float(self.history_delta[-1])
                          if self.history_delta else float("nan"),
            p_nash=sim.p_nash.tolist(),
            p_mono=sim.p_mono.tolist(),
            converged=False, elapsed_s=0.0,
            q_table=sim.Q.copy(),
            action_prices=sim.prices.copy(),
            explore_count=int(sim.explore_count),
            exploit_count=int(sim.exploit_count),
            explore_per_firm=sim.explore_per_firm.tolist(),
            exploit_per_firm=sim.exploit_per_firm.tolist(),
        )
        try:
            paths = export_q_tables_csv(cfg, [rr], out_dir)
        except Exception as e:
            messagebox.showerror("Export-Fehler", str(e))
            return
        self._print(f"{len(paths)} CSV-Datei(en) geschrieben nach {out_dir}.")
        messagebox.showinfo(
            "Export erfolgreich",
            f"{len(paths)} Q-Tabellen wurden gespeichert in:\n{out_dir}")

    def open_batch_window(self):
        """Oeffnet das Batch-Konfigurations-Fenster, vorbefuellt mit den
        aktuellen Parametern aus der Haupt-GUI."""
        try:
            params = self._collect_params()
        except Exception as e:
            messagebox.showerror(
                "Parameter-Fehler",
                f"Bitte zuerst gueltige Parameter im Hauptfenster setzen.\n\n{e}",
            )
            return
        BatchWindow(self.root, params)

    def open_compare_window(self):
        """Oeffnet das Vergleichsfenster mit zwei parallelen Konfigurationen.

        Beide Konfigurationen werden mit den aktuellen Parametern aus der
        Haupt-GUI vorbefuellt; im Fenster selbst koennen einzelne Werte fuer
        A oder B angepasst und beide Laeufe gleichzeitig gestartet werden.
        Typischer Einsatzfall: zwei μ-Werte oder zwei Spielerzahlen direkt
        gegenueberstellen, ohne separate Single-Runs.
        """
        try:
            params = self._collect_params()
        except Exception as e:
            messagebox.showerror(
                "Parameter-Fehler",
                f"Bitte zuerst gueltige Parameter im Hauptfenster setzen.\n\n{e}",
            )
            return
        CompareWindow(self.root, params)


# ==========================================================
# BATCH-FENSTER
# ==========================================================

class BatchWindow:
    """Eigenes Fenster zur Konfiguration und Ausfuehrung von Batch-Laeufen.

    Vergleicht eine Konfiguration ueber viele Seeds, sammelt Delta-Werte und
    erlaubt den Export als CSV plus optional Q-Tabellen und Visits-Heatmap.
    Diese Erweiterungen entsprechen den Punkten 1.2 und 1.4 aus Ideen.docx:
    Q-Tabellen als CSV speichern, Visits-Heatmap als Diagnostik, Q-Init-Strategie
    pro Batch waehlbar.
    """

    def __init__(self, parent, base_params: dict):
        self.parent = parent
        self.base_params = base_params  # n, m, alpha, beta, delta, mu, a0, xi, a, c, episodes
        self.win = tk.Toplevel(parent)
        self.win.title("Batch Run — Replikationen ueber mehrere Seeds")
        self.win.geometry("1100x800")

        self.runner: Optional[BatchRunner] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.results: List[RunResult] = []
        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.win, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # Linke Seite: Konfiguration
        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        # Rechte Seite: Ergebnisse
        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- Aktuelle Konfiguration anzeigen (read-only) ----
        cfg_frame = ttk.LabelFrame(left, text="Konfiguration (aus Hauptfenster)",
                                    padding=8)
        cfg_frame.pack(fill=tk.X, pady=4)
        p = self.base_params
        info = (
            f"n = {p['n']}    m = {p['m']}\n"
            f"α = {p['alpha']}   β = {p['beta']}\n"
            f"δ = {p['delta']}   μ = {p['mu']}\n"
            f"a₀ = {p['a0']}   ξ = {p['xi']}\n"
            f"a = {[round(x, 3) for x in p['a']]}\n"
            f"c = {[round(x, 3) for x in p['c']]}\n"
            f"Init-Strategie: {p.get('init_strategy', 'best_response')}\n"
            f"Iterationen pro Lauf: {p['episodes']:,}"
        )
        ttk.Label(cfg_frame, text=info, font=("Courier", 9), justify=tk.LEFT).pack(
            anchor=tk.W
        )

        # ---- Batch-Parameter ----
        batch_frame = ttk.LabelFrame(left, text="Batch-Parameter", padding=8)
        batch_frame.pack(fill=tk.X, pady=4)

        # Variablen fuer die Batch-Konfiguration
        self.n_reps_var = tk.IntVar(value=20)
        self.first_seed_var = tk.IntVar(value=1)
        self.avg_window_var = tk.IntVar(value=1000)
        self.n_workers_var = tk.IntVar(value=max(1, (os.cpu_count() or 2) - 1))
        self.save_q_var = tk.BooleanVar(value=False)
        # Neu (Ideen.docx 1.1, 1.2, 1.4): pro Batch eine einheitliche
        # Initialisierungsstrategie, ein Q-Tabellen-Format und optional
        # die Visits-Heatmap.
        self.init_strategy_var = tk.StringVar(
            value=self.base_params.get("init_strategy", "best_response"))
        self.save_visits_var = tk.BooleanVar(value=False)
        self.q_format_var = tk.StringVar(value="npz")  # "npz" oder "csv"

        def add_row(parent, r, label, var, width=10):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky=tk.W, pady=1)
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=r, column=1, sticky=tk.W, padx=4, pady=1
            )

        add_row(batch_frame, 0, "Anzahl Replikationen:", self.n_reps_var)
        add_row(batch_frame, 1, "Erster Seed:", self.first_seed_var)
        add_row(batch_frame, 2, "Mittelungsfenster:", self.avg_window_var)
        add_row(batch_frame, 3, f"Worker-Prozesse (max {os.cpu_count() or 1}):",
                self.n_workers_var)

        ttk.Label(
            batch_frame,
            text="Seeds: erster, erster+1, ..., erster+n-1\n"
                 "Mittelungsfenster: Δ wird ueber\n"
                 "die letzten N Iterationen jedes Laufs gemittelt.",
            font=("Helvetica", 8), foreground="gray", justify=tk.LEFT,
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        # Q-Init-Strategie pro Batch (methodisch wichtig: alle Replikationen
        # eines Batches verwenden die gleiche Initialisierung, sonst sind die
        # Delta-Werte nicht vergleichbar).
        ttk.Label(batch_frame, text="Q-Init-Strategie:").grid(
            row=5, column=0, sticky=tk.W, pady=(8, 1))
        ttk.Combobox(
            batch_frame, textvariable=self.init_strategy_var,
            values=list(QLearningSimulation.INIT_STRATEGIES),
            state="readonly", width=14,
        ).grid(row=5, column=1, sticky=tk.W, padx=4, pady=(8, 1))

        # Q-Tabellen-Export-Option
        self.save_q_check = ttk.Checkbutton(
            batch_frame, text="Q-Tabellen mit speichern",
            variable=self.save_q_var, command=self._on_save_q_toggled,
        )
        self.save_q_check.grid(row=6, column=0, columnspan=2, sticky=tk.W,
                               pady=(8, 0))
        # Format-Wahl: NPZ ist kompakt, CSV ist Excel-lesbar.
        # Der Betreuer bevorzugt CSV (siehe Ideen.docx 1.2).
        fmt_frame = ttk.Frame(batch_frame)
        fmt_frame.grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=(18, 0))
        ttk.Radiobutton(
            fmt_frame, text="als .npz (kompakt)", variable=self.q_format_var,
            value="npz",
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            fmt_frame, text="als .csv pro Firma", variable=self.q_format_var,
            value="csv",
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Visits-Heatmap zusaetzlich speichern (Ideen.docx 1.4): zeigt im
        # Anhang, wie oft jeder (state, action)-Eintrag aktualisiert wurde.
        self.save_visits_check = ttk.Checkbutton(
            batch_frame, text="Visits-Heatmap mit speichern",
            variable=self.save_visits_var,
        )
        self.save_visits_check.grid(row=8, column=0, columnspan=2, sticky=tk.W,
                                     pady=(4, 0))

        self.q_size_label = ttk.Label(
            batch_frame, text="", font=("Helvetica", 8), foreground="gray",
        )
        self.q_size_label.grid(row=9, column=0, columnspan=2, sticky=tk.W)

        # ---- Steuerung ----
        ctrl_frame = ttk.LabelFrame(left, text="Steuerung", padding=8)
        ctrl_frame.pack(fill=tk.X, pady=4)

        self.start_batch_btn = ttk.Button(ctrl_frame, text="Batch starten",
                                           command=self.start_batch)
        self.start_batch_btn.pack(fill=tk.X, pady=1)

        self.cancel_batch_btn = ttk.Button(ctrl_frame, text="Abbrechen",
                                            command=self.cancel_batch,
                                            state=tk.DISABLED)
        self.cancel_batch_btn.pack(fill=tk.X, pady=1)

        self.export_btn = ttk.Button(ctrl_frame, text="CSV exportieren...",
                                      command=self.export_csv,
                                      state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X, pady=1)

        # ---- Status ----
        status_frame = ttk.LabelFrame(left, text="Status", padding=8)
        status_frame.pack(fill=tk.X, pady=4)
        self.status_label = ttk.Label(
            status_frame, text="Bereit. Klick 'Batch starten' zum Loslegen.",
            wraplength=300, justify=tk.LEFT, font=("Courier", 9),
        )
        self.status_label.pack(fill=tk.X)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100, length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=4)

        self.eta_label = ttk.Label(
            status_frame, text="", font=("Courier", 8), foreground="gray",
        )
        self.eta_label.pack(fill=tk.X)

        # ---- Aktive Replikationen (Live-Anzeige) ----
        active_frame = ttk.LabelFrame(left, text="Aktive Laeufe", padding=4)
        active_frame.pack(fill=tk.BOTH, expand=False, pady=4)

        active_cols = ("seed", "fortschritt", "delta", "eps")
        self.active_tree = ttk.Treeview(
            active_frame, columns=active_cols, show="headings", height=6,
        )
        for c, t, w in [
            ("seed", "Seed", 50),
            ("fortschritt", "Fortschritt", 130),
            ("delta", "Δ (live)", 70),
            ("eps", "ε", 60),
        ]:
            self.active_tree.heading(c, text=t)
            self.active_tree.column(c, width=w, anchor=tk.W)
        self.active_tree.pack(fill=tk.X)

        # Tracking: pro Seed, was haben wir gerade
        self._active_runs: dict = {}
        self._batch_start_time: Optional[float] = None

        # ---- Ergebnisse: Zusammenfassung + Plots ----
        summary_frame = ttk.LabelFrame(right, text="Zusammenfassung", padding=8)
        summary_frame.pack(fill=tk.X, pady=4)
        self.summary_label = ttk.Label(
            summary_frame, text="(noch keine Ergebnisse)",
            font=("Courier", 10), justify=tk.LEFT,
        )
        self.summary_label.pack(anchor=tk.W)

        # Plot-Bereich
        plot_frame = ttk.LabelFrame(right, text="Verteilung Kollusionsindex Δ",
                                     padding=4)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_hist = self.fig.add_subplot(2, 1, 1)
        self.ax_seeds = self.fig.add_subplot(2, 1, 2)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tabelle der Einzelergebnisse
        table_frame = ttk.LabelFrame(right, text="Einzellaeufe", padding=4)
        table_frame.pack(fill=tk.X, pady=4)

        cols = ("seed", "delta", "konvergiert", "preise", "zeit")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                  height=6)
        for c, t, w in [
            ("seed", "Seed", 60),
            ("delta", "Δ", 80),
            ("konvergiert", "Konvergiert", 90),
            ("preise", "Finale Preise", 320),
            ("zeit", "Zeit (s)", 70),
        ]:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=tk.W)
        sb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL,
                            command=self.tree.yview)
        self.tree.configure(yscroll=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_save_q_toggled(self):
        """Wenn die Q-Tabellen-Checkbox aktiviert wird, schaetzen wir die
        zu erwartende Dateigroesse und warnen ggf. den Anwender."""
        if not self.save_q_var.get():
            self.q_size_label.config(text="")
            return
        try:
            n_reps = int(self.n_reps_var.get())
        except (ValueError, tk.TclError):
            n_reps = len(self.base_params.get("seeds", [])) or 20
        size_bytes = (
            self.base_params["n"]
            * (self.base_params["m"] ** self.base_params["n"])
            * self.base_params["m"]
            * 8
            * max(1, n_reps)
        )
        size_mb = size_bytes / (1024 * 1024)
        self.q_size_label.config(
            text=f"Geschaetzte Groesse (vor Komprimierung): {size_mb:.1f} MB",
            foreground="firebrick" if size_mb > 100 else "gray",
        )
        if size_mb > 100:
            keep = messagebox.askyesno(
                "Speicher-Warnung",
                f"Die Q-Tabellen werden geschaetzt {size_mb:.0f} MB belegen "
                f"(vor Komprimierung).\n\n"
                f"Konfiguration: n={self.base_params['n']}, "
                f"m={self.base_params['m']}, {n_reps} Replikationen.\n\n"
                "Komprimiertes .npz wird kleiner sein (typisch 2-10x), "
                "aber je nach Konfiguration trotzdem mehrere hundert MB.\n\n"
                "Q-Tabellen wirklich speichern?",
                parent=self.win,
            )
            if not keep:
                self.save_q_var.set(False)
                self.q_size_label.config(text="")

    def start_batch(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        try:
            n_reps = int(self.n_reps_var.get())
            first_seed = int(self.first_seed_var.get())
            avg_w = int(self.avg_window_var.get())
            n_workers = int(self.n_workers_var.get())
            if n_reps < 1:
                raise ValueError("Anzahl Replikationen muss mindestens 1 sein.")
            if n_workers < 1:
                raise ValueError("Anzahl Worker muss mindestens 1 sein.")
        except Exception as e:
            messagebox.showerror("Parameter-Fehler", str(e))
            return

        seeds = list(range(first_seed, first_seed + n_reps))
        p = self.base_params
        # Visits werden gespeichert, wenn die Checkbox aktiv ist. Q-Tabellen
        # werden in diesem Fall ebenfalls mitgenommen, weil Visits ohne den
        # Kontext der zugehoerigen Q-Werte schwer interpretierbar sind.
        save_visits = bool(self.save_visits_var.get())
        cfg = BatchConfig(
            n=p["n"], m=p["m"], alpha=p["alpha"], beta=p["beta"],
            delta=p["delta"], mu=p["mu"], a0=p["a0"], xi=p["xi"],
            a=list(p["a"]), c=list(p["c"]),
            episodes=p["episodes"], seeds=seeds, avg_window=avg_w,
            save_q_tables=bool(self.save_q_var.get()) or save_visits,
            init_strategy=str(self.init_strategy_var.get()),
            save_visits=save_visits,
        )
        self.runner = BatchRunner(cfg, n_workers=n_workers)
        self.results = []

        # UI-Status
        self.start_batch_btn.config(state=tk.DISABLED)
        self.cancel_batch_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.DISABLED)
        self.tree.delete(*self.tree.get_children())
        self.active_tree.delete(*self.active_tree.get_children())
        self._active_runs = {}
        self._batch_start_time = time.time()
        self.progress_var.set(0.0)
        self.eta_label.config(text="")
        self._set_status(f"Batch gestartet: {n_reps} Replikationen "
                         f"auf {n_workers} Prozessen ...")

        # Worker-Thread, der den BatchRunner ausfuehrt
        self.worker_thread = threading.Thread(
            target=self._run_in_thread, args=(cfg,), daemon=True
        )
        self.worker_thread.start()

    def _run_in_thread(self, cfg: BatchConfig):
        def on_progress(res, done, total):
            self.win.after(0, lambda r=res, d=done, t=total:
                            self._on_progress(r, d, t))

        def on_done(results, cancelled):
            self.win.after(0, lambda rs=results, c=cancelled:
                            self._on_done(rs, c))

        def on_live(seed, kind, info):
            self.win.after(0, lambda s=seed, k=kind, i=info:
                            self._on_live_update(s, k, i))

        try:
            self.runner.run(on_progress=on_progress, on_done=on_done,
                            on_live_update=on_live)
        except Exception as e:
            self.win.after(0, lambda err=str(e):
                            self._set_status(f"FEHLER: {err}"))

    def _on_live_update(self, seed: int, kind: str, info: dict):
        """Verarbeite Live-Update aus einem Worker."""
        if kind == "started":
            self._active_runs[seed] = {
                "t": 0, "total": info.get("episodes", 1),
                "delta": float("nan"), "eps": 1.0,
                "status": "lauft",
            }
        elif kind == "progress":
            if seed in self._active_runs:
                self._active_runs[seed].update({
                    "t": info.get("t", 0),
                    "delta": info.get("delta", float("nan")),
                    "eps": info.get("eps", 0.0),
                })
        elif kind == "done":
            self._active_runs.pop(seed, None)
        elif kind == "error":
            self._active_runs.pop(seed, None)

        self._refresh_active_runs()
        self._refresh_eta()

    def _refresh_active_runs(self):
        """Aktualisiere die Tabelle der aktiven Laeufe."""
        self.active_tree.delete(*self.active_tree.get_children())
        for seed in sorted(self._active_runs.keys()):
            info = self._active_runs[seed]
            t = info["t"]
            total = info["total"]
            pct = 100.0 * t / total if total > 0 else 0.0
            fortschritt = f"{t:>7,} / {total:,} ({pct:.0f}%)"
            delta_str = (f"{info['delta']:+.3f}"
                         if not np.isnan(info["delta"]) else "—")
            eps_str = f"{info['eps']:.3f}" if info["eps"] > 0 else "—"
            self.active_tree.insert("", tk.END, values=(
                seed, fortschritt, delta_str, eps_str,
            ))

    def _refresh_eta(self):
        """Schaetze die verbleibende Wallzeit basierend auf bisherigem Verlauf."""
        if self._batch_start_time is None:
            return
        elapsed = time.time() - self._batch_start_time
        n_done = len(self.results)
        n_total = len(self.runner.cfg.seeds) if self.runner else 0

        partial = 0.0
        for info in self._active_runs.values():
            if info["total"] > 0:
                partial += info["t"] / info["total"]
        effective_done = n_done + partial
        if effective_done < 0.05 or n_total == 0:
            self.eta_label.config(text=f"Verstrichen: {elapsed:.0f}s")
            return
        total_estimated = elapsed * n_total / effective_done
        eta = max(0, total_estimated - elapsed)

        def fmt(s):
            s = int(s)
            if s < 60:
                return f"{s}s"
            if s < 3600:
                return f"{s//60}m {s%60:02d}s"
            return f"{s//3600}h {(s%3600)//60:02d}m"

        self.progress_var.set(100.0 * effective_done / n_total)
        self.eta_label.config(
            text=f"Verstrichen: {fmt(elapsed)}   |   "
                 f"verbleibend (geschaetzt): {fmt(eta)}"
        )

    def _on_progress(self, res: RunResult, done: int, total: int):
        # Tabelle ergaenzen
        prices_str = ", ".join(f"{p:.3f}" for p in res.final_prices) \
                     if res.final_prices else "(Fehler)"
        self.tree.insert("", tk.END, values=(
            res.seed,
            f"{res.final_delta:+.4f}" if not np.isnan(res.final_delta) else "NaN",
            "ja" if res.converged else "nein",
            prices_str,
            f"{res.elapsed_s:.1f}",
        ))
        self.results.append(res)
        self.progress_var.set(100.0 * done / total)
        self._set_status(f"Lauf {done}/{total} fertig ... "
                         f"(Seed {res.seed}: Δ = {res.final_delta:+.4f})")
        self._refresh_plots()

    def _on_done(self, results, cancelled):
        self.start_batch_btn.config(state=tk.NORMAL)
        self.cancel_batch_btn.config(state=tk.DISABLED)
        if results:
            self.export_btn.config(state=tk.NORMAL)
        if cancelled:
            self._set_status(f"Abgebrochen nach {len(results)} Replikationen.")
        else:
            self._set_status(f"Fertig. {len(results)} Replikationen abgeschlossen.")
        self._active_runs = {}
        self.active_tree.delete(*self.active_tree.get_children())
        if self._batch_start_time is not None:
            total = time.time() - self._batch_start_time
            self.eta_label.config(text=f"Gesamtdauer: {int(total)}s")
        self._refresh_summary()
        self._refresh_plots()

    def cancel_batch(self):
        if self.runner:
            self.runner.cancel()
            self._set_status("Abbruch angefordert ...")

    def _set_status(self, msg: str):
        self.status_label.config(text=msg)

    def _refresh_summary(self):
        if not self.results:
            self.summary_label.config(text="(noch keine Ergebnisse)")
            return
        agg = aggregate_results(self.results)
        if agg["n_runs"] == 0:
            self.summary_label.config(
                text=f"Alle {agg['n_failed']} Laeufe fehlgeschlagen."
            )
            return
        txt = (
            f"Erfolgreiche Laeufe: {agg['n_runs']}"
            + (f" (von denen {agg['n_failed']} fehlgeschlagen)"
               if agg['n_failed'] else "")
            + f"\nKonvergiert:         {agg['n_converged']}/{agg['n_runs']}\n"
            f"Δ Mittelwert:        {agg['mean_delta']:+.4f}\n"
            f"Δ Standardabw.:      {agg['std_delta']:.4f}\n"
            f"Δ Median:            {agg['median_delta']:+.4f}\n"
            f"Δ Min / Max:         {agg['min_delta']:+.4f}  /  "
            f"{agg['max_delta']:+.4f}"
        )
        self.summary_label.config(text=txt)

    def _refresh_plots(self):
        ok = [r for r in self.results if not r.error]
        self.ax_hist.clear()
        self.ax_seeds.clear()

        if not ok:
            self.canvas.draw_idle()
            return

        deltas = np.array([r.final_delta for r in ok])
        n_bins = min(20, max(5, len(deltas) // 2))
        self.ax_hist.hist(deltas, bins=n_bins, color="steelblue",
                          edgecolor="white", alpha=0.85)
        self.ax_hist.axvline(0, color="red", linestyle="--", alpha=0.6,
                              label="Nash (Δ = 0)")
        self.ax_hist.axvline(1, color="green", linestyle="--", alpha=0.6,
                              label="Monopol (Δ = 1)")
        self.ax_hist.axvline(deltas.mean(), color="black", linewidth=2,
                              label=f"Mittel = {deltas.mean():+.3f}")
        self.ax_hist.set_xlabel("Kollusionsindex Δ (Mittelwert ueber "
                                 "Endphase)")
        self.ax_hist.set_ylabel("Anzahl Replikationen")
        self.ax_hist.set_title(
            f"Verteilung von Δ ueber {len(deltas)} Replikationen"
        )
        self.ax_hist.legend(fontsize=7, loc="best")
        self.ax_hist.grid(True, alpha=0.3)

        ok_sorted = sorted(ok, key=lambda r: r.seed)
        seeds = [r.seed for r in ok_sorted]
        ds = [r.final_delta for r in ok_sorted]
        conv = [r.converged for r in ok_sorted]
        colors = ["seagreen" if c else "indianred" for c in conv]
        self.ax_seeds.scatter(seeds, ds, c=colors, s=40, edgecolor="black",
                               linewidth=0.5)
        self.ax_seeds.axhline(0, color="red", linestyle="--", alpha=0.5)
        self.ax_seeds.axhline(1, color="green", linestyle="--", alpha=0.5)
        self.ax_seeds.axhline(deltas.mean(), color="black", alpha=0.5)
        self.ax_seeds.set_xlabel("Seed")
        self.ax_seeds.set_ylabel("Δ")
        self.ax_seeds.set_title("Δ pro Seed (gruen: konvergiert, "
                                 "rot: nicht konvergiert)")
        self.ax_seeds.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def export_csv(self):
        if not self.results or not self.runner:
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            parent=self.win,
            defaultextension=".csv",
            filetypes=[("CSV-Datei", "*.csv"), ("Alle Dateien", "*.*")],
            initialfile=f"calvano_batch_n{self.runner.cfg.n}_"
                        f"reps{len(self.results)}.csv",
        )
        if not path:
            return
        try:
            export_results_csv(self.runner.cfg, self.results, path)
        except Exception as e:
            messagebox.showerror("Export-Fehler", str(e))
            return

        # Falls Q-Tabellen vorhanden sind, parallel exportieren.
        # Format ist per Radiobutton waehlbar: NPZ (kompakt) oder CSV
        # (eine Datei pro Firma und Seed, gut fuer Excel oder pandas).
        has_q = any(r.q_table is not None for r in self.results
                    if not r.error)
        msg_parts = [f"CSV-Zusammenfassung: {path}"]
        fmt = self.q_format_var.get()
        if has_q:
            base = os.path.splitext(path)[0]
            try:
                if fmt == "npz":
                    npz_path = base + "_qtables.npz"
                    export_q_tables_npz(self.runner.cfg, self.results, npz_path)
                    size_mb = os.path.getsize(npz_path) / (1024 * 1024)
                    msg_parts.append(
                        f"Q-Tabellen: {npz_path}\n   ({size_mb:.1f} MB komprimiert)")
                else:
                    out_dir = base + "_qtables"
                    paths = export_q_tables_csv(
                        self.runner.cfg, self.results, out_dir)
                    msg_parts.append(
                        f"Q-Tabellen ({len(paths)} CSV-Dateien): {out_dir}")
            except Exception as e:
                messagebox.showerror(
                    "Q-Tabellen-Export-Fehler",
                    f"CSV-Zusammenfassung wurde gespeichert, "
                    f"aber Q-Tabellen-Export fehlgeschlagen:\n\n{e}",
                )
                return

        # Visits-Heatmap separat speichern (immer als CSV).
        has_visits = any(r.visits is not None for r in self.results
                         if not r.error)
        if has_visits:
            visits_dir = os.path.splitext(path)[0] + "_visits"
            try:
                vp = export_visits_csv(
                    self.runner.cfg, self.results, visits_dir)
                msg_parts.append(
                    f"Visits ({len(vp)} CSV-Dateien): {visits_dir}")
            except Exception as e:
                msg_parts.append(f"Visits-Export fehlgeschlagen: {e}")

        messagebox.showinfo(
            "Export erfolgreich",
            "Ergebnisse gespeichert:\n\n" + "\n".join(msg_parts))


# ==========================================================
# MAIN
# ==========================================================



# ==========================================================
# VERGLEICHSFENSTER: zwei Konfigurationen gleichzeitig
# ==========================================================

class CompareWindow:
    """Fenster zum direkten Vergleich zweier Modellkonfigurationen A und B.

    Beide Laeufe starten parallel als eigene Prozesse (ProcessPoolExecutor),
    streamen Fortschritts-Updates per Manager-Queue zurueck und werden in zwei
    Plots Side-by-side dargestellt. Am Ende erscheint eine numerische
    Vergleichstabelle mit Delta, Konvergenz, Laufzeit und finalen Preisen.

    Typischer Use-Case fuer die Bachelorarbeit: zwei μ-Werte oder zwei n-Werte
    direkt gegenueberstellen, um die ökonomische Auswirkung schnell sichtbar
    zu machen.
    """

    # Tabelle der Parameter, die in beiden Konfigurationen einstellbar sind.
    # Format: (Anzeigename, Schluessel im params-dict, Typ)
    PARAM_LABELS = [
        ("Anzahl Firmen n:", "n", "int"),
        ("Preispunkte m:", "m", "int"),
        ("Lernrate α:", "alpha", "float"),
        ("Explorationsrate β:", "beta", "float"),
        ("Diskontfaktor δ:", "delta", "float"),
        ("Differenzierung μ:", "mu", "float"),
        ("Outside Option a₀:", "a0", "float"),
        ("Preisspanne ξ:", "xi", "float"),
        ("Iterationen:", "episodes", "int"),
        ("Seed:", "seed", "int"),
    ]

    def __init__(self, parent, base_params: dict):
        self.parent = parent
        self.base = base_params
        self.win = tk.Toplevel(parent)
        self.win.title("Vergleichsfenster — Konfiguration A vs. B")
        self.win.geometry("1400x900")
        # Zwei Variablensaetze: A bekommt seed 1, B bekommt seed 2,
        # damit progress-Updates eindeutig zugeordnet werden koennen.
        self.vars_a = self._make_vars(base_params, default_seed=1)
        self.vars_b = self._make_vars(base_params, default_seed=2)
        # Laufzeit-Zustand
        self.runner_thread = None
        self._executor: Optional[ProcessPoolExecutor] = None
        self._futures = []
        self._progress_queue = None
        self._manager = None
        self._cancelled = False
        self.result_a: Optional[RunResult] = None
        self.result_b: Optional[RunResult] = None
        # Live-Trajektorien fuer A und B (werden aus Worker-Progress gefuellt)
        self._traj_a = {"t": [], "delta": [], "eps": []}
        self._traj_b = {"t": [], "delta": [], "eps": []}
        self._poll_thread = None
        self._poll_stop = False
        self.seed_to_label = {}
        self._build_ui()

    def _make_vars(self, p: dict, default_seed: int) -> dict:
        return {
            "n": tk.IntVar(value=int(p["n"])),
            "m": tk.IntVar(value=int(p["m"])),
            "alpha": tk.DoubleVar(value=float(p["alpha"])),
            "beta": tk.DoubleVar(value=float(p["beta"])),
            "delta": tk.DoubleVar(value=float(p["delta"])),
            "mu": tk.DoubleVar(value=float(p["mu"])),
            "a0": tk.DoubleVar(value=float(p["a0"])),
            "xi": tk.DoubleVar(value=float(p["xi"])),
            "episodes": tk.IntVar(value=int(p["episodes"])),
            "seed": tk.IntVar(value=default_seed),
        }

    def _build_ui(self):
        # --- Parameter-Setup: drei Spalten (Label, A, B) ---
        param_frame = ttk.LabelFrame(self.win, text="Parameter-Setup", padding=8)
        param_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(param_frame, text="Parameter", width=22).grid(
            row=0, column=0, padx=4, sticky=tk.W)
        ttk.Label(param_frame, text="Konfiguration A", width=18,
                  foreground="darkblue").grid(row=0, column=1, padx=4)
        ttk.Label(param_frame, text="Konfiguration B", width=18,
                  foreground="darkred").grid(row=0, column=2, padx=4)
        for i, (lbl, key, _) in enumerate(self.PARAM_LABELS, start=1):
            ttk.Label(param_frame, text=lbl, width=22).grid(
                row=i, column=0, sticky=tk.W, padx=4, pady=2)
            ttk.Entry(param_frame, textvariable=self.vars_a[key],
                      width=18).grid(row=i, column=1, padx=4, pady=2)
            ttk.Entry(param_frame, textvariable=self.vars_b[key],
                      width=18).grid(row=i, column=2, padx=4, pady=2)

        # --- Steuerung + Status ---
        ctrl = ttk.Frame(self.win, padding=4)
        ctrl.pack(fill=tk.X, padx=8)
        self.start_btn = ttk.Button(
            ctrl, text="Beide starten", command=self.start_compare)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.cancel_btn = ttk.Button(
            ctrl, text="Abbrechen", command=self.cancel_compare,
            state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=4)
        self.status_label = ttk.Label(ctrl, text="Bereit.", foreground="gray")
        self.status_label.pack(side=tk.LEFT, padx=8)

        # --- Plots A und B nebeneinander ---
        plot_frame = ttk.Frame(self.win)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.fig = Figure(figsize=(12, 4.5), dpi=100)
        self.ax_a = self.fig.add_subplot(1, 2, 1)
        self.ax_b = self.fig.add_subplot(1, 2, 2)
        self._reset_axes()
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame).update()

        # --- Vergleichstabelle ---
        cmp_frame = ttk.LabelFrame(self.win, text="Vergleich A vs. B", padding=8)
        cmp_frame.pack(fill=tk.X, padx=8, pady=4)
        self.cmp_label = ttk.Label(
            cmp_frame, text="(noch keine Ergebnisse)",
            font=("Courier", 10), justify=tk.LEFT)
        self.cmp_label.pack(anchor=tk.W)

    def _reset_axes(self):
        for ax, title, color in [
            (self.ax_a, "Konfiguration A — Δ über t", "darkblue"),
            (self.ax_b, "Konfiguration B — Δ über t", "darkred"),
        ]:
            ax.clear()
            ax.set_title(title, color=color)
            ax.set_xlabel("Iteration t")
            ax.set_ylabel("Δ")
            ax.set_ylim(-0.1, 1.2)
            ax.axhline(0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.axhline(1, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
            ax.grid(True, alpha=0.3)

    def _build_config(self, vars_dict: dict) -> BatchConfig:
        n = int(vars_dict["n"].get())
        # Firmenspezifische Parameter a_i / c_i aus den Basis-Listen ableiten;
        # bei abweichendem n auffuellen oder kuerzen.
        a_base = list(self.base.get("a", [2.0] * n))
        c_base = list(self.base.get("c", [1.0] * n))
        a = (a_base + [a_base[0]] * n)[:n] if a_base else [2.0] * n
        c = (c_base + [c_base[0]] * n)[:n] if c_base else [1.0] * n
        return BatchConfig(
            n=n,
            m=int(vars_dict["m"].get()),
            alpha=float(vars_dict["alpha"].get()),
            beta=float(vars_dict["beta"].get()),
            delta=float(vars_dict["delta"].get()),
            mu=float(vars_dict["mu"].get()),
            a0=float(vars_dict["a0"].get()),
            xi=float(vars_dict["xi"].get()),
            a=a, c=c,
            episodes=int(vars_dict["episodes"].get()),
            seeds=[int(vars_dict["seed"].get())],
            avg_window=1000,
            init_strategy=str(self.base.get("init_strategy", "best_response")),
        )

    def start_compare(self):
        if self.runner_thread is not None and self.runner_thread.is_alive():
            return
        try:
            cfg_a = self._build_config(self.vars_a)
            cfg_b = self._build_config(self.vars_b)
        except Exception as e:
            messagebox.showerror("Parameter-Fehler", str(e))
            return
        # Unterschiedliche Seeds verlangen, damit progress-Updates eindeutig
        # einer Konfiguration zugeordnet werden koennen.
        if cfg_a.seeds[0] == cfg_b.seeds[0]:
            messagebox.showwarning(
                "Hinweis",
                "Bitte unterschiedliche Seeds fuer A und B verwenden.")
            return
        self.seed_to_label = {cfg_a.seeds[0]: "A", cfg_b.seeds[0]: "B"}
        self.result_a = None
        self.result_b = None
        self._traj_a = {"t": [], "delta": [], "eps": []}
        self._traj_b = {"t": [], "delta": [], "eps": []}
        self._reset_axes()
        self.canvas.draw_idle()
        self.cmp_label.config(text="läuft ...")
        self.status_label.config(text="Beide Läufe gestartet ...",
                                  foreground="darkorange")
        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self._cancelled = False
        self.runner_thread = threading.Thread(
            target=self._run_both, args=(cfg_a, cfg_b), daemon=True)
        self.runner_thread.start()

    def _run_both(self, cfg_a: BatchConfig, cfg_b: BatchConfig):
        """Worker-Thread: startet beide Replikationen parallel und sammelt
        progress-Updates ueber eine Manager-Queue."""
        try:
            self._manager = mp.Manager()
            self._progress_queue = self._manager.Queue()
            self._executor = ProcessPoolExecutor(max_workers=2)
            self._futures = [
                self._executor.submit(
                    _run_single_replication,
                    (cfg_a, cfg_a.seeds[0], self._progress_queue)),
                self._executor.submit(
                    _run_single_replication,
                    (cfg_b, cfg_b.seeds[0], self._progress_queue)),
            ]

            # Polling-Thread fuer Live-Updates aus den Worker-Prozessen
            self._poll_stop = False

            def poll_loop():
                while not self._poll_stop:
                    try:
                        msg = self._progress_queue.get(timeout=0.2)
                    except queue_module.Empty:
                        continue
                    except Exception:
                        break
                    try:
                        kind, seed, info = msg
                    except Exception:
                        continue
                    label = self.seed_to_label.get(seed)
                    if not label:
                        continue
                    if kind == "progress":
                        self.win.after(0, lambda l=label, i=info:
                                        self._update_traj(l, i))

            self._poll_thread = threading.Thread(target=poll_loop, daemon=True)
            self._poll_thread.start()

            # Warte auf beide Futures
            for fut in self._futures:
                if self._cancelled:
                    break
                try:
                    res = fut.result()
                except Exception as e:
                    self.win.after(0, lambda err=str(e):
                                    self.status_label.config(
                                        text=f"Worker-Fehler: {err}",
                                        foreground="red"))
                    continue
                label = self.seed_to_label.get(res.seed)
                if label == "A":
                    self.result_a = res
                elif label == "B":
                    self.result_b = res

            self._poll_stop = True
            self.win.after(0, self._finalize)
        except Exception as e:
            self.win.after(0, lambda err=str(e):
                            self.status_label.config(
                                text=f"Fehler: {err}", foreground="red"))
        finally:
            try:
                if self._executor:
                    self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._executor = None
            try:
                if self._manager:
                    self._manager.shutdown()
            except Exception:
                pass
            self._manager = None
            self._progress_queue = None

    def _update_traj(self, label: str, info: dict):
        """Aktualisiere die Trajektorie und das Plot fuer A oder B."""
        t = info.get("t", 0)
        d = info.get("delta", float("nan"))
        traj = self._traj_a if label == "A" else self._traj_b
        traj["t"].append(t)
        traj["delta"].append(d)
        ax = self.ax_a if label == "A" else self.ax_b
        color = "darkblue" if label == "A" else "darkred"
        ax.clear()
        ax.plot(traj["t"], traj["delta"], color=color, linewidth=1.0)
        ax.axhline(0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(1, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_title(f"Konfiguration {label} — Δ über t", color=color)
        ax.set_xlabel("Iteration t")
        ax.set_ylabel("Δ")
        ax.set_ylim(-0.1, 1.2)
        ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def _finalize(self):
        """Nach Abschluss beider Laeufe Vergleichstabelle anzeigen."""
        if self.result_a is not None and self.result_b is not None:
            ra, rb = self.result_a, self.result_b
            def fmt_list(xs):
                return "[" + ", ".join(f"{p:.3f}" for p in xs) + "]"
            txt = (
                f"{'':22}{'A':>12}{'B':>12}{'A − B':>12}\n"
                f"{'Δ final':22}{ra.final_delta:>+12.4f}{rb.final_delta:>+12.4f}"
                f"{(ra.final_delta - rb.final_delta):>+12.4f}\n"
                f"{'Konvergiert':22}{('ja' if ra.converged else 'nein'):>12}"
                f"{('ja' if rb.converged else 'nein'):>12}\n"
                f"{'Laufzeit (s)':22}{ra.elapsed_s:>12.1f}{rb.elapsed_s:>12.1f}\n"
                f"{'Seed':22}{ra.seed:>12}{rb.seed:>12}\n"
                f"\n"
                f"p_Nash    A: {fmt_list(ra.p_nash)}\n"
                f"p_Nash    B: {fmt_list(rb.p_nash)}\n"
                f"p_Monopol A: {fmt_list(ra.p_mono)}\n"
                f"p_Monopol B: {fmt_list(rb.p_mono)}\n"
                f"erlernte Preise A: {fmt_list(ra.final_prices)}\n"
                f"erlernte Preise B: {fmt_list(rb.final_prices)}"
            )
            self.cmp_label.config(text=txt)
            self.status_label.config(
                text="Beide Läufe abgeschlossen.", foreground="darkgreen")
        else:
            self.status_label.config(
                text="Lauf unvollständig (abgebrochen oder Fehler).",
                foreground="orange")
        self.start_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)

    def cancel_compare(self):
        self._cancelled = True
        try:
            if self._executor:
                for f in self._futures:
                    try:
                        f.cancel()
                    except Exception:
                        pass
                self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.status_label.config(text="Abbruch angefordert.", foreground="orange")
        self.cancel_btn.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    CalvanoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
