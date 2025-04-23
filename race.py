"""Monte Carlo Race Simulation with Advanced Strategies

This program simulates car races using a Monte‑Carlo approach.  It supports
several **tactics**:

* **Tyre degradation** – gradual loss of pace until a pit‑stop refreshes tyres.
* **Boost laps**       – moments of all‑out attack that add raw speed but raise
                         crash risk.
* **Flexible stops**   – any number of scheduled pit laps.

The high‑level driver code at the bottom shows seven competitors, each with a
unique mix of tactics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Car:
    """Single competitor with strategy parameters."""

    name: str
    base_speed: float         # [m/s] baseline pace on fresh tyres, clear air
    speed_std: float          # [m/s] lap‑to‑lap stochastic variance (driver consistency)
    crash_prob: float         # baseline probability to crash on any lap

    # Pit‑stop & tyre model
    pit_mean: float           # [s] mean stationary time in pit lane
    pit_std: float            # [s] pit‑stop time variation
    pit_laps: List[int] = field(default_factory=list)
    tyre_decay: float = 0.0   # [% of base_speed] pace lost per lap since last tyre change

    # "Push" mode – optional laps with extra speed and extra danger
    boost_laps: List[int] = field(default_factory=list)
    boost_speed: float = 0.0  # [m/s] flat speed boost during push laps
    boost_crash: float = 0.0  # absolute addition to crash probability during push laps

@dataclass
class Track:
    """Simple circuit model."""

    lap_length: float  # [m]
    n_laps: int
    weather_variance: float = 0.0  # extra Gaussian noise all cars share each lap [m/s]

# ---------------------------------------------------------------------------
# Core simulation engine – *one* race
# ---------------------------------------------------------------------------

class RaceSimulator:
    """Simulates a single stochastic race with given cars & track."""

    def __init__(self, cars: List[Car], track: Track):
        self.cars = cars
        self.track = track

    # ---------------- Internal helpers ----------------
    @staticmethod
    def _pit_stop_time(car: Car) -> float:
        return max(np.random.normal(car.pit_mean, car.pit_std), 0)

    def run(self, verbose: bool = False) -> Dict[str, Dict]:
        """Run race once; returns {car_name: {'time', 'status', 'laps'}}."""
        results: Dict[str, Dict] = {}

        for car in self.cars:
            total_time = 0.0
            status = "Finished"
            last_pit_lap = 0

            for lap in range(1, self.track.n_laps + 1):
                # -- Determine crash risk for this lap
                lap_crash_prob = car.crash_prob
                if lap in car.boost_laps:
                    lap_crash_prob += car.boost_crash
                if np.random.rand() < lap_crash_prob:
                    status = f"DNF on lap {lap}"
                    break

                # -- Determine effective speed for this lap
                laps_since_pit = lap - last_pit_lap - 1  # 0 on first lap after stop
                tyre_penalty = car.base_speed * car.tyre_decay * laps_since_pit
                speed = (
                    np.random.normal(car.base_speed, car.speed_std)
                    - tyre_penalty
                    + (car.boost_speed if lap in car.boost_laps else 0)
                    + np.random.normal(0, self.track.weather_variance)
                )
                speed = max(speed, 1e-3)  # safety clamp
                lap_time = self.track.lap_length / speed
                total_time += lap_time

                # -- Optional pit‑stop at end of lap
                if lap in car.pit_laps:
                    total_time += self._pit_stop_time(car)
                    last_pit_lap = lap

            results[car.name] = {
                "time": np.nan if status != "Finished" else total_time,
                "status": status,
                "laps": lap if status != "Finished" else self.track.n_laps,
            }
            if verbose:
                print(f"{car.name:<12} {status:<13} laps={results[car.name]['laps']:<3} time={total_time:8.1f}s")

        return results

# ---------------------------------------------------------------------------
# Monte Carlo wrapper – many races
# ---------------------------------------------------------------------------

class MonteCarloRace:
    """Runs many simulations and aggregates stats."""

    def __init__(self, cars: List[Car], track: Track):
        self.cars = cars
        self.track = track

    def run(self, n_iter: int = 10_000, progress_every: int = 0) -> pd.DataFrame:
        # Tracking dicts
        winner_counts = {c.name: 0 for c in self.cars}
        podium_counts = {c.name: 0 for c in self.cars}
        finish_times = {c.name: [] for c in self.cars}
        dnfs = {c.name: 0 for c in self.cars}

        for i in range(1, n_iter + 1):
            res = RaceSimulator(self.cars, self.track).run()
            finished = {n: d for n, d in res.items() if d["status"] == "Finished"}
            if finished:
                # -- Win & podium stats
                order = sorted(finished.items(), key=lambda kv: kv[1]["time"])
                winner_counts[order[0][0]] += 1
                for n, _ in order[:3]:
                    podium_counts[n] += 1
            # -- Time & DNF stats
            for n, d in res.items():
                if d["status"] == "Finished":
                    finish_times[n].append(d["time"])
                else:
                    dnfs[n] += 1
            if progress_every and i % progress_every == 0:
                print(f"Simulations completed: {i}/{n_iter}")

        # Build summary table
        summary = []
        for c in self.cars:
            name = c.name
            finishes = len(finish_times[name])
            summary.append({
                "Car": name,
                "Win %": 100 * winner_counts[name] / n_iter,
                "Podium %": 100 * podium_counts[name] / n_iter,
                "DNF %": 100 * dnfs[name] / n_iter,
                "Avg time (s)": np.mean(finish_times[name]) if finishes else np.nan,
            })
        return (
            pd.DataFrame(summary)
            .sort_values("Win %", ascending=False)
            .reset_index(drop=True)
        )

# ---------------------------------------------------------------------------
# Example: 7 unique tactics
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)  # remove/reseed for different runs

    cars = [
        # Fast early stint, single stop
        Car(
            "Rocket",
            base_speed=85,
            speed_std=2.5,
            crash_prob=0.01,
            pit_mean=18,
            pit_std=2,
            pit_laps=[20],
            tyre_decay=0.003,
            boost_laps=[5, 6],
            boost_speed=4,
            boost_crash=0.01,
        ),
        # One stop, very consistent
        Car(
            "Steady",
            base_speed=80,
            speed_std=1.0,
            crash_prob=0.002,
            pit_mean=20,
            pit_std=1.5,
            pit_laps=[22],
            tyre_decay=0.002,
        ),
        # Two‑stop with big boosts
        Car(
            "Risky",
            base_speed=87,
            speed_std=4.0,
            crash_prob=0.03,
            pit_mean=17,
            pit_std=3,
            pit_laps=[18, 34],
            tyre_decay=0.004,
            boost_laps=[18, 34],
            boost_speed=5,
            boost_crash=0.02,
        ),
        # No stops, minimal risk, low tyre drop‑off
        Car(
            "Eco",
            base_speed=78,
            speed_std=1.5,
            crash_prob=0.001,
            pit_mean=0,
            pit_std=0,
            pit_laps=[],
            tyre_decay=0.001,
        ),
        # Two‑stop sprinter: huge early jump then tyre fade
        Car(
            "Sprinter",
            base_speed=90,
            speed_std=3.0,
            crash_prob=0.015,
            pit_mean=21,
            pit_std=2,
            pit_laps=[12, 28],
            tyre_decay=0.005,
            boost_laps=[1, 2, 30],
            boost_speed=6,
            boost_crash=0.015,
        ),
        # Ultra conservative – finishes almost always, zero stops
        Car(
            "Conserve",
            base_speed=76,
            speed_std=1.0,
            crash_prob=0.0008,
            pit_mean=0,
            pit_std=0,
            pit_laps=[],
            tyre_decay=0.0005,
        ),
        # Balanced hybrid strategy
        Car(
            "Balanced",
            base_speed=82,
            speed_std=1.8,
            crash_prob=0.005,
            pit_mean=19,
            pit_std=1.8,
            pit_laps=[25],
            tyre_decay=0.002,
            boost_laps=[35],
            boost_speed=3,
            boost_crash=0.005,
        ),
    ]

    track = Track(lap_length=5000, n_laps=40, weather_variance=0.5)
    mc = MonteCarloRace(cars, track)

    summary = mc.run(n_iter=8000)
    print("\n==== Monte Carlo Summary (8000 races) ====")
    pd.set_option("display.float_format", lambda x: f"{x:0.2f}")
    print(summary.to_string(index=False))

    summary.to_csv("race_montecarlo_summary.csv", index=False)
    print("\nSaved: race_montecarlo_summary.csv")
