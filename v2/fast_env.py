"""Fast, engine-faithful Orbit Wars simulator for self-play training.

This re-implements the Kaggle `interpreter()` turn loop as a standalone, steppable
environment that runs *without* the kaggle_environments harness — so it can be driven
at high throughput for self-play RL. See `rl_research/SIMULATOR_AUDIT.md` for the
fidelity requirements this addresses (opponent-in-sim, moving fleets with continuous
collision, engine-exact two-stage combat, correct turn order, orbit rotation, comets).

Design choices for fidelity:
  * Internal state uses the engine's *exact* list layouts:
        planet = [id, owner, x, y, radius, ships, production]
        fleet  = [id, owner, x, y, angle, from_planet_id, ships]
  * Map generation and comet paths reuse the engine's own `generate_planets` /
    `generate_comet_paths` (identical RNG seeding), so starts match bit-for-bit.
  * The per-step logic is a line-by-line port of `interpreter()`'s turn order:
        0 launch (all players) -> 1 production -> 2 fleet move + continuous collision
        -> 3 rotation + comet move + sweep -> 4 combat -> terminate/score.

This is the *scalar* reference implementation. Once `tests` confirm it matches the
Kaggle engine step-for-step, a batched (vectorized) version reuses this exact logic.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    generate_comet_paths,
    generate_planets,
    point_to_segment_distance,
)

# Engine defaults (from orbit_wars.json specification).
DEFAULT_EPISODE_STEPS = 500
DEFAULT_SHIP_SPEED = 6.0
DEFAULT_COMET_SPEED = 4.0

# Move = [from_planet_id, angle, ships]; an agent returns list[Move].
Move = list
AgentFn = Callable[[dict], list]  # obs dict -> moves


def _planet_obs(p: list) -> dict:
    return {
        "id": p[0],
        "owner": p[1],
        "x": p[2],
        "y": p[3],
        "radius": p[4],
        "ships": p[5],
        "production": p[6],
    }


def _fleet_obs(f: list) -> dict:
    return {
        "id": f[0],
        "owner": f[1],
        "x": f[2],
        "y": f[3],
        "angle": f[4],
        "from_planet_id": f[5],
        "ships": f[6],
    }


@dataclass
class StepResult:
    done: bool
    rewards: list[float]  # per-player terminal reward (+1/-1), all 0 until done
    scores: list[float]  # per-player score (ships on owned planets + owned fleets)


class FastOrbitWars:
    """Standalone, engine-faithful Orbit Wars environment.

    Usage:
        env = FastOrbitWars(num_agents=2, seed=123)
        obs = env.reset()
        while not env.done:
            actions = [agent0(env.observation(0)), agent1(env.observation(1))]
            res = env.step(actions)
    """

    def __init__(
        self,
        num_agents: int = 2,
        seed: int | None = None,
        episode_steps: int = DEFAULT_EPISODE_STEPS,
        ship_speed: float = DEFAULT_SHIP_SPEED,
        comet_speed: float = DEFAULT_COMET_SPEED,
    ) -> None:
        self.num_agents = num_agents
        self.episode_steps = episode_steps
        self.ship_speed = ship_speed
        self.comet_speed = comet_speed
        self._seed = seed
        self.reset(seed=seed)

    # ── Reset / map generation (mirrors interpreter init block) ──────────────
    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self._seed = seed
        seed_val = self._seed if self._seed is not None else random.randrange(2**31)
        self._episode_seed = seed_val
        rng = random.Random(seed_val)

        self.angular_velocity = rng.uniform(0.025, 0.05)
        self.planets = generate_planets(rng)
        self.initial_planets = [p.copy() for p in self.planets]
        self.fleets: list[list] = []
        self.next_fleet_id = 0
        self.comets: list[dict] = []
        self.comet_planet_ids: list[int] = []
        self.step_num = 0
        self.done = False
        self.rewards = [0.0] * self.num_agents

        # Assign home planets — random symmetric group of 4 (mirrors engine).
        num_groups = len(self.planets) // 4
        if num_groups > 0:
            base = rng.randint(0, num_groups - 1) * 4
            if self.num_agents == 2:
                self.planets[base][1] = 0
                self.planets[base][5] = 10
                self.planets[base + 3][1] = 1
                self.planets[base + 3][5] = 10
            elif self.num_agents == 4:
                for j in range(4):
                    self.planets[base + j][1] = j
                    self.planets[base + j][5] = 10
        return self.observation(0)

    # ── Observation (matches the dict layout parse_observation expects) ──────
    def observation(self, player: int) -> dict:
        return {
            "player": player,
            "step": self.step_num,
            "angular_velocity": self.angular_velocity,
            "planets": [_planet_obs(p) for p in self.planets],
            "fleets": [_fleet_obs(f) for f in self.fleets],
            "initial_planets": [_planet_obs(p) for p in self.initial_planets],
            "comets": self.comets,
            "comet_planet_ids": list(self.comet_planet_ids),
        }

    # ── One full turn (line-by-line port of interpreter()) ───────────────────
    def step(self, actions: list[list]) -> StepResult:
        if self.done:
            return StepResult(True, self.rewards, self._scores())

        # Remove expired comets before launch (so agents can't act on them).
        self._expire_comets_pre_launch()
        # Spawn comets at designated steps (engine spawns when step+1 in schedule).
        self._spawn_comets()

        # 0. Fleet launch — all players, in order.
        for pid in range(self.num_agents):
            self._process_moves(pid, actions[pid] if pid < len(actions) else [])

        # 1. Production.
        for planet in self.planets:
            if planet[1] != -1:
                planet[5] += planet[6]

        # 2. Fleet movement + continuous collision (planet -> bounds -> sun).
        fleets_to_remove: list[list] = []
        combat_lists: dict[int, list] = {p[0]: [] for p in self.planets}
        self._move_fleets(fleets_to_remove, combat_lists)

        # 3. Planet rotation + comet movement + sweep.
        self._rotate_and_sweep(fleets_to_remove, combat_lists)

        # Remove dead fleets (identity-based, like the engine).
        self.fleets = [f for f in self.fleets if f not in fleets_to_remove]

        # 4. Combat resolution (engine-exact two-stage).
        self._resolve_combat(combat_lists)

        # Advance step and check termination.
        self.step_num += 1
        return self._check_termination()

    # ── Phase helpers ────────────────────────────────────────────────────────
    def _process_moves(self, player_id: int, action: list) -> None:
        if not action or not isinstance(action, list):
            return
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            from_planet = next((p for p in self.planets if p[0] == from_id), None)
            if from_planet and from_planet[1] == player_id:
                if from_planet[5] >= ships and ships > 0:
                    from_planet[5] -= ships
                    start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                    start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                    self.fleets.append(
                        [self.next_fleet_id, player_id, start_x, start_y, angle, from_id, ships]
                    )
                    self.next_fleet_id += 1

    def _move_fleets(self, fleets_to_remove: list, combat_lists: dict) -> None:
        max_speed = self.ship_speed
        for fleet in self.fleets:
            angle = fleet[4]
            ships = fleet[6]
            speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
            speed = min(speed, max_speed)
            old_pos = (fleet[2], fleet[3])
            fleet[2] += math.cos(angle) * speed
            fleet[3] += math.sin(angle) * speed
            new_pos = (fleet[2], fleet[3])

            # Planet collision first (so fast fleets get credit for a hit en route).
            hit_planet = False
            for planet in self.planets:
                if point_to_segment_distance((planet[2], planet[3]), old_pos, new_pos) < planet[4]:
                    combat_lists[planet[0]].append(fleet)
                    fleets_to_remove.append(fleet)
                    hit_planet = True
                    break
            if hit_planet:
                continue
            # Out of bounds.
            if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
                fleets_to_remove.append(fleet)
                continue
            # Sun crossing.
            if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
                fleets_to_remove.append(fleet)

    def _rotate_and_sweep(self, fleets_to_remove: list, combat_lists: dict) -> None:
        comet_pid_set = set(self.comet_planet_ids)
        initial_by_id = {p[0]: p for p in self.initial_planets}

        def sweep(planet, old_pos, new_pos):
            if old_pos == new_pos:
                return
            for fleet in self.fleets:
                if fleet not in fleets_to_remove:
                    if (
                        point_to_segment_distance((fleet[2], fleet[3]), old_pos, new_pos)
                        < planet[4]
                    ):
                        combat_lists[planet[0]].append(fleet)
                        fleets_to_remove.append(fleet)

        # Regular rotation (engine uses self.step_num as the rotation step).
        for planet in self.planets:
            if planet[0] in comet_pid_set:
                continue
            initial_p = initial_by_id.get(planet[0])
            if not initial_p:
                continue
            dx = initial_p[2] - CENTER
            dy = initial_p[3] - CENTER
            r = math.sqrt(dx**2 + dy**2)
            old_pos = (planet[2], planet[3])
            if r + planet[4] < ROTATION_RADIUS_LIMIT:
                initial_angle = math.atan2(dy, dx)
                current_angle = initial_angle + self.angular_velocity * self.step_num
                planet[2] = CENTER + r * math.cos(current_angle)
                planet[3] = CENTER + r * math.sin(current_angle)
            sweep(planet, old_pos, (planet[2], planet[3]))

        # Comet movement along precomputed paths.
        expired: list[int] = []
        for group in self.comets:
            group["path_index"] += 1
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                planet = next((p for p in self.planets if p[0] == pid), None)
                if planet is None:
                    continue
                p_path = group["paths"][i]
                if idx >= len(p_path):
                    expired.append(pid)
                else:
                    old_pos = (planet[2], planet[3])
                    planet[2] = p_path[idx][0]
                    planet[3] = p_path[idx][1]
                    if old_pos[0] >= 0:
                        sweep(planet, old_pos, (planet[2], planet[3]))
        if expired:
            self._remove_planets(set(expired))

    def _resolve_combat(self, combat_lists: dict) -> None:
        for pid, planet_fleets in combat_lists.items():
            planet = next((p for p in self.planets if p[0] == pid), None)
            if not planet or not planet_fleets:
                continue
            # Sum arriving ships per player (garrison NOT included — engine-exact).
            player_ships: dict[int, int] = {}
            for fleet in planet_fleets:
                player_ships[fleet[1]] = player_ships.get(fleet[1], 0) + fleet[6]
            if not player_ships:
                continue
            sorted_players = sorted(player_ships.items(), key=lambda kv: kv[1], reverse=True)
            top_player, top_ships = sorted_players[0]
            if len(sorted_players) > 1:
                second_ships = sorted_players[1][1]
                survivor_ships = top_ships - second_ships
                if sorted_players[0][1] == sorted_players[1][1]:
                    survivor_ships = 0
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships
            # Survivor vs garrison.
            if survivor_ships > 0:
                if planet[1] == survivor_owner:
                    planet[5] += survivor_ships
                else:
                    planet[5] -= survivor_ships
                    if planet[5] < 0:
                        planet[1] = survivor_owner
                        planet[5] = abs(planet[5])

    # ── Comet spawn / expire (mirror engine) ─────────────────────────────────
    def _expire_comets_pre_launch(self) -> None:
        expired = []
        for group in self.comets:
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                if idx >= len(group["paths"][i]):
                    expired.append(pid)
        if expired:
            self._remove_planets(set(expired))

    def _spawn_comets(self) -> None:
        if (self.step_num + 1) not in COMET_SPAWN_STEPS:
            return
        comet_rng = random.Random(f"orbit_wars-comet-{self._episode_seed}-{self.step_num + 1}")
        paths = generate_comet_paths(
            self.initial_planets,
            self.angular_velocity,
            self.step_num + 1,
            self.comet_planet_ids,
            self.comet_speed,
            rng=comet_rng,
        )
        if not paths:
            return
        next_id = max(p[0] for p in self.planets) + 1
        comet_ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        group = {"planet_ids": [], "paths": paths, "path_index": -1}
        for i in range(len(paths)):
            pid = next_id + i
            group["planet_ids"].append(pid)
            self.comet_planet_ids.append(pid)
            planet = [pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
            self.planets.append(planet)
            self.initial_planets.append(planet[:])
        self.comets.append(group)

    def _remove_planets(self, dead: set[int]) -> None:
        self.planets = [p for p in self.planets if p[0] not in dead]
        self.initial_planets = [p for p in self.initial_planets if p[0] not in dead]
        self.comet_planet_ids = [pid for pid in self.comet_planet_ids if pid not in dead]
        for group in self.comets:
            group["planet_ids"] = [pid for pid in group["planet_ids"] if pid not in dead]
        self.comets = [g for g in self.comets if g["planet_ids"]]

    # ── Termination / scoring (mirror engine) ────────────────────────────────
    def _scores(self) -> list[float]:
        scores = [0.0] * self.num_agents
        for p in self.planets:
            if p[1] != -1:
                scores[p[1]] += p[5]
        for f in self.fleets:
            scores[f[1]] += f[6]
        return scores

    def _check_termination(self) -> StepResult:
        terminated = self.step_num >= self.episode_steps - 2
        alive = set()
        for p in self.planets:
            if p[1] != -1:
                alive.add(p[1])
        for f in self.fleets:
            alive.add(f[1])
        if len(alive) <= 1:
            terminated = True

        if terminated:
            self.done = True
            scores = self._scores()
            max_score = max(scores) if scores else 0
            for i in range(self.num_agents):
                self.rewards[i] = 1.0 if (scores[i] == max_score and max_score > 0) else -1.0
            return StepResult(True, self.rewards, scores)
        return StepResult(False, [0.0] * self.num_agents, self._scores())
