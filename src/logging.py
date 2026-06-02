from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import EvalConfig, TrainConfig
from .policy import TransformerPolicy


@dataclass(slots=True)
class EvalResult:
    opponent_name: str
    win_rate: float
    loss_rate: float
    tie_rate: float
    n_games: int


class TrainLogger:
    """TensorBoard + CSV logger for training metrics."""

    def __init__(self, log_dir: str | Path, run_name: str) -> None:
        self.log_path = Path(log_dir) / run_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(self.log_path))

        # CSV. Metric keys can vary across updates (e.g. the PPG aux phase only
        # runs every aux_every updates, adding aux_value_loss/aux_kl), so the
        # header is the running UNION of all keys seen: when a new key first
        # appears we expand the header and rewrite the file. Rows are kept in
        # memory for that (rare) rewrite. All metrics also go to TensorBoard, so
        # nothing is lost regardless.
        self._csv_path = self.log_path / "metrics.csv"
        self._csv_fieldnames: list[str] = ["update"]
        self._csv_rows: list[dict] = []
        self._csv_file = open(self._csv_path, "w", newline="")  # noqa: SIM115
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=self._csv_fieldnames, extrasaction="ignore")
        self._csv_writer.writeheader()

    def log_update(self, update: int, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, update)

        row = {"update": update, **metrics}
        self._csv_rows.append(row)

        new_keys = [k for k in metrics if k not in self._csv_fieldnames]
        if new_keys:
            # Expand header (keep "update" first, rest sorted) and rewrite the file.
            self._csv_fieldnames = ["update"] + sorted(
                set(self._csv_fieldnames[1:]) | set(new_keys))
            self._csv_file.close()
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_fieldnames, extrasaction="ignore")
            self._csv_writer.writeheader()
            self._csv_writer.writerows(self._csv_rows)
        else:
            self._csv_writer.writerow(row)
        self._csv_file.flush()

    def log_eval(self, update: int, results: list[EvalResult]) -> None:
        for r in results:
            self.writer.add_scalar(f"eval/{r.opponent_name}_win_rate", r.win_rate, update)
            self.writer.add_scalar(f"eval/{r.opponent_name}_loss_rate", r.loss_rate, update)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
        self._csv_file.close()


def make_eval_agent(
    policy: TransformerPolicy,
    cfg: TrainConfig,
    device: torch.device,
) -> Any:
    """Create a Kaggle-compatible agent callable from a policy."""
    from .opponents import _policy_act

    # Snapshot weights so eval doesn't interfere with training
    eval_policy = TransformerPolicy(cfg.model, cfg.env).to(device)
    eval_policy.load_state_dict(policy.state_dict())
    eval_policy.eval()

    def agent(obs: Any, config: Any = None) -> list:
        return _policy_act(eval_policy, obs, cfg, device, deterministic=True)

    return agent


def run_periodic_eval(
    policy: TransformerPolicy,
    cfg: TrainConfig,
    device: torch.device,
) -> list[EvalResult]:
    """Run evaluation games against configured opponents."""
    from evaluation.evaluate import run_games

    eval_agent = make_eval_agent(policy, cfg, device)
    results: list[EvalResult] = []

    for opp_name in cfg.eval.eval_opponents:
        opp_callable = _get_eval_opponent(opp_name)
        raw = run_games(eval_agent, opp_callable, n_games=cfg.eval.eval_games)
        results.append(EvalResult(
            opponent_name=opp_name,
            win_rate=raw["win_rate"],
            loss_rate=raw["loss_rate"],
            tie_rate=raw["tie_rate"],
            n_games=raw["n_games"],
        ))

    return results


def _get_eval_opponent(name: str) -> Any:
    if name == "apex":
        from agents.apex import agent as apex_agent
        return apex_agent
    if name == "random":
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent
        return random_agent
    if name == "hybrid":
        from agents.hybrid import agent as hybrid_agent
        return hybrid_agent
    raise ValueError(f"Unknown eval opponent: {name}")
