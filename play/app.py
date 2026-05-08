"""Flask app for interactive Orbit Wars browser play."""

import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

# Ensure repo root on path for agent imports
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from agents.aggressive import agent as aggressive_agent


def _serialize_obs(obs):
    """Convert a kaggle Struct observation to a plain dict for JSON."""
    d = dict(obs)
    # planets / fleets / initial_planets are already plain lists of lists
    # comets may contain Structs with nested data
    if d.get("comets"):
        serialized = []
        for c in d["comets"]:
            if hasattr(c, "keys"):
                serialized.append(dict(c))
            else:
                serialized.append(c)
        d["comets"] = serialized
    return d


class GameManager:
    """Wraps kaggle-environments to run a 2-player Orbit Wars game turn-by-turn."""

    def __init__(self, opponent_fn):
        self.opponent_fn = opponent_fn
        self.env = None
        self.log = None
        self.done = False

    def new_game(self):
        from kaggle_environments import make

        self.env = make("orbit_wars", debug=True)
        result = self.env.reset(2)
        self.done = False
        self.log = {"turns": [], "result": None, "config": {"opponent": "aggressive"}}

        obs0 = _serialize_obs(result[0].observation)
        obs1 = _serialize_obs(result[1].observation)

        # Record initial state
        self.log["turns"].append({
            "step": 0,
            "observation": obs0,
            "human_action": None,
            "opponent_action": None,
        })

        return {"observation": obs0, "done": False, "reward": 0, "step": 0}

    def step(self, human_action):
        if self.done or self.env is None:
            return None

        # Get opponent's observation from current state and compute its action
        opp_obs = self.env.state[1].observation
        opp_action = self.opponent_fn(opp_obs)

        # Step the environment: player 0 = human, player 1 = opponent
        result = self.env.step([human_action, opp_action])

        obs0 = _serialize_obs(result[0].observation)
        reward = result[0].reward
        status = result[0].status

        done = status != "ACTIVE"
        self.done = done

        # Record turn
        self.log["turns"].append({
            "step": obs0.get("step", 0),
            "observation": obs0,
            "human_action": human_action,
            "opponent_action": opp_action,
        })

        if done:
            self.log["result"] = {
                "reward": reward,
                "steps": obs0.get("step", 0),
                "won": reward is not None and reward > 0,
            }

        return {
            "observation": obs0,
            "done": done,
            "reward": reward,
            "step": obs0.get("step", 0),
        }

    def export_log(self):
        return self.log


def create_app(opponent="aggressive"):
    app = Flask(__name__, template_folder="templates", static_folder="static")

    if opponent == "aggressive":
        opponent_fn = aggressive_agent
    else:
        # Default: random does nothing
        opponent_fn = lambda obs, config=None: []

    game = GameManager(opponent_fn)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/new-game", methods=["POST"])
    def new_game():
        state = game.new_game()
        return jsonify(state)

    @app.route("/api/step", methods=["POST"])
    def step():
        if game.done or game.env is None:
            return jsonify({"error": "Game is over or not started"}), 400

        data = request.get_json(force=True)
        human_action = data.get("action", [])
        result = game.step(human_action)

        if result is None:
            return jsonify({"error": "Game is over"}), 400

        return jsonify(result)

    @app.route("/api/export", methods=["GET"])
    def export_log():
        log = game.export_log()
        if log is None:
            return jsonify({"error": "No game data"}), 400
        return app.response_class(
            response=json.dumps(log, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=orbit_wars_log.json"},
        )

    return app
