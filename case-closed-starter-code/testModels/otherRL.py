"""
Flask agent with RL model inference
CRITICAL: Must respond within 4 seconds per move
"""

import os
import sys
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import time

# Add parent directory to import game logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case_closed_game import Game, Direction, GameResult

# Import our trained model
from fast_inference import FastInferenceAgent

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

PARTICIPANT = "YourTeamName"  # CHANGE THIS
AGENT_NAME = "SelfPlayRL"

# Load trained model (CHANGE PATH TO YOUR MODEL)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoint_10.pt")
try:
    rl_agent = FastInferenceAgent(MODEL_PATH)
    print(f"[OK] RL Model loaded successfully from {MODEL_PATH}")
    USE_RL = True
except Exception as e:
    print(f"[ERROR] Failed to load RL model: {e}")
    print("  Falling back to simple heuristic")
    USE_RL = False


@app.route("/", methods=["GET"])
def info():
    """Health check endpoint"""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update local game state from judge"""
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Receive game state from judge"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Return move decision (MUST complete within 4 seconds!)"""
    start_time = time.time()

    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        current_direction = my_agent.direction

    # ================== RL INFERENCE ==================
    move = "RIGHT"  # Default fallback

    if USE_RL:
        try:
            # Get move from trained neural network
            move = rl_agent.get_move(state, player_number, current_direction)
        except Exception as e:
            print(f"RL inference failed: {e}")
            # Fallback to simple heuristic
            move = simple_heuristic_move(state, player_number, current_direction)
    else:
        # Use simple heuristic if model didn't load
        move = simple_heuristic_move(state, player_number, current_direction)

    # ================== TIMING CHECK ==================
    elapsed = time.time() - start_time
    if elapsed > 3.5:  # Log if we're close to the 4s limit
        print(f"⚠ WARNING: Move took {elapsed:.3f}s (limit: 4.0s)")

    return jsonify({"move": move}), 200


def simple_heuristic_move(state, player_number, current_direction):
    """Fallback heuristic: pick move toward most open space"""
    # This is a very simple heuristic - just picks a random valid direction
    # You can replace this with your teammate's heuristic if RL fails

    valid_moves = ["UP", "DOWN", "LEFT", "RIGHT"]

    # Remove opposite direction
    opposites = {
        Direction.UP: "DOWN",
        Direction.DOWN: "UP",
        Direction.LEFT: "RIGHT",
        Direction.RIGHT: "LEFT",
    }

    if current_direction in opposites:
        opposite = opposites[current_direction]
        if opposite in valid_moves:
            valid_moves.remove(opposite)

    # Just return first valid move (very simple!)
    return valid_moves[0] if valid_moves else "RIGHT"


@app.route("/end", methods=["POST"])
def end_game():
    """Game over notification"""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting agent on port {port}")
    print(f"Using RL Model: {USE_RL}")
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False for production
