import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

# --- Local Imports ---
# You MUST copy your model.py file into the same directory as this agent.py
# so that this import works.
from model import StandalonePpoPolicy 

# --- Constants ---
# These must match your training setup
BOARD_HEIGHT = 18
BOARD_WIDTH = 20

# --- Agent/Participant Info ---
PARTICIPANT = "ParticipantX" # TODO: Change this
AGENT_NAME = "PPO_Agent"     # TODO: Change this

# --- Global State ---
app = Flask(__name__)
GLOBAL_GAME = None  # We don't use GLOBAL_GAME, we use LAST_POSTED_STATE
LAST_POSTED_STATE = {}
game_lock = Lock()

# --- Model Loading ---
DEVICE = torch.device("cpu") # Judge runs on CPU
try:
    # Load the trained policy
    # The 'ppo_policy_weights.pth' file must be in the same directory
    policy = StandalonePpoPolicy().to(DEVICE)
    policy.load_state_dict(torch.load("ppo_policy_weights.pth", map_location=DEVICE))
    policy.eval()
    print("--- PPO Policy loaded successfully ---")
except Exception as e:
    print(f"--- FATAL: Could not load model 'ppo_policy_weights.pth' ---")
    print(e)
    # Create a dummy policy if load fails, so server can start
    policy = None 

# --- Action Translation Maps ---
# These maps translate your model's output into the judge's required format
# (dx, dy) -> "DIRECTION_STR"
ABS_DIR_MAP = {
    (0, -1): "UP",
    (0, 1): "DOWN",
    (-1, 0): "LEFT",
    (1, 0): "RIGHT",
}

# (current_dir_idx, rel_action_idx) -> (new_dx, new_dy)
# This MUST match your utils.py
RELATIVE_MAP_PY = {
    0: {0: (0, -1), 1: (-1, 0), 2: (1, 0)}, # Facing UP
    1: {0: (0, 1),  1: (1, 0),  2: (-1, 0)}, # Facing DOWN
    2: {0: (-1, 0), 1: (0, 1),  2: (0, -1)}, # Facing LEFT
    3: {0: (1, 0),  1: (0, -1), 2: (0, 1)},  # Facing RIGHT
}
# (dx, dy) -> current_dir_idx
DIR_TO_IDX = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}

# --- Flask API Endpoints ---

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)
        
    return jsonify({"status": "state received"}), 200

@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished."""
    return jsonify({"status": "acknowledged"}), 200

@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move."""
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
    
    if not state or policy is None:
        # --- DEBUG 1: CHECK FOR DEFAULT ---
        print(">>> DEBUG: Policy or state is NONE. Returning default move 'RIGHT'.")
        return jsonify({"move": "RIGHT"}), 200
        
    # -----------------YOUR PPO LOGIC HERE-------------------
    
    # 1. Re-create the 5-channel observation tensor
    obs_tensor = build_obs_from_state(state, player_number)

    # 2. Get action from your policy
    with torch.no_grad():
        # --- DEBUG 2: CONFIRM POLICY IS CALLED ---
        print(f">>> DEBUG P{player_number}: Calling policy...")
        logits, _ = policy(obs_tensor.to(DEVICE))
        
        # --- DEBUG 3: SHOW MODEL'S RAW OUTPUT ---
        print(f">>> DEBUG P{player_number}: Model raw logits: {logits}")
        action_idx = logits.argmax().item() # Greedy action, e.g., 4
        print(f">>> DEBUG P{player_number}: Model chose action_idx: {action_idx}")


    # 3. Parse the action index (e.g., 4)
    rel_dir_idx = action_idx % 3  # (0=F, 1=L, 2=R), e.g., 1 (Left)
    use_boost = (action_idx >= 3) # e.g., True

    # 4. Get current absolute direction (from trail)
    if player_number == 1:
        my_trail = state.get("agent1_trail", [])
    else:
        my_trail = state.get("agent2_trail", [])

    # Default to RIGHT if trail is too short
    current_dx, current_dy = (1, 0) 
    if len(my_trail) >= 2:
        head = my_trail[-1]
        prev = my_trail[-2]
        # Handle torus wrap
        dx = (head[0] - prev[0])
        dy = (head[1] - prev[1])
        if abs(dx) > 1: dx = -int(dx / abs(dx)) # Normalize to -1 or 1
        if abs(dy) > 1: dy = -int(dy / abs(dy)) # Normalize to -1 or 1
        current_dx, current_dy = int(dx), int(dy)

    # 5. Translate (Relative + Absolute) -> New Absolute
    current_dir_idx = DIR_TO_IDX.get((current_dx, current_dy), 3) # Default to 3 (RIGHT)
    new_dx, new_dy = RELATIVE_MAP_PY[current_dir_idx][rel_dir_idx]
    
    # 6. Format the final move string
    move_str = ABS_DIR_MAP.get((new_dx, new_dy), "RIGHT")
    
    if use_boost:
        # Final check: do we actually have boosts?
        my_boosts = state.get(f"agent{player_number}_boosts", 0)
        if my_boosts > 0:
            move = f"{move_str}:BOOST"
        else:
            move = move_str # Can't boost, send regular move
    else:
        move = move_str
        
    # -----------------END PPO LOGIC--------------------

    # --- DEBUG 4: SHOW THE FINAL TRANSLATED MOVE ---
    print(f">>> DEBUG P{player_number}: Current dir ({current_dx}, {current_dy}). Relative action {rel_dir_idx} ('Forward'). Final move: {move}")
    return jsonify({"move": move}), 200

# --- THE KEY HELPER FUNCTION ---

def build_obs_from_state(state: dict, player_number: int) -> torch.Tensor:
    """
    Creates the (1, 5, H, W) state tensor from the judge's JSON state.
    This logic mirrors _get_obs from batched_env.py.
    """
    obs = torch.zeros((1, 5, BOARD_HEIGHT, BOARD_WIDTH), dtype=torch.float32)

    # 1. Assign "my" and "opponent" based on player_number
    if player_number == 1:
        my_trail = state.get("agent1_trail", [])
        opp_trail = state.get("agent2_trail", [])
        my_alive = state.get("agent1_alive", True)
        opp_alive = state.get("agent2_alive", True)
        my_boosts = state.get("agent1_boosts", 0)
    else:
        my_trail = state.get("agent2_trail", [])
        opp_trail = state.get("agent1_trail", [])
        my_alive = state.get("agent2_alive", True)
        opp_alive = state.get("agent1_alive", True)
        my_boosts = state.get("agent2_boosts", 0)

    # 2. Draw trails (Channels 0 & 1)
    # Channel 0: My Trail
    for x, y in my_trail:
        obs[0, 0, y % BOARD_HEIGHT, x % BOARD_WIDTH] = 1.0

    # Channel 1: Opponent's Trail
    for x, y in opp_trail:
        obs[0, 1, y % BOARD_HEIGHT, x % BOARD_WIDTH] = 1.0
        
    # 3. Draw heads (Channels 2 & 3)
    # Channel 2: My Head
    if my_alive and my_trail:
        my_head = my_trail[-1]
        obs[0, 2, my_head[1] % BOARD_HEIGHT, my_head[0] % BOARD_WIDTH] = 1.0
            
    # Channel 3: Opponent's Head
    if opp_alive and opp_trail:
        opp_head = opp_trail[-1]
        obs[0, 3, opp_head[1] % BOARD_HEIGHT, opp_head[0] % BOARD_WIDTH] = 1.0

    # 4. Add boost plane (Channel 4)
    # Normalized boost count (0.0 to 1.0)
    norm_boosts = float(max(my_boosts, 0)) / 3.0
    obs[0, 4] = norm_boosts  # This broadcasts to the whole (H, W) plane
        
    return obs

# --- Main Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print(f"--- Starting {AGENT_NAME} ({PARTICIPANT}) on port {port} ---")
    app.run(host="0.0.0.0", port=port, debug=False)