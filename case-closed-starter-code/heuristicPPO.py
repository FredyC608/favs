import os
import torch
import torch.nn as nn
import time
import math
import copy
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

# --- PPO Model Imports ---
from model import StandalonePpoPolicy 

# --- Heuristic Model Imports ---
from case_closed_game import Game, Direction, GameResult, Agent, GameBoard

# --- Constants ---
# The turn count on which to switch from the Heuristic brain to the PPO brain.
# 0 = PPO Only
# 30 = Heuristic for 30 turns, then PPO
PPO_END_TIME = 25 

BOARD_HEIGHT = 18
BOARD_WIDTH = 20

# --- Agent/Participant Info ---
PARTICIPANT = "HybridParticipant"
AGENT_NAME = "HeuristicPPO_Agent"

# --- Global State ---
app = Flask(__name__)
# GLOBAL_GAME is used by the Heuristic brain
GLOBAL_GAME = Game()
# LAST_POSTED_STATE is used by the PPO brain
LAST_POSTED_STATE = {}
game_lock = Lock()

# --- Model Loading (From agent.py) ---
DEVICE = torch.device("cpu") # Judge runs on CPU
try:
    # Load the trained policy
    # The '.pth' file must be in the same directory
    policy = StandalonePpoPolicy().to(DEVICE)
    policy.load_state_dict(torch.load("ppo_policy_update_120.pth", map_location=DEVICE))
    policy.eval()
    print("--- PPO Policy loaded successfully ---")
except Exception as e:
    print(f"--- FATAL: Could not load PPO model 'ppo_policy_update_440.pth' ---")
    print(e)
    policy = None 

# --- PPO Brain: Action Translation Maps (From agent.py) ---
ABS_DIR_MAP = {
    (0, -1): "UP",
    (0, 1): "DOWN",
    (-1, 0): "LEFT",
    (1, 0): "RIGHT",
}
RELATIVE_MAP_PY = {
    0: {0: (0, -1), 1: (-1, 0), 2: (1, 0)}, # Facing UP
    1: {0: (0, 1),  1: (1, 0),  2: (-1, 0)}, # Facing DOWN
    2: {0: (-1, 0), 1: (0, 1),  2: (0, -1)}, # Facing LEFT
    3: {0: (1, 0),  1: (0, -1), 2: (0, 1)},  # Facing RIGHT
}
DIR_TO_IDX = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}

# --- Heuristic Brain: Search Depth (From heuristicAgent.py) ---
STRATEGIC_SEARCH_DEPTH = 6

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 1: FLASK SERVER API
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

def _update_local_game_from_post(data: dict):
    """
    Update the local GLOBAL_GAME using the JSON posted by the judge.
    This is required for the Heuristic brain.
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        # This syncs the GLOBAL_GAME object for the heuristic to use
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
    """Judge calls this to push the current game state."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    # Update both the dict (for PPO) and the object (for Heuristic)
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 2: PPO "BRAIN" (From agent.py)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    obs[0, 4] = norm_boosts
        
    return obs

def get_ppo_move(state, player_number):
    """
    Runs the PPO model inference.
    """
    # 1. Re-create the 5-channel observation tensor
    obs_tensor = build_obs_from_state(state, player_number)

    # 2. Get action from your policy
    with torch.no_grad():
        print(f">>> DEBUG P{player_number}: Calling policy...")
        logits, _ = policy(obs_tensor.to(DEVICE))
        
        print(f">>> DEBUG P{player_number}: Model raw logits: {logits}")
        action_idx = logits.argmax().item() # Greedy action, e.g., 4
        print(f">>> DEBUG P{player_number}: Model chose action_idx: {action_idx}")

    # 3. Parse the action index (e.g., 4)
    rel_dir_idx = action_idx % 3  # (0=F, 1=L, 2=R)
    use_boost = (action_idx >= 3)

    # 4. Get current absolute direction (from trail)
    if player_number == 1:
        my_trail = state.get("agent1_trail", [])
    else:
        my_trail = state.get("agent2_trail", [])

    current_dx, current_dy = (1, 0) # Default
    if len(my_trail) >= 2:
        head = my_trail[-1]
        prev = my_trail[-2]
        dx = (head[0] - prev[0])
        dy = (head[1] - prev[1])
        if abs(dx) > 1: dx = -int(dx / abs(dx))
        if abs(dy) > 1: dy = -int(dy / abs(dy))
        current_dx, current_dy = int(dx), int(dy)

    # 5. Translate (Relative + Absolute) -> New Absolute
    current_dir_idx = DIR_TO_IDX.get((current_dx, current_dy), 3) # Default to 3 (RIGHT)
    new_dx, new_dy = RELATIVE_MAP_PY[current_dir_idx][rel_dir_idx]
    
    # 6. Format the final move string
    move_str = ABS_DIR_MAP.get((new_dx, new_dy), "RIGHT")
    
    if use_boost:
        my_boosts = state.get(f"agent{player_number}_boosts", 0)
        if my_boosts > 0:
            move = f"{move_str}:BOOST"
        else:
            move = move_str # Can't boost
    else:
        move = move_str
        
    print(f">>> DEBUG P{player_number}: Current dir ({current_dx}, {current_dy}). Relative action {rel_dir_idx}. Final move: {move}")
    return move

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 3: HEURISTIC "BRAIN" (From heuristicAgent.py)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _run_bfs(start_pos, grid, board_height, board_width):
    """
    Helper for Voronoi. Runs a BFS from a start_pos.
    """
    distances = [[-1 for _ in range(board_width)] for _ in range(board_height)]
    q = deque([(start_pos, 0)])
    distances[start_pos[1]][start_pos[0]] = 0
    
    while q:
        (x, y), dist = q.popleft()
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = (x + dx) % board_width, (y + dy) % board_height
            
            if distances[ny][nx] == -1 and grid[ny][nx] == 0:
                distances[ny][nx] = dist + 1
                q.append(((nx, ny), dist + 1))
    return distances

def calculate_voronoi_heuristic(game_state, my_player_num):
    """
    This is the "Judge" or "Base Case" evaluation function.
    It calculates (my_territory - opponent_territory).
    """
    my_agent = game_state.agent1 if my_player_num == 1 else game_state.agent2
    opp_agent = game_state.agent2 if my_player_num == 1 else game_state.agent1

    if not my_agent.alive: return -float('inf')
    if not opp_agent.alive: return float('inf')

    my_head = my_agent.trail[-1]
    opp_head = opp_agent.trail[-1]
    
    grid = game_state.board.grid
    h = game_state.board.height
    w = game_state.board.width
    
    my_dist = _run_bfs(my_head, grid, h, w)
    opp_dist = _run_bfs(opp_head, grid, h, w)
    
    my_score = 0
    opp_score = 0
    
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 0: # Only count empty squares
                my_d = my_dist[y][x]
                opp_d = opp_dist[y][x]
                
                if my_d == -1 and opp_d == -1: continue
                elif my_d != -1 and (my_d < opp_d or opp_d == -1):
                    my_score += 1
                elif opp_d != -1 and (opp_d < my_d or my_d == -1):
                    opp_score += 1
                    
    return my_score - opp_score

def _get_valid_moves(agent):
    """Helper to get the 3 valid directions (non-reversing)"""
    moves = {
        Direction.UP: "UP", 
        Direction.DOWN: "DOWN", 
        Direction.LEFT: "LEFT", 
        Direction.RIGHT: "RIGHT"
    }
    current_dir = agent.direction
    if current_dir == Direction.UP:    del moves[Direction.DOWN]
    elif current_dir == Direction.DOWN: del moves[Direction.UP]
    elif current_dir == Direction.LEFT: del moves[Direction.RIGHT]
    elif current_dir == Direction.RIGHT: del moves[Direction.LEFT]
        
    return list(moves.keys()) # Return list of Direction enums

def simulate_simultaneous_move(game_state, my_player_num, my_move_dir, my_boost, opp_move_dir, opp_boost):
    """
    This is the simulation function for the heuristic.
    It uses the *real* game.step() logic on a deep copy.
    """
    new_state = copy.deepcopy(game_state) 
    
    my_agent = new_state.agent1 if my_player_num == 1 else new_state.agent2
    opp_agent = new_state.agent2 if my_player_num == 1 else new_state.agent1

    can_my_boost = my_boost and my_agent.boosts_remaining > 0
    can_opp_boost = opp_boost and opp_agent.boosts_remaining > 0

    if my_player_num == 1:
        p1_dir, p1_boost = my_move_dir, can_my_boost
        p2_dir, p2_boost = opp_move_dir, can_opp_boost
    else:
        p1_dir, p1_boost = opp_move_dir, can_opp_boost
        p2_dir, p2_boost = my_move_dir, can_my_boost

    new_state.step(p1_dir, p2_dir, p1_boost, p2_boost)
    return new_state

def find_strategic_value(game_state, depth, my_player_num):
    """
    This is the fast, recursive "Strategic Brain".
    It does NOT consider boosts.
    """
    my_agent = game_state.agent1 if my_player_num == 1 else game_state.agent2
    opp_agent = game_state.agent2 if my_player_num == 1 else game_state.agent1

    # --- BASE CASE ---
    if depth == 0 or not my_agent.alive or not opp_agent.alive:
        return calculate_voronoi_heuristic(game_state, my_player_num)

    # --- RECURSIVE STEP ---
    my_simple_moves = _get_valid_moves(my_agent)
    opp_simple_moves = _get_valid_moves(opp_agent)

    best_score_for_me = -float('inf') 

    for my_dir in my_simple_moves:
        worst_score_from_this_move = float('inf') 
        
        if not opp_simple_moves:
            opp_simple_moves = [opp_agent.direction]

        for opp_dir in opp_simple_moves:
            sim_state = simulate_simultaneous_move(game_state, my_player_num, my_dir, False, opp_dir, False)
            score = find_strategic_value(sim_state, depth - 1, my_player_num)
            worst_score_from_this_move = min(worst_score_from_this_move, score)
        
        best_score_for_me = max(best_score_for_me, worst_score_from_this_move)

    return best_score_for_me

def find_best_move(current_game_state, my_player_num):
    """
    This is the "master" heuristic function.
    It runs the "Tactical" search (1-ply, boost-enabled)
    which then calls the "Strategic" search.
    """
    my_agent = current_game_state.agent1 if my_player_num == 1 else current_game_state.agent2
    opp_agent = current_game_state.agent2 if my_player_num == 1 else current_game_state.agent1

    # 1. Get *my* 3 or 6 "job" moves (dir, boost_flag)
    my_move_jobs = []
    my_valid_dirs = _get_valid_moves(my_agent)
    for d in my_valid_dirs: my_move_jobs.append((d, False))
    if my_agent.boosts_remaining > 0:
        for d in my_valid_dirs: my_move_jobs.append((d, True))

    # 2. Get *opponent's* 3 or 6 "job" moves
    opp_move_jobs = []
    opp_valid_dirs = _get_valid_moves(opp_agent)
    if not opp_valid_dirs: opp_valid_dirs = [opp_agent.direction]
    for d in opp_valid_dirs: opp_move_jobs.append((d, False))
    if opp_agent.boosts_remaining > 0:
        for d in opp_valid_dirs: opp_move_jobs.append((d, True))

    best_final_score = -float('inf')
    best_move_tuple = my_move_jobs[0] if my_move_jobs else (Direction.UP, False)

    for (my_dir, my_boost) in my_move_jobs:
        worst_case_score = float('inf') 
        
        for (opp_dir, opp_boost) in opp_move_jobs:
            # 3. Simulate the TACTICAL (boost-enabled) move
            sim_state = simulate_simultaneous_move(current_game_state, my_player_num, my_dir, my_boost, opp_dir, opp_boost)
            
            # 4. Call the STRATEGIC (non-boost) brain
            score = find_strategic_value(sim_state, STRATEGIC_SEARCH_DEPTH - 1, my_player_num)
            
            worst_case_score = min(worst_case_score, score)

        # 5. I (the maximizer) pick the max of the worst-cases
        if worst_case_score > best_final_score:
            best_final_score = worst_case_score
            best_move_tuple = (my_dir, my_boost)
            
    # 6. Convert the winning tuple to a string
    (final_dir, final_boost) = best_move_tuple
    
    move_str = "UP" 
    if final_dir == Direction.UP:    move_str = "UP"
    if final_dir == Direction.DOWN:  move_str = "DOWN"
    if final_dir == Direction.LEFT:  move_str = "LEFT"
    if final_dir == Direction.RIGHT: move_str = "RIGHT"
    
    if final_boost:
        return f"{move_str}:BOOST"
    else:
        return move_str

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 4: THE "MASTER SWITCH" ENDPOINT
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move."""
    player_number = request.args.get("player_number", default=1, type=int)
    
    with game_lock:
        state = dict(LAST_POSTED_STATE)
    
    if not state:
        return jsonify({"move": "RIGHT"}), 200 # Fallback
    
    # --- THIS IS THE HYBRID LOGIC ---
    current_turn = state.get("turn_count", 0)

    if current_turn > PPO_END_TIME:
        # --- 1. Early Game: Use the Heuristic Brain ---
        print(f">>> DEBUG P{player_number}: Turn {current_turn}. Using HEURISTIC brain.")
        
        # 1. Sync the GLOBAL_GAME object with the dict
        #    (We've already done this in /send-state)
        
        # 2. Get a safe copy (the heuristic modifies its copy)
        with game_lock:
             game_copy = copy.deepcopy(GLOBAL_GAME)
        
        # 3. Call the heuristic brain
        move = find_best_move(game_copy, player_number)
        
    else:
        # --- 2. Mid/End Game: Use the PPO Brain ---
        print(f">>> DEBUG P{player_number}: Turn {current_turn}. Using PPO brain.")
        
        # Check if PPO model is loaded
        if policy is None:
            print(">>> DEBUG: PPO Policy not loaded! Defaulting to RIGHT.")
            return jsonify({"move": "RIGHT"}), 200

        # 1. Call the PPO brain
        move = get_ppo_move(state, player_number)
    
    return jsonify({"move": move}), 200


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 5: RUN THE SERVER
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    print(f"--- Starting {AGENT_NAME} ({PARTICIPANT}) on port {port} ---")
    app.run(host="0.0.0.0", port=port, debug=False)