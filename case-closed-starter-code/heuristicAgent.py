"""
Sample agent for Case Closed Challenge - Maximin Version
This agent runs as a Flask server and responds to judge requests.
It uses a simultaneous-move Minimax algorithm with a Voronoi heuristic.
"""

import os
import uuid
import time
import math
import copy
from collections import deque
from threading import Lock
from flask import Flask, request, jsonify

# Import all the game objects we need
from case_closed_game import Game, Direction, GameResult, Agent, GameBoard

# Flask API server setup
app = Flask(__name__)

# --- Identity ---
PARTICIPANT = os.getenv("PARTICIPANT", "MaximinParticipant")
AGENT_NAME = os.getenv("AGENT_NAME", "MaxminiAgent")

# --- Global State (Upgraded) ---
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 1: FLASK SERVER API
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge."""
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
    """Judge calls this to push the current game state to the agent server."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + YOUR PROVIDED "BRAIN" CODE (UNCHANGED)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# --- CONSTANTS ---
STRATEGIC_SEARCH_DEPTH = 3

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 1: THE "JUDGE" (VORONOI HEURISTIC)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _run_bfs(start_pos, grid, board_height, board_width):
    """
    Helper for Voronoi. Runs a BFS from a start_pos.
    Returns a 2D grid of distances, respecting torus (wrap-around) logic.
    """
    distances = [[-1 for _ in range(board_width)] for _ in range(board_height)]
    q = deque([(start_pos, 0)])
    distances[start_pos[1]][start_pos[0]] = 0
    
    while q:
        (x, y), dist = q.popleft()
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # Torus (wrap-around) logic
            nx, ny = (x + dx) % board_width, (y + dy) % board_height
            
            # If unvisited AND it's an empty cell
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

    # === BASE CASE 1: Game Over ===
    # Check for crashes, which is an infinitely good/bad score.
    if not my_agent.alive:
        return -float('inf')
    if not opp_agent.alive:
        return float('inf')

    # === BASE CASE 2: Territory Score ===
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
                
                if my_d == -1 and opp_d == -1: # Unreachable by both
                    continue
                elif my_d != -1 and (my_d < opp_d or opp_d == -1):
                    my_score += 1
                elif opp_d != -1 and (opp_d < my_d or my_d == -1):
                    opp_score += 1
                # Note: Ties (my_d == opp_d) are not counted for either.
                    
    return my_score - opp_score

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 2: THE "RECURSIVE BRAIN" (STRATEGIC SEARCH)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _get_valid_moves(agent):
    """Helper to get the 3 valid directions (non-reversing)"""
    moves = {
        Direction.UP: "UP", 
        Direction.DOWN: "DOWN", 
        Direction.LEFT: "LEFT", 
        Direction.RIGHT: "RIGHT"
    }
    # This logic prevents 180-degree turns
    current_dir = agent.direction
    if current_dir == Direction.UP:    del moves[Direction.DOWN]
    elif current_dir == Direction.DOWN: del moves[Direction.UP]
    elif current_dir == Direction.LEFT: del moves[Direction.RIGHT]
    elif current_dir == Direction.RIGHT: del moves[Direction.LEFT]
        
    return list(moves.keys()) # Return list of Direction enums

def simulate_simultaneous_move(game_state, my_player_num, my_move_dir, my_boost, opp_move_dir, opp_boost):
    """
    This is the new, correct simulation function.
    It uses the *real* game.step() logic.
    """
    # Use deepcopy to prevent changing the original state
    new_state = copy.deepcopy(game_state) 
    
    # Get agents for boost checking
    my_agent = new_state.agent1 if my_player_num == 1 else new_state.agent2
    opp_agent = new_state.agent2 if my_player_num == 1 else new_state.agent1

    # Check if boosts are legal (agent has them)
    can_my_boost = my_boost and my_agent.boosts_remaining > 0
    can_opp_boost = opp_boost and opp_agent.boosts_remaining > 0

    # Assign moves to the correct player
    if my_player_num == 1:
        p1_dir, p1_boost = my_move_dir, can_my_boost
        p2_dir, p2_boost = opp_move_dir, can_opp_boost
    else:
        p1_dir, p1_boost = opp_move_dir, can_opp_boost
        p2_dir, p2_boost = my_move_dir, can_my_boost

    # Call the "source of truth" step function
    new_state.step(p1_dir, p2_dir, p1_boost, p2_boost)
    
    return new_state

def find_strategic_value(game_state, depth, my_player_num):
    """
    This is the fast, recursive "Strategic Brain".
    It does NOT consider boosts.
    It finds the Maximin value of a board state.
    """
    my_agent = game_state.agent1 if my_player_num == 1 else game_state.agent2
    opp_agent = game_state.agent2 if my_player_num == 1 else game_state.agent1

    # --- BASE CASE ---
    if depth == 0 or not my_agent.alive or not opp_agent.alive:
        return calculate_voronoi_heuristic(game_state, my_player_num)

    # --- RECURSIVE STEP ---
    my_simple_moves = _get_valid_moves(my_agent)
    opp_simple_moves = _get_valid_moves(opp_agent)

    best_score_for_me = -float('inf') # My (Maximizer) goal

    for my_dir in my_simple_moves:
        worst_score_from_this_move = float('inf') # Opponent's (Minimizer) goal
        
        # Failsafe if opponent is trapped
        if not opp_simple_moves:
            opp_simple_moves = [opp_agent.direction]

        for opp_dir in opp_simple_moves:
            # Simulate this (fast) non-boost move pair
            sim_state = simulate_simultaneous_move(game_state, my_player_num, my_dir, False, opp_dir, False)
            
            # Recurse
            score = find_strategic_value(sim_state, depth - 1, my_player_num)
            
            # Opponent will pick the move that minimizes my score
            worst_score_from_this_move = min(worst_score_from_this_move, score)
        
        # I will pick the move that maximizes my worst-case score
        best_score_for_me = max(best_score_for_me, worst_score_from_this_move)

    return best_score_for_me

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 3: THE "ROOT" FUNCTION (TACTICAL SEARCH)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

def find_best_move(current_game_state, my_player_num):
    """
    This is the "master" function that agent.py calls.
    It runs the "Tactical" search (1-ply, boost-enabled)
    which then calls the "Strategic" search.
    """
    my_agent = current_game_state.agent1 if my_player_num == 1 else current_game_state.agent2
    opp_agent = current_game_state.agent2 if my_player_num == 1 else current_game_state.agent1

    # 1. Get *my* 3 or 6 "job" moves (dir, boost_flag)
    my_move_jobs = []
    my_valid_dirs = _get_valid_moves(my_agent)
    for d in my_valid_dirs: my_move_jobs.append((d, False))
    if my_agent.boosts_remaining > 0: # Check for boosts
        for d in my_valid_dirs: my_move_jobs.append((d, True))

    # 2. Get *opponent's* 3 or 6 "job" moves
    opp_move_jobs = []
    opp_valid_dirs = _get_valid_moves(opp_agent)
    # Failsafe if opponent is trapped
    if not opp_valid_dirs: opp_valid_dirs = [opp_agent.direction]
    for d in opp_valid_dirs: opp_move_jobs.append((d, False))
    if opp_agent.boosts_remaining > 0: # Check for boosts
        for d in opp_valid_dirs: opp_move_jobs.append((d, True))

    best_final_score = -float('inf')
    best_move_tuple = my_move_jobs[0] if my_move_jobs else (Direction.UP, False) # Failsafe

    # Loop through each of *my* 3 or 6 possible moves
    for (my_dir, my_boost) in my_move_jobs:
        
        worst_case_score = float('inf') # Find opponent's best reply
        
        # Loop through all *opponent's* 3 or 6 replies
        for (opp_dir, opp_boost) in opp_move_jobs:
            
            # 3. Simulate the TACTICAL (boost-enabled) simultaneous move
            sim_state = simulate_simultaneous_move(current_game_state, my_player_num, my_dir, my_boost, opp_dir, opp_boost)
            
            # 4. Call the STRATEGIC (non-boost) brain for the deep future
            score = find_strategic_value(sim_state, STRATEGIC_SEARCH_DEPTH - 1, my_player_num)
            
            # Opponent will pick the *minimum* score
            worst_case_score = min(worst_case_score, score)

        # 5. I (the maximizer) will pick the move that gives
        # the *maximum* of all the "worst-case" scores.
        if worst_case_score > best_final_score:
            best_final_score = worst_case_score
            best_move_tuple = (my_dir, my_boost)
            
    # 6. Convert the winning tuple (e.g., (Direction.UP, True)) to a string
    (final_dir, final_boost) = best_move_tuple
    
    # Convert Direction enum to string
    move_str = "UP" # Failsafe
    if final_dir == Direction.UP:    move_str = "UP"
    if final_dir == Direction.DOWN:  move_str = "DOWN"
    if final_dir == Direction.LEFT:  move_str = "LEFT"
    if final_dir == Direction.RIGHT: move_str = "RIGHT"
    
    if final_boost:
        return f"{move_str}:BOOST"
    else:
        return move_str

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 4: THE "SEND_MOVE" ENDPOINT
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move."""
    player_number = request.args.get("player_number", default=1, type=int)
    
    # Get a safe copy of the *current* game state
    with game_lock:
        game_copy = copy.deepcopy(GLOBAL_GAME) 
    
    # Call your brain logic
    move = find_best_move(game_copy, player_number)
    
    return jsonify({"move": move}), 200

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++
# + SECTION 5: RUN THE SERVER
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    # For development only. Port can be overridden with the PORT env var.
    # The sample agent MUST run on a different port (e.g., 5009)
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)