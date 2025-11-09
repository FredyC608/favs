import os
import time
import copy
from collections import deque
from threading import Lock
from flask import Flask, request, jsonify

# We still import these for reference, though we use a custom state dict for speed
from case_closed_game import Game, Direction

# Flask API server setup
app = Flask(__name__)

# Global game state, updated by the judge
LAST_POSTED_STATE = {}
game_lock = Lock()

PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX_Voronoi_Minimax"

# --- FLASK STANDARD ENDPOINTS ---

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge pushes the current game state. We just store it."""
    data = request.get_json()
    if data:
        with game_lock:
            LAST_POSTED_STATE.clear()
            LAST_POSTED_STATE.update(data)
    return jsonify({"status": "state received"}), 200

@app.route("/end", methods=["POST"])
def end_game():
    """Game over notification."""
    return jsonify({"status": "acknowledged"}), 200

# --- AI ENGINE: VORONOI MINIMAX ---

def initialize_search_state(raw_state):
    """
    Converts raw JSON state into a clean format optimized for our search.
    Crucially, it converts coordinate lists [x,y] into tuples (x,y) 
    so they can be used in sets for the Voronoi BFS.
    """
    # Default empty board if missing (18x20 standard)
    board = raw_state.get("board", [[0]*20 for _ in range(18)])
    
    state = {
        'board': [row[:] for row in board], # Deep copy grid
        'agent1_trail': [tuple(p) for p in raw_state.get("agent1_trail", [])],
        'agent2_trail': [tuple(p) for p in raw_state.get("agent2_trail", [])],
        'agent1_alive': bool(raw_state.get("agent1_alive", True)),
        'agent2_alive': bool(raw_state.get("agent2_alive", True)),
        'agent1_boosts': int(raw_state.get("agent1_boosts", 3)),
        'agent2_boosts': int(raw_state.get("agent2_boosts", 3)),
        'agent1_length': int(raw_state.get("agent1_length", 0)),
        'agent2_length': int(raw_state.get("agent2_length", 0)),
        'turn': int(raw_state.get("turn_count", 0))
    }
    return state

def get_possible_moves_including_boosts(state, player_number):
    """Returns valid moves (UP, DOWN, LEFT, RIGHT) plus BOOST variants."""
    trail = state[f'agent{player_number}_trail']
    boosts = state[f'agent{player_number}_boosts']
    
    # Default direction if no trail yet
    current_direction = "UP"
    
    # Determine current direction to avoid 180s
    if len(trail) >= 2:
        head = trail[-1]
        neck = trail[-2]
        dx, dy = head[0] - neck[0], head[1] - neck[1]
        
        # Handle torus wrap detection
        if dx > 1: dx = -1
        elif dx < -1: dx = 1
        if dy > 1: dy = -1
        elif dy < -1: dy = 1
        
        if (dx, dy) == (0, -1): current_direction = "UP"
        elif (dx, dy) == (0, 1): current_direction = "DOWN"
        elif (dx, dy) == (-1, 0): current_direction = "LEFT"
        elif (dx, dy) == (1, 0): current_direction = "RIGHT"

    opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    possible_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    if current_direction in opposites:
        possible_moves.remove(opposites[current_direction])
        
    # Add boost options if available
    if boosts > 0:
        boost_moves = [f"{m}:BOOST" for m in possible_moves]
        possible_moves.extend(boost_moves)
        
    # Optimization: Shuffle to add variety if scores are equal
    import random
    random.shuffle(possible_moves)
    return possible_moves

def apply_and_log_tick(state, move1, move2):
    """
    Applies a simultaneous move for both players to 'state' IN-PLACE.
    Returns a 'changes' dict specifically designed to UNDO this exact tick.
    """
    changes = {
        'p1_trail_added': None, 'p2_trail_added': None, # Stores the pos added
        'p1_boost_used': False, 'p2_boost_used': False,
        'p1_died_this_tick': False, 'p2_died_this_tick': False,
        'grid_writes': [] # List of (x, y, old_val) to revert board
    }

    height = len(state['board'])
    width = len(state['board'][0])
    
    # Pre-calculate next positions to handle simultaneous head-on collisions
    next_heads = {}
    
    for p_num, move in [(1, move1), (2, move2)]:
        if not state[f'agent{p_num}_alive']: continue
        
        parts = move.split(':')
        direction = parts[0]
        is_boost = (len(parts) > 1 and parts[1] == 'BOOST')
        
        # Consumable checks
        if is_boost and state[f'agent{p_num}_boosts'] > 0:
            state[f'agent{p_num}_boosts'] -= 1
            changes[f'p{p_num}_boost_used'] = True
            steps = 2
        else:
            steps = 1
            
        dx, dy = {'UP':(0,-1), 'DOWN':(0,1), 'LEFT':(-1,0), 'RIGHT':(1,0)}[direction]
        curr_pos = state[f'agent{p_num}_trail'][-1]
        
        # Move step-by-step to check for wall hits mid-boost
        crashed = False
        for _ in range(steps):
            curr_pos = ((curr_pos[0] + dx) % width, (curr_pos[1] + dy) % height)
            # Check standard wall collision
            if state['board'][curr_pos[1]][curr_pos[0]] != 0:
                crashed = True
                break
        
        if crashed:
            state[f'agent{p_num}_alive'] = False
            changes[f'p{p_num}_died_this_tick'] = True
        else:
            next_heads[p_num] = curr_pos

    # Handle Head-on Head Collisions (Simultaneous same square)
    if 1 in next_heads and 2 in next_heads:
        if next_heads[1] == next_heads[2]:
            # Both crash into same square
            state['agent1_alive'] = False
            state['agent2_alive'] = False
            changes['p1_died_this_tick'] = True
            changes['p2_died_this_tick'] = True
            # Remove from valid next heads so they don't draw trails there
            del next_heads[1]
            del next_heads[2]
        # Handle "Swap" collision (moving through each other)
        elif next_heads[1] == state['agent2_trail'][-1] and next_heads[2] == state['agent1_trail'][-1]:
             state['agent1_alive'] = False
             state['agent2_alive'] = False
             changes['p1_died_this_tick'] = True
             changes['p2_died_this_tick'] = True
             del next_heads[1]
             del next_heads[2]

    # Apply confirmed moves to board and trails
    for p_num, head in next_heads.items():
        # Save old board state for undo (should be 0, but good practice)
        old_val = state['board'][head[1]][head[0]]
        changes['grid_writes'].append((head[0], head[1], old_val))
        
        state['board'][head[1]][head[0]] = 1 # Mark occupied
        state[f'agent{p_num}_trail'].append(head)
        state[f'agent{p_num}_length'] += 1
        changes[f'p{p_num}_trail_added'] = head

    return changes

def undo_tick(state, changes):
    """Reverts the state using the changes log."""
    # 1. Revert grid writes
    for (x, y, val) in reversed(changes['grid_writes']):
        state['board'][y][x] = val
        
    # 2. Revert trails and lengths
    if changes['p1_trail_added']:
        state['agent1_trail'].pop()
        state['agent1_length'] -= 1
    if changes['p2_trail_added']:
        state['agent2_trail'].pop()
        state['agent2_length'] -= 1
        
    # 3. Revert life status
    if changes['p1_died_this_tick']: state['agent1_alive'] = True
    if changes['p2_died_this_tick']: state['agent2_alive'] = True
    
    # 4. Revert boosts
    if changes['p1_boost_used']: state['agent1_boosts'] += 1
    if changes['p2_boost_used']: state['agent2_boosts'] += 1

def calculate_voronoi_score(state, my_player_num):
    """
    BFS flood fill from both agents simultaneously to determine territory control.
    Higher score = more territory controlled by 'my_player_num'.
    """
    p1_alive = state['agent1_alive']
    p2_alive = state['agent2_alive']

    # Game Over Terminal States
    if not p1_alive and not p2_alive: return 0 # Draw
    if my_player_num == 1 and not p1_alive: return -10000 # I lost
    if my_player_num == 2 and not p2_alive: return -10000 # I lost
    if my_player_num == 1 and not p2_alive: return 10000  # I won
    if my_player_num == 2 and not p1_alive: return 10000  # I won

    # Both alive, run Voronoi
    board = state['board']
    h, w = len(board), len(board[0])
    
    q = deque()
    visited = set()
    
    p1_head = state['agent1_trail'][-1]
    p2_head = state['agent2_trail'][-1]
    
    q.append((p1_head, 1, 0)) # (pos, owner, distance)
    q.append((p2_head, 2, 0))
    visited.add(p1_head)
    visited.add(p2_head)
    
    p1_territory = 0
    p2_territory = 0
    
    while q:
        curr_pos, owner, dist = q.popleft()
        
        cx, cy = curr_pos
        # Explore neighbors
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = (cx + dx) % w, (cy + dy) % h
            if (nx, ny) not in visited and board[ny][nx] == 0:
                visited.add((nx, ny))
                if owner == 1: p1_territory += 1
                else: p2_territory += 1
                q.append(((nx, ny), owner, dist + 1))

    if my_player_num == 1:
        return p1_territory - p2_territory
    else:
        return p2_territory - p1_territory

def alpha_beta_search(state, depth, alpha, beta, is_maximizing, my_p_num):
    """Recursive Minimax with Alpha-Beta Pruning."""
    # 1. Check terminal state or depth limit
    if depth == 0 or not state['agent1_alive'] or not state['agent2_alive']:
        return calculate_voronoi_score(state, my_p_num)

    # 2. Determine whose turn it is conceptually (simultaneous in reality, 
    # but we model it as Max choosing their move, then Min choosing theirs 
    # to find the worst-case outcome for Max's choice).
    
    # Simplification for simultaneous games in Minimax:
    # We assume WE choose a move, and the OPPONENT simultaneously chooses 
    # the move that minimizes our score.
    
    if is_maximizing:
        best_score = -float('inf')
        possible_moves = get_possible_moves_including_boosts(state, my_p_num)
        
        # Heuristic sorting could go here to improve pruning
        
        for my_move in possible_moves:
            # We need to conceptually 'wait' for the opponent's simultaneous move.
            # In standard minimax for simultaneous games, this is complex.
            # A common simplification: We commit to 'my_move', then see what 
            # the BEST counter-move (minimizing node) for the opponent is.
             
            # Pass committed move down to minimizer
            score = alpha_beta_simul_min_turn(state, depth, alpha, beta, my_p_num, my_move_committed=my_move)
            
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break # Beta cutoff
        return best_score
        
def alpha_beta_simul_min_turn(state, depth, alpha, beta, my_p_num, my_move_committed):
    """
    The 'Minimizer' turn in this simultaneous model.
    The maximizer has committed to 'my_move_committed'.
    Now we find the worst-case scenario: what if the opponent guesses perfectly?
    """
    op_num = 3 - my_p_num
    op_moves = get_possible_moves_including_boosts(state, op_num)
    min_score = float('inf')
    
    for op_move in op_moves:
        # Resolve simultaneous tick
        move1 = my_move_committed if my_p_num == 1 else op_move
        move2 = op_move if my_p_num == 1 else my_move_committed
        
        changes = apply_and_log_tick(state, move1, move2)
        
        # Next turn, back to maximizer
        score = alpha_beta_search(state, depth - 1, alpha, beta, True, my_p_num)
        
        undo_tick(state, changes)
        
        min_score = min(min_score, score)
        beta = min(beta, min_score)
        if beta <= alpha:
            break # Alpha cutoff
            
    return min_score

def decide_best_move(raw_state, my_player_number):
    """Root of the search tree."""
    search_state = initialize_search_state(raw_state)
    
    possible_moves = get_possible_moves_including_boosts(search_state, my_player_number)
    if not possible_moves: return "UP" # Should not happen if alive
    
    best_move = possible_moves[0]
    best_score = -float('inf')
    
    # Depth can be adjusted based on performance.
    # 4 is usually safe for Python in <200ms if branching factor isn't insane.
    SEARCH_DEPTH = 4
    alpha = -float('inf')
    beta = float('inf')
    
    for move in possible_moves:
        # For the root, we essentially run the 'min_turn' for each of our possible moves
        score = alpha_beta_simul_min_turn(search_state, SEARCH_DEPTH, alpha, beta, my_player_number, move)
        
        print(f"Move {move} eval: {score}") # Debug logging
        
        if score > best_score:
            best_score = score
            best_move = move
        
        # Root level alpha update
        alpha = max(alpha, best_score)

    return best_move

# --- MAIN MOVE HANDLER ---

@app.route("/send-move", methods=["GET"])
def send_move():
    """Main agent logic entry point."""
    player_number = request.args.get("player_number", default=1, type=int)
    
    with game_lock:
        # Snapshot the current state safely
        current_state_snapshot = copy.deepcopy(LAST_POSTED_STATE)

    if not current_state_snapshot:
        return jsonify({"move": "UP"}), 200

    try:
        t0 = time.time()
        chosen_move = decide_best_move(current_state_snapshot, player_number)
        duration = (time.time() - t0) * 1000
        print(f" Turn {current_state_snapshot.get('turn_count')}: Chose {chosen_move} in {duration:.2f}ms")
        return jsonify({"move": chosen_move}), 200
    except Exception as e:
        print(f"ERROR in decide_move: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to a safe simple move if we crash
        return jsonify({"move": "UP"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)