import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
# In agent.py, add this import at the top
import copy

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
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
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

# --- BEGIN: AI STRATEGY IMPLEMENTATION ---

# In agent.py, add these two new helper functions

# In agent.py, add these two new helper functions

def apply_and_log_tick(state, move1, move2):
    """
    Modifies the state dictionary IN-PLACE and returns a change log
    dictionary for undoing the move.
    """
    changes = {
        'p1_trail_added': False, 'p2_trail_added': False,
        'p1_boost_used': False, 'p2_boost_used': False,
        'p1_became_dead': False, 'p2_became_dead': False,
        'grid_cells_changed': []
    }
    
    board = state['board']
    height, width = len(board), len(board[0])
    
    players = [
        {'num': 1, 'move': move1, 'trail': state['agent1_trail'], 'boosts': state['agent1_boosts']},
        {'num': 2, 'move': move2, 'trail': state['agent2_trail'], 'boosts': state['agent2_boosts']}
    ]
    new_heads = {}
    
    # Phase 1: Calculate new head positions
    for p in players:
        # Save original alive status to detect changes
        was_alive = state[f'agent{p["num"]}_alive']

        parts = p['move'].split(':')
        direction_str = parts[0]
        use_boost = len(parts) > 1 and parts[1] == 'BOOST'
        
        moves = 2 if use_boost and p['boosts'] > 0 else 1
        if use_boost and p['boosts'] > 0:
            state[f'agent{p["num"]}_boosts'] -= 1
            changes[f'p{p["num"]}_boost_used'] = True

        d_map = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
        dx, dy = d_map[direction_str]

        new_pos = p['trail'][-1]
        for _ in range(moves):
            new_pos = ((new_pos[0] + dx) % width, (new_pos[1] + dy) % height)
            if board[new_pos[1]][new_pos[0]] == 1:
                state[f'agent{p["num"]}_alive'] = False
                break
        
        if state[f'agent{p["num"]}_alive']:
            new_heads[p['num']] = new_pos
        elif was_alive: # The agent just died
            changes[f'p{p["num"]}_became_dead'] = True


    # Phase 2: Resolve head-on collisions
    if 1 in new_heads and 2 in new_heads:
        if new_heads[1] == new_heads[2] or \
           (new_heads[1] == state['agent2_trail'][-1] and new_heads[2] == state['agent1_trail'][-1]):
            if state['agent1_alive']: changes['p1_became_dead'] = True
            if state['agent2_alive']: changes['p2_became_dead'] = True
            state['agent1_alive'], state['agent2_alive'] = False, False

    # Phase 3: Update trails for surviving agents
    for p_num, head_pos in new_heads.items():
        if state[f'agent{p_num}_alive']:
            state[f'agent{p_num}_trail'].append(head_pos)
            changes[f'p{p_num}_trail_added'] = True
            state[f'agent{p_num}_length'] += 1
            
            # Log the changed grid cell
            changes['grid_cells_changed'].append(head_pos)
            board[head_pos[1]][head_pos[0]] = 1
            
    return changes


def undo_tick(state, changes):
    """
    Reverts the state dictionary IN-PLACE using the provided change log.
    """
    if changes['p1_trail_added']:
        pos = state['agent1_trail'].pop()
        state['board'][pos[1]][pos[0]] = 0 # EMPTY is 0 [2]
        state['agent1_length'] -= 1
    if changes['p2_trail_added']:
        pos = state['agent2_trail'].pop()
        state['board'][pos[1]][pos[0]] = 0
        state['agent2_length'] -= 1
        
    if changes['p1_boost_used']:
        state['agent1_boosts'] += 1
    if changes['p2_boost_used']:
        state['agent2_boosts'] += 1
        
    if changes['p1_became_dead']:
        state['agent1_alive'] = True
    if changes['p2_became_dead']:
        state['agent2_alive'] = True

    # This handles grid cells for trails that were added but the agent died later
    for pos in changes['grid_cells_changed']:
        if not changes['p1_trail_added'] and pos in state['agent1_trail']: continue
        if not changes['p2_trail_added'] and pos in state['agent2_trail']: continue
        state['board'][pos[1]][pos[0]] = 0

def get_possible_moves_including_boosts(state, player_number):
    """
    Determines all valid moves for a player, including boosts if available.
    An agent cannot move 180 degrees opposite its current direction.
    """
    if player_number == 1:
        trail = state.get("agent1_trail", [])
        boosts = state.get("agent1_boosts", 0)
    else:
        trail = state.get("agent2_trail", [])
        boosts = state.get("agent2_boosts", 0)

    current_direction = "UP"
    if len(trail) >= 2:
        head = trail[-1]
        neck = trail[-2]
        dx, dy = head[0] - neck[0], head[1] - neck[1]
        
        # Handle torus wrap-around from case_closed_game.py [2]
        if abs(dx) > 1: dx = -1 if dx > 0 else 1
        if abs(dy) > 1: dy = -1 if dy > 0 else 1
        
        if (dx, dy) == (0, -1): current_direction = "UP"
        elif (dx, dy) == (0, 1): current_direction = "DOWN"
        elif (dx, dy) == (-1, 0): current_direction = "LEFT"
        elif (dx, dy) == (1, 0): current_direction = "RIGHT"

    all_directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    
    if current_direction in opposites:
        all_directions.remove(opposites[current_direction])
    
    possible_moves = list(all_directions)
    if boosts > 0:
        for direction in all_directions:
            possible_moves.append(f"{direction}:BOOST")
            
    return possible_moves

def simulate_simultaneous_tick(state, move1, move2):
    """
    Simulates a single game tick with moves from both players.
    Returns a new, modified state dictionary.
    """
    new_state = copy.deepcopy(state)
    board = new_state['board']
    height, width = len(board), len(board[0])
    
    players = [
        {'num': 1, 'move': move1, 'trail': new_state['agent1_trail'], 'boosts': new_state['agent1_boosts']},
        {'num': 2, 'move': move2, 'trail': new_state['agent2_trail'], 'boosts': new_state['agent2_boosts']}
    ]
    new_heads = {}
    
    # Phase 1: Calculate new head positions
    for p in players:
        parts = p['move'].split(':')
        direction_str = parts[0]
        use_boost = len(parts) > 1 and parts[1] == 'BOOST'
        
        moves = 2 if use_boost and p['boosts'] > 0 else 1
        if use_boost and p['boosts'] > 0:
            new_state[f'agent{p["num"]}_boosts'] -= 1

        d_map = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
        dx, dy = d_map[direction_str]

        new_pos = p['trail'][-1]
        for _ in range(moves):
            new_pos = ((new_pos[0] + dx) % width, (new_pos[1] + dy) % height)
            if board[new_pos[1]][new_pos[0]] == 1: # agent value is 1 [2]
                new_state[f'agent{p["num"]}_alive'] = False
                break
        
        if new_state[f'agent{p["num"]}_alive']:
            new_heads[p['num']] = new_pos

    # Phase 2: Resolve head-on collisions [2]
    if 1 in new_heads and 2 in new_heads:
        if new_heads[1] == new_heads[2]:
            new_state['agent1_alive'], new_state['agent2_alive'] = False, False
        elif new_heads[1] == state['agent2_trail'][-1] and new_heads[2] == state['agent1_trail'][-1]:
             new_state['agent1_alive'], new_state['agent2_alive'] = False, False

    # Phase 3: Update trails for surviving agents
    for p_num, head_pos in new_heads.items():
        if new_state[f'agent{p_num}_alive']:
            new_state[f'agent{p_num}_trail'].append(head_pos)
            board[head_pos[1]][head_pos[0]] = 1
            new_state[f'agent{p_num}_length'] += 1

    return new_state

def calculate_voronoi_score(state, my_player_number):
    """Calculates territory control using a simultaneous BFS."""
    if not state[f'agent{my_player_number}_alive']: return -float('inf')
    opponent_player_number = 3 - my_player_number
    if not state[f'agent{opponent_player_number}_alive']: return float('inf')

    board = state['board']
    height, width = len(board), len(board[0])
    my_head = tuple(state[f'agent{my_player_number}_trail'][-1])
    op_head = tuple(state[f'agent{opponent_player_number}_trail'][-1])

    q = deque([(my_head, my_player_number), (op_head, opponent_player_number)])
    visited = {my_head, op_head}
    my_territory, op_territory = 1, 1

    while q:
        (x, y), player = q.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = (x + dx) % width, (y + dy) % height
            if (nx, ny) not in visited and board[ny][nx] == 0:
                visited.add((nx, ny))
                if player == my_player_number: my_territory += 1
                else: op_territory += 1
                q.append(((nx, ny), player))
    return my_territory - op_territory

# In agent.py, replace your alpha_beta_min_turn function

# In agent.py, replace your alpha_beta_min_turn function

def alpha_beta_min_turn(state, depth, alpha, beta, my_player_number):
    """Minimizer's turn. Finds the best counter-move for the opponent."""
    if depth == 0 or not state['agent1_alive'] or not state['agent2_alive']:
        return calculate_voronoi_score(state, my_player_number)

    opponent_player_number = 3 - my_player_number
    opponent_moves = get_possible_moves_including_boosts(state, opponent_player_number)
    
    worst_score = float('inf')
    
    # We must assume the Maximizer (our agent) also plays optimally.
    # We find its best move from this state to pass down.
    my_best_next_move = get_possible_moves_including_boosts(state, my_player_number)[0] # Simplified choice

    for opponent_move in opponent_moves:
        if my_player_number == 1:
            move1, move2 = my_best_next_move, opponent_move
        else:
            move1, move2 = opponent_move, my_best_next_move

        changes = apply_and_log_tick(state, move1, move2)
        score = alpha_beta_max_turn(state, depth - 1, alpha, beta, my_player_number)
        undo_tick(state, changes)

        worst_score = min(worst_score, score)
        beta = min(beta, worst_score)
        if beta <= alpha:
            break
    return worst_score

def alpha_beta_max_turn(state, depth, alpha, beta, my_player_number):
    """Maximizer's turn. Finds the best move for our agent."""
    if depth == 0 or not state['agent1_alive'] or not state['agent2_alive']:
        return calculate_voronoi_score(state, my_player_number)

    my_moves = get_possible_moves_including_boosts(state, my_player_number)
    best_score = -float('inf')

    # We must assume the Minimizer (opponent) also plays optimally.
    # Find its best counter-move to pass down.
    opponent_best_next_move = get_possible_moves_including_boosts(state, 3 - my_player_number)[0] # Simplified

    for my_move in my_moves:
        if my_player_number == 1:
            move1, move2 = my_move, opponent_best_next_move
        else:
            move1, move2 = opponent_best_next_move, my_move

        changes = apply_and_log_tick(state, move1, move2)
        score = alpha_beta_min_turn(state, depth - 1, alpha, beta, my_player_number)
        undo_tick(state, changes)

        best_score = max(best_score, score)
        alpha = max(alpha, best_score)
        if beta <= alpha:
            break
    return best_score

# In agent.py, replace your decide_move function

# In agent.py, replace your decide_move function

def decide_move(current_state, my_player_number):
    """Main decision function. Applies/undos top-level moves."""
    my_moves = get_possible_moves_including_boosts(current_state, my_player_number)
    if not my_moves: return "UP"

    best_move = my_moves[0]
    best_score = -float('inf')
    search_depth = 6 # With this optimization, you should be able to increase this value.

    # We only need one deepcopy to protect the original state.
    search_state = copy.deepcopy(current_state)
    
    opponent_player_number = 3 - my_player_number
    # Assume the opponent makes their best move. We can use a simple heuristic for the top-level.
    opponent_best_move = get_possible_moves_including_boosts(search_state, opponent_player_number)[0]

    for my_move in my_moves:
        if my_player_number == 1:
            move1, move2 = my_move, opponent_best_move
        else:
            move1, move2 = opponent_best_move, my_move
        
        # Apply, recurse, and undo for each top-level move.
        changes = apply_and_log_tick(search_state, move1, move2)
        move_score = alpha_beta_min_turn(search_state, search_depth - 1, -float('inf'), float('inf'), my_player_number)
        undo_tick(search_state, changes)
        
        if move_score > best_score:
            best_score = move_score
            best_move = my_move
            
    return best_move
# --- END: AI STRATEGY IMPLEMENTATION ---

@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    # Simple example: always go RIGHT (replace this with your logic)
    # To use a boost: move = "RIGHT:BOOST"
    if state: # Check if the state dictionary is not empty
        move = decide_move(state, player_number)
    else:
        # Fallback for the very first move before a state is posted
        move = "UP"    
    # Example: Use boost if available and it's late in the game
    # turn_count = state.get("turn_count", 0)
    # if boosts_remaining > 0 and turn_count > 50:
    #     move = "RIGHT:BOOST"
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)