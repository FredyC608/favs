import os
import time
import copy
import math
import random
from collections import deque
from threading import Lock
from flask import Flask, request, jsonify
from case_closed_game import Game, Direction, GameResult, Agent, EMPTY, AGENT, UP, DOWN, LEFT, RIGHT

app = Flask(__name__)
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

PARTICIPANT = "MinimaxMaster"
AGENT_NAME = "DeepBlueJay v4 (Tactical)"

# --- AI HELPER FUNCTIONS START ---

def get_valid_moves(agent: Agent, include_boosts: bool = True) -> list[tuple[Direction, bool]]:
    """Returns list of (Direction, is_boost) tuples."""
    current_direction = agent.direction
    turn_left_map = {Direction.UP: Direction.LEFT, Direction.DOWN: Direction.RIGHT,
                     Direction.LEFT: Direction.DOWN, Direction.RIGHT: Direction.UP}
    turn_right_map = {Direction.UP: Direction.RIGHT, Direction.DOWN: Direction.LEFT,
                      Direction.LEFT: Direction.UP, Direction.RIGHT: Direction.DOWN}
    base_moves = [current_direction, turn_left_map[current_direction], turn_right_map[current_direction]]
    valid_moves = []
    for move_dir in base_moves:
        valid_moves.append((move_dir, False))
        if include_boosts and agent.boosts_remaining > 0:
             valid_moves.append((move_dir, True))
    return valid_moves

def run_flood_fill(board, start_pos: tuple[int, int]) -> int:
    """BFS to count reachable EMPTY cells."""
    q = deque([start_pos])
    visited = set([start_pos])
    count = 0
    while q:
        x, y = q.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = board._torus_check((x + dx, y + dy))
            if (nx, ny) not in visited and board.get_cell_state((nx, ny)) == EMPTY:
                visited.add((nx, ny))
                q.append((nx, ny))
                count += 1
    return count

def evaluate_board(game_state: Game, my_id: int) -> float:
    """Heuristic: Territory + Proximity Boosts + Wrap Traps + Tactical Cut-offs."""
    my_agent = game_state.agent1 if my_id == 1 else game_state.agent2
    op_agent = game_state.agent2 if my_id == 1 else game_state.agent1

    if not my_agent.alive: return -10000.0
    if not op_agent.alive: return 10000.0
    if not my_agent.trail or not op_agent.trail: return 0.0

    my_head = my_agent.trail[-1]
    op_head = op_agent.trail[-1]

    # --- 1. Base Territory Score ---
    my_space = run_flood_fill(game_state.board, my_head)
    op_space = run_flood_fill(game_state.board, op_head)
    score = (2.0 * my_space) - op_space
    
    # --- 2. Calculate Torus Distances ---
    dx = min(abs(my_head[0] - op_head[0]), game_state.board.width - abs(my_head[0] - op_head[0]))
    dy = min(abs(my_head[1] - op_head[1]), game_state.board.height - abs(my_head[1] - op_head[1]))
    dist_to_op = dx + dy

    # --- 3. Dynamic Boost Valuation ---
    if dist_to_op <= 5:
        score += my_agent.boosts_remaining * 2.0 
    else:
        score += my_agent.boosts_remaining * 20.0

    # --- 4. Wrap Trap Bonus ---
    x_wrapped = abs(my_head[0] - op_head[0]) > game_state.board.width / 2
    y_wrapped = abs(my_head[1] - op_head[1]) > game_state.board.height / 2
    if (x_wrapped or y_wrapped) and dist_to_op <= 4:
        score += 50.0

    # --- 5. NEW: Tactical Cut-Off Bonus ---
    # If we are adjacent to their head, we are threatening them.
    if dist_to_op == 1:
        # Massive bonus for being right in their face, effectively a "check" in chess.
        score += 30.0
        # Extra bonus if we have a boost advantage to escape any counter-trap
        if my_agent.boosts_remaining > op_agent.boosts_remaining:
             score += 20.0

    # --- 6. Center Bias Tie-Breaker ---
    head_x, head_y = my_agent.trail[-1]
    dist_to_center = math.sqrt((head_x - 10)**2 + (head_y - 9)**2)
    score -= (dist_to_center * 0.01) 

    return score

def minimax(game_state: Game, my_id: int, depth: int, alpha: float, beta: float, is_maximizing: bool, start_time: float, time_limit: float) -> float:
    """Minimax with Alpha-Beta Pruning."""
    if (time.time() - start_time) > time_limit: return 0
    my_agent = game_state.agent1 if my_id == 1 else game_state.agent2
    op_agent = game_state.agent2 if my_id == 1 else game_state.agent1

    if depth == 0 or not my_agent.alive or not op_agent.alive or game_state.turns >= 200:
        return evaluate_board(game_state, my_id)

    use_boosts_in_search = (depth > 1)

    if is_maximizing:
        max_eval = -math.inf
        my_possible_moves = get_valid_moves(my_agent, include_boosts=use_boosts_in_search)
        for my_move, my_boost in my_possible_moves:
            min_eval_for_move = math.inf
            op_possible_moves = get_valid_moves(op_agent, include_boosts=use_boosts_in_search)
            for op_move, op_boost in op_possible_moves:
                sim_game = copy.deepcopy(game_state)
                dir1 = my_move if my_id == 1 else op_move
                dir2 = op_move if my_id == 1 else my_move
                boost1 = my_boost if my_id == 1 else op_boost
                boost2 = op_boost if my_id == 1 else my_boost
                sim_game.step(dir1, dir2, boost1, boost2)
                eval = minimax(sim_game, my_id, depth - 1, alpha, beta, False, start_time, time_limit)
                min_eval_for_move = min(min_eval_for_move, eval)
            max_eval = max(max_eval, min_eval_for_move)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = math.inf
        op_possible_moves = get_valid_moves(op_agent, include_boosts=use_boosts_in_search)
        for op_move, op_boost in op_possible_moves:
            max_eval_for_move = -math.inf
            my_possible_moves = get_valid_moves(my_agent, include_boosts=use_boosts_in_search)
            for my_move, my_boost in my_possible_moves:
                sim_game = copy.deepcopy(game_state)
                dir1 = my_move if my_id == 1 else op_move
                dir2 = op_move if my_id == 1 else my_move
                boost1 = my_boost if my_id == 1 else op_boost
                boost2 = op_boost if my_id == 1 else my_boost
                sim_game.step(dir1, dir2, boost1, boost2)
                eval = minimax(sim_game, my_id, depth - 1, alpha, beta, True, start_time, time_limit)
                max_eval_for_move = max(max_eval_for_move, eval)
            min_eval = min(min_eval, max_eval_for_move)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

# --- AI HELPER FUNCTIONS END ---

@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if data:
        with game_lock:
            LAST_POSTED_STATE.clear()
            LAST_POSTED_STATE.update(data)
            if "board" in data: GLOBAL_GAME.board.grid = data["board"]
            if "agent1_trail" in data: GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
            if "agent2_trail" in data: GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])
            if "agent1_length" in data: GLOBAL_GAME.agent1.length = int(data["agent1_length"])
            if "agent2_length" in data: GLOBAL_GAME.agent2.length = int(data["agent2_length"])
            if "agent1_alive" in data: GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
            if "agent2_alive" in data: GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
            if "agent1_boosts" in data: GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
            if "agent2_boosts" in data: GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
            if "turn_count" in data: GLOBAL_GAME.turns = int(data["turn_count"])
            if len(GLOBAL_GAME.agent1.trail) >= 2:
                t1, t2 = GLOBAL_GAME.agent1.trail[-2], GLOBAL_GAME.agent1.trail[-1]
                dx, dy = t2[0]-t1[0], t2[1]-t1[1]
                if dx > 1: dx = -1
                elif dx < -1: dx = 1
                if dy > 1: dy = -1
                elif dy < -1: dy = 1
                for d in Direction:
                    if d.value == (dx, dy): GLOBAL_GAME.agent1.direction = d; break
            if len(GLOBAL_GAME.agent2.trail) >= 2:
                t1, t2 = GLOBAL_GAME.agent2.trail[-2], GLOBAL_GAME.agent2.trail[-1]
                dx, dy = t2[0]-t1[0], t2[1]-t1[1]
                if dx > 1: dx = -1
                elif dx < -1: dx = 1
                if dy > 1: dy = -1
                elif dy < -1: dy = 1
                for d in Direction:
                    if d.value == (dx, dy): GLOBAL_GAME.agent2.direction = d; break
    return jsonify({"status": "state received"}), 200

@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)
    start_time = time.time()
    TIME_LIMIT = 3.8
    with game_lock:
        sandbox_game = copy.deepcopy(GLOBAL_GAME)
    my_id = player_number
    my_agent = sandbox_game.agent1 if my_id == 1 else sandbox_game.agent2
    op_agent = sandbox_game.agent2 if my_id == 1 else sandbox_game.agent1
    best_move_final = (my_agent.direction, False)

    try:
        depth = 1
        while True:
            if time.time() - start_time > TIME_LIMIT - 0.2: break
            current_depth_best_move = None
            current_depth_best_score = -math.inf
            valid_moves = get_valid_moves(my_agent, include_boosts=True)
            valid_moves.sort(key=lambda x: (x[1], x[0] != my_agent.direction))

            for my_move, my_boost in valid_moves:
                worst_op_response_score = math.inf
                op_valid_moves = get_valid_moves(op_agent, include_boosts=True)
                for op_move, op_boost in op_valid_moves:
                    sim_game = copy.deepcopy(sandbox_game)
                    dir1 = my_move if my_id == 1 else op_move
                    dir2 = op_move if my_id == 1 else my_move
                    boost1 = my_boost if my_id == 1 else op_boost
                    boost2 = op_boost if my_id == 1 else my_boost
                    sim_game.step(dir1, dir2, boost1, boost2)
                    score = minimax(sim_game, my_id, depth - 1, -math.inf, math.inf, False, start_time, TIME_LIMIT)
                    worst_op_response_score = min(worst_op_response_score, score)
                if worst_op_response_score > current_depth_best_score:
                    current_depth_best_score = worst_op_response_score
                    current_depth_best_move = (my_move, my_boost)
            if time.time() - start_time > TIME_LIMIT: break
            if current_depth_best_move:
                 best_move_final = current_depth_best_move
            depth += 1
    except Exception as e:
        print(f"Error during search: {e}")

    final_dir, final_boost = best_move_final
    move_str = final_dir.name
    if final_boost: move_str += ":BOOST"
    return jsonify({"move": move_str}), 200

@app.route("/end", methods=["POST"])
def end_game():
    return jsonify({"status": "acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=False)
