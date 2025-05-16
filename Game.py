import random
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
import time

class PlayerType(Enum):
    HUMAN = 1
    AI = 2

class StrategyType(Enum):
    OFFENSIVE = 1
    DEFENSIVE = 2

class GameConfig:
    NUM_PLAYERS = 6
    TOKENS_PER_PLAYER = 3
    DIE_SIDES = 8
    BOARD_RADIUS = 3

class HexLudoBoard:
    def __init__(self):
        self.graph = nx.Graph()
        self.create_hex_board()

    def create_hex_board(self):
        for q in range(-GameConfig.BOARD_RADIUS, GameConfig.BOARD_RADIUS + 1):
            for r in range(-GameConfig.BOARD_RADIUS, GameConfig.BOARD_RADIUS + 1):
                s = -q - r
                if -GameConfig.BOARD_RADIUS <= s <= GameConfig.BOARD_RADIUS:
                    self.graph.add_node((q, r))

        for (q, r) in list(self.graph.nodes):
            for dq, dr in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                neighbor = (q + dq, r + dr)
                if neighbor in self.graph:
                    self.graph.add_edge((q, r), neighbor)

    def get_all_reachable_positions(self, start_pos, steps):
        result = []
        visited = set()

        def dfs(pos, remaining_steps, path):
            if pos not in self.graph:
                return
            if remaining_steps == 0:
                result.append(pos)
                return
            for neighbor in self.graph.neighbors(pos):
                if neighbor not in path:
                    dfs(neighbor, remaining_steps - 1, path + [neighbor])

        dfs(start_pos, steps, [start_pos])
        return result

    def visualize_board(self, tokens_by_position=None):
        def axial_to_cartesian(q, r):
            x = 3/2 * q
            y = (3**0.5) * (r + q/2)
            return (x, y)

        pos = {node: axial_to_cartesian(*node) for node in self.graph.nodes}
        labels = {node: f"{node}" for node in self.graph.nodes}

        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(self.graph, pos, with_labels=True, labels=labels,
                node_size=500, node_color='skyblue', edge_color='gray',
                font_size=8, ax=ax)

        if tokens_by_position:
            for token_pos, token_info in tokens_by_position.items():
                if token_pos in pos:
                    ax.text(pos[token_pos][0], pos[token_pos][1], f"{token_info}",
                            color='red', fontsize=12, ha='center', va='center')

        ax.set_title("Hexagonal Ludo Board Layout (Axial Coordinates)")
        ax.set_aspect('equal')
        plt.pause(10)
        plt.close(fig)

class Token:
    def __init__(self):
        self.position = None

    def is_at_home(self):
        return self.position is None

    def move(self, new_position):
        self.position = new_position

class Player:
    def __init__(self, player_id, player_type):
        self.id = player_id
        self.type = player_type
        self.strategy = StrategyType.OFFENSIVE if player_id % 2 == 0 else StrategyType.DEFENSIVE
        self.tokens = [Token() for _ in range(GameConfig.TOKENS_PER_PLAYER)]
        self.start_pos = self.assign_start()

    def assign_start(self):
        while True:
            q = random.randint(-GameConfig.BOARD_RADIUS, GameConfig.BOARD_RADIUS)
            r = random.randint(-GameConfig.BOARD_RADIUS, GameConfig.BOARD_RADIUS)
            s = -q - r
            if -GameConfig.BOARD_RADIUS <= s <= GameConfig.BOARD_RADIUS:
                return (q, r)

    def has_tokens_on_board(self):
        return any(not t.is_at_home() for t in self.tokens)

    def get_movable_tokens(self):
        return [t for t in self.tokens if not t.is_at_home()]

def heuristic_evaluation(game_state, player_index):
    player = game_state.players[player_index]
    return sum(1 for t in player.tokens if not t.is_at_home())

def minimax(game_state, depth, player_index, alpha, beta):
    if depth == 0 or game_state.is_terminal():
        return heuristic_evaluation(game_state, player_index)

    is_maximizing = (game_state.players[player_index].type == PlayerType.AI)

    best_val = float('-inf') if is_maximizing else float('inf')
    for move in game_state.get_possible_moves(player_index):
        new_state = game_state.simulate_move(player_index, move)
        eval = minimax(new_state, depth - 1, (player_index + 1) % GameConfig.NUM_PLAYERS, alpha, beta)

        if is_maximizing:
            best_val = max(best_val, eval)
            alpha = max(alpha, eval)
        else:
            best_val = min(best_val, eval)
            beta = min(beta, eval)

        if beta <= alpha:
            break

    return best_val

def get_best_move(state, player_index, depth=2):
    best_score = float('-inf')
    best_move = None
    player = state.players[player_index]

    for move in state.get_possible_moves(player_index):
        simulated_state = state.simulate_move(player_index, move)
        score = minimax(simulated_state, depth - 1, (player_index + 1) % GameConfig.NUM_PLAYERS, float('-inf'), float('inf'))

        if player.strategy == StrategyType.OFFENSIVE:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            dist = abs(move[2][0]) + abs(move[2][1])
            score -= dist
            if score > best_score:
                best_score = score
                best_move = move

    return best_move

class GameState:
    def __init__(self, board, players):
        self.board = board
        self.players = players
        self.current_turn = 0

    def is_terminal(self):
        return False

    def get_possible_moves(self, player_index):
        player = self.players[player_index]
        moves = []
        for idx, token in enumerate(player.tokens):
            if token.is_at_home():
                moves.append((idx, 1, player.start_pos))
            else:
                for roll in range(1, GameConfig.DIE_SIDES + 1):
                    if token.position in self.board.graph:
                        reachable = self.board.get_all_reachable_positions(token.position, roll)
                        for pos in reachable:
                            moves.append((idx, roll, pos))
        return moves

    def simulate_move(self, player_index, move):
        new_state = self.clone()
        token_index = move[0]
        player = new_state.players[player_index]
        token = player.tokens[token_index]

        if token.is_at_home():
            token.move(player.start_pos)
        else:
            token.move(move[2])

        return new_state

    def clone(self):
        new_board = self.board
        new_players = [Player(p.id, p.type) for p in self.players]
        for i, p in enumerate(self.players):
            new_players[i].start_pos = p.start_pos
            for j, t in enumerate(p.tokens):
                new_players[i].tokens[j].position = t.position
        new_state = GameState(new_board, new_players)
        new_state.current_turn = self.current_turn
        return new_state

def play_game():
    plt.ion()
    board = HexLudoBoard()
    players = [Player(i, PlayerType.AI if i % 2 == 0 else PlayerType.HUMAN) for i in range(GameConfig.NUM_PLAYERS)]
    state = GameState(board, players)

    for turn in range(30):
        player = state.players[state.current_turn]
        print(f"\nTurn {turn + 1}: Player {player.id} ({'AI' if player.type == PlayerType.AI else 'HUMAN'})")

        if player.type == PlayerType.AI:
            move = get_best_move(state, state.current_turn)
        else:
            # Human player's turn
            print("\nYour tokens:")
            for i, token in enumerate(player.tokens):
                status = "at home" if token.is_at_home() else f"at position {token.position}"
                print(f"Token {i}: {status}")
            
            # Get token selection from user
            while True:
                try:
                    token_index = int(input("\nSelect a token (0-2): "))
                    if 0 <= token_index < GameConfig.TOKENS_PER_PLAYER:
                        break
                    print("Invalid token index. Please select 0, 1, or 2.")
                except ValueError:
                    print("Please enter a valid number.")

            # Get die roll from user
            while True:
                try:
                    roll = int(input("Enter your die roll (1-8): "))
                    if 1 <= roll <= GameConfig.DIE_SIDES:
                        break
                    print(f"Invalid roll. Please enter a number between 1 and {GameConfig.DIE_SIDES}.")
                except ValueError:
                    print("Please enter a valid number.")

            token = player.tokens[token_index]
            if token.is_at_home():
                move = (token_index, roll, player.start_pos)
            else:
                possible_positions = board.get_all_reachable_positions(token.position, roll)
                if not possible_positions:
                    print("No possible moves for this roll.")
                    state.current_turn = (state.current_turn + 1) % GameConfig.NUM_PLAYERS
                    continue
                
                print("\nPossible positions:")
                for i, pos in enumerate(possible_positions):
                    print(f"{i}: {pos}")
                
                while True:
                    try:
                        pos_index = int(input("Select a position (enter the number): "))
                        if 0 <= pos_index < len(possible_positions):
                            break
                        print("Invalid position index.")
                    except ValueError:
                        print("Please enter a valid number.")
                
                move = (token_index, roll, possible_positions[pos_index])

        print(f"Player {player.id} moves token {move[0]} to {move[2]}")
        state = state.simulate_move(state.current_turn, move)

        tokens_by_position = {}
        for p in state.players:
            for i, t in enumerate(p.tokens):
                if not t.is_at_home():
                    label = f"P{p.id}T{i}"
                    tokens_by_position[t.position] = label

        board.visualize_board(tokens_by_position)
        state.current_turn = (state.current_turn + 1) % GameConfig.NUM_PLAYERS

if __name__ == "__main__":
    play_game()