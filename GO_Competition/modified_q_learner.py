import numpy as np
import random
import json
import itertools
WIN_VALUE = 1.0
DRAW_VALUE = 0.0
LOSS_VALUE = -1.0
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.8
INITIAL_Q_VALUE = 0.0
class MyPlayer():
    game_counter = 0
    def __init__(self):
        self.move_q_values_black = self.read_dict_for_black()
        self.move_q_values_white = self.read_dict_for_white()
        self.type = 'my'
        self.move_q_values = {}
        self.move_history = []
        self.epsilon = [0.99, 0.0]
        self.epsilon_counter = 1
        super().__init__()

    def encode_state(self, state, BOARD_SIZE):
        return ''.join([str(state[i][j]) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)])

    def get_q_values(self, state_hash_value):
        if state_hash_value in self.move_q_values:
            current_state_q_values = self.move_q_values[state_hash_value]
        else:
            current_state_q_values = 0
        return current_state_q_values

    def get_current_game_state(self, my_go):
        return my_go.copy_board()

    def read_dict_for_black(self):
        try:
            with open('black_player.json') as json_file:
                state_move_q_values = json.load(json_file)
            return state_move_q_values
        except FileNotFoundError:
            return {}

    def read_dict_for_white(self):
        try:
            with open('white_player.json') as json_file:
                state_move_q_values = json.load(json_file)
            return state_move_q_values
        except FileNotFoundError:
            return {}

    def get_epsilon_move(self, go, piece_type):
        valid_moves = self.find_valid_moves(go,piece_type)

        move = random.choice(valid_moves)
        move = move[0] * go.size + move[1]
        return move

    def get_move(self, go, piece_type) -> int:
        valid_moves = self.find_valid_moves(go,piece_type)

        max_value = -1000
        move = None
        for row, col in valid_moves:
            state = self.get_current_game_state(go)
            state.place_chess(row, col, piece_type)
            state_hash_value = self.encode_state(state.board, state.size)
            current_q_values = self.get_q_values(state_hash_value)
            if current_q_values > max_value:
                move = (row, col)
                max_value = current_q_values
        return move

    def return_q_move(self, go, piece_type):
        if go.game_end(piece_type):
            return

        self.side = piece_type

        if np.random.uniform(0, 1) <= self.epsilon[self.epsilon_counter]:
            move = self.get_epsilon_move(go, piece_type)
            row = int(move / go.size)
            col = int(move % go.size)
        else:
            self.board = go.board
            move = self.get_move(go, self.side)
            row, col = move[0], move[1]
            move = row * go.size + col
        state = self.get_current_game_state(go)
        state.place_chess(row, col, piece_type)
        self.move_history.append((self.encode_state(state.board, state.size), move))
        return row, col



    def return_greedy_and_minmax_move(self, go, piece_type):
        min_max_moves = self.determine_minimax_algo_position(go, piece_type)
        if len(min_max_moves) == 1:
            return min_max_moves[0]
        else:
            greedy_moves = self.calculate_best_greedy_placement(go, piece_type)
            return self.find_correct_move( greedy_moves,go)

    def number_of_opponents_killed(self, piece_type, state, placements):
        for row, col in placements:
            if not state.place_chess(row, col, piece_type):
                return -1
        return len(state.find_died_pieces(3 - piece_type))

    def find_valid_moves(self, go, piece_type):
        valid_moves = list()
        for row in range(go.size):
            for col in range(go.size):
                if go.valid_place_check(row, col, piece_type, test_check=True):
                    valid_moves.append((row, col))
        return valid_moves

    def determine_minimax_algo_position(self, go, piece_type):
        valid_moves = self.find_valid_moves(go, piece_type)
        most_current_points = 0
        best_placements = list()
        for move in valid_moves:
            state = self.get_current_game_state(go)
            state.place_chess(move[0], move[1], piece_type)
            state.remove_died_pieces(3 - piece_type)
            pseudo_valid_moves = self.find_valid_moves(go, piece_type)
            current_minimum = 1000
            for pseudo_move in pseudo_valid_moves:
                state_2 = self.get_current_game_state(state)
                state_2.place_chess(pseudo_move[0], pseudo_move[1], piece_type)
                state_2.remove_died_pieces(3 - piece_type)
                achieved_points = state_2.score(piece_type)
                if achieved_points < current_minimum:
                    current_minimum = achieved_points
            if current_minimum == most_current_points:
                best_placements.append(move)
            elif current_minimum > most_current_points:
                del best_placements[:]
                best_placements.append(move)
                most_current_points = current_minimum
        return best_placements

    def search_for_current_greedy_move(self, piece_type, my_go, board_size):
        most_opponents_killed = -999
        possible_moves = list()
        for row in range(0, board_size):
            for col in range(0, board_size):
                if my_go.valid_place_check(row, col, piece_type, test_check=True):
                    current_state = self.get_current_game_state(my_go)
                    current_state.place_chess(row, col, piece_type)
                    opponents_killed = len(current_state.find_died_pieces(3 - piece_type))
                    if opponents_killed == most_opponents_killed:
                        move = row * board_size + col
                        possible_moves.append(move)
                    elif most_opponents_killed < opponents_killed:
                        most_opponents_killed = opponents_killed
                        del possible_moves[:]
                        move = row * board_size + col
                        possible_moves.append(move)
        ret = [most_opponents_killed, possible_moves]
        return ret

    def calculate_best_greedy_placement(self, my_go, piece_type):
        required_vals = self.search_for_current_greedy_move(piece_type, my_go, my_go.size)
        first_level_greedy_result = required_vals[0]
        first_level_greedy_moves = list()
        temp = required_vals[1]
        for move in temp:
            row = int(move / my_go.size)
            col = int(move % my_go.size)
            first_level_greedy_moves.append((row,col))
        if 0 < first_level_greedy_result:
            return first_level_greedy_moves
        valid_moves = self.find_valid_moves(my_go, piece_type)
        if len(valid_moves) == 1:
            return first_level_greedy_moves
        most_opponents_killed = 0
        possible_moves = list()
        for placements in itertools.combinations(valid_moves, 2):
            state = self.get_current_game_state(my_go)
            opponents_killed = self.number_of_opponents_killed(piece_type, state,  placements)
            if most_opponents_killed == opponents_killed:
                possible_moves.append(placements[0])
            elif most_opponents_killed < opponents_killed:
                del possible_moves[:]
                possible_moves.append(placements[0])
                most_opponents_killed = opponents_killed
        if len(possible_moves) < 1:
            return valid_moves
        else:
            return possible_moves

    def calculate_displacement(self, position, go):
        return abs(position[0] - (go.size - 1) / 2) + abs(position[1] - (go.size - 1) / 2)

    def find_correct_move(self, placements, go):
        possible_placements = []
        min_displacement = 1000
        for position in placements:
            displacement = self.calculate_displacement(position,go)
            if displacement == min_displacement:
                possible_placements.append(position)
            if displacement < min_displacement:
                possible_placements = []
                possible_placements.append(position)
                min_displacement = displacement
        return random.choice(possible_placements)

    def get_input(self, go, piece_type):

        valid_moves = self.find_valid_moves(go,piece_type)

        if piece_type == 1:
            self.move_q_values = self.move_q_values_black
        else:
            self.move_q_values = self.move_q_values_white

        return self.my_move(go, piece_type, valid_moves)

    def my_move(self, go, piece_type, valid_moves):
        for move in valid_moves:
            state = self.get_current_game_state(go)
            state.place_chess(move[0], move[1], piece_type)
            state_hash_value = self.encode_state(state.board, state.size)
            if state_hash_value in self.move_q_values:
                return self.return_q_move(go, piece_type)
        return self.return_greedy_and_minmax_move(go, piece_type)