import numpy as np


WIN_VALUE = 1.0
DRAW_VALUE = 0.5
LOSS_VALUE = 0.0
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.7
INITIAL_Q_VALUE = 0.6

class QLearner:

    GAME_NUM = 300000


    def __init__(self):

        self.side = None
        self.move_q_values = {}
        self.move_history = []
        super().__init__()


    def get_q_values(self, state_hash_value):

        if state_hash_value in self.move_q_values:
            current_state_q_values = self.move_q_values[state_hash_value]
        else:
            current_state_q_values = np.full(9,INITIAL_Q_VALUE)
            self.move_q_values[state_hash_value] = current_state_q_values

        return current_state_q_values

    def get_move(self, board):

        state_hash_value = board.encode_state()
        current_state_q_values = self.get_q_values(state_hash_value)
        while True:
            move = np.argmax(current_state_q_values)
            row = 0
            col = 0
            if move == 1:
                row = 0
                col = 1
            elif move == 2:
                row = 0
                col = 2
            elif move == 3:
                row = 1
                col = 0
            elif move == 4:
                row = 1
                col = 1
            elif move == 5:
                row = 1
                col = 2
            elif move == 6:
                col = 0
                row = 2
            elif move == 7:
                row = 2
                col = 1
            elif move == 8:
                row = 2
                col = 2
            if board.is_valid_move(row,col):
                return move
            else:
                current_state_q_values[move] = -1.0


    def move(self, board):

        if board.game_over():
            return
        move = self.get_move(board)
        row = 0
        col = 0
        if move == 1:
            row = 0
            col = 1
        elif move == 2:
            row = 0
            col = 2
        elif move == 3:
            row = 1
            col = 0
        elif move == 4:
            row = 1
            col = 1
        elif move == 5:
            row = 1
            col = 2
        elif move == 6:
            col = 0
            row = 2
        elif move == 7:
            row = 2
            col = 1
        elif move == 8:
            row = 2
            col = 2
        self.move_history.append((board.encode_state(), move))
        return board.move(row, col, self.side)


    def learn(self,board):

        result = board._check_winner()

        if (result == 1 and self.side == 1) or (
                result == 2 and self.side == 2):
            final_value = WIN_VALUE
        elif (result == 1 and self.side == 2) or (
                result == 2 and self.side == 1):
            final_value = LOSS_VALUE
        elif result == 0:
            final_value = DRAW_VALUE
        else:
            raise ValueError("Unexpected game result {}".format(result))

        self.move_history.reverse()
        current_max_val = -1.0

        for move in self.move_history:
            current_state_q_values = self.get_q_values(move[0])
            if current_max_val < 0:
                current_state_q_values[move[1]] = final_value
            else:
                current_state_q_values[move[1]] = current_state_q_values[move[1]] * (
                        1.0 - LEARNING_RATE) + LEARNING_RATE * DISCOUNT_FACTOR * current_max_val

            current_max_val = max(current_state_q_values)
        self.move_history = []


    # do not change this function
    def set_side(self, side):
        self.side = side