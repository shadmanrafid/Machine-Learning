import sys
import random
import numpy as np
class Bhushan_player():
    q = {}
    side = 0
    # type = 'BP'
    def __init__(self):
        """ Do whatever you like here. e.g. initialize learning rate
        """
        # =========================================================
        #
        #
        self.type = 'bp'
        self.reading()
        # self.q = self.  # type: #Dict[int, [float]]
        self.move_history = []  # type: #List[(int, int)]
        self.learning_rate = 0.2
        self.value_discount = 0.8
        self.q_init_val = 0.0
        self.game_counter = 0
        self.epsilon = [0.99, 0.5, 0.2, 0.1, 0.0]
        self.epsilon_counter = 4
        super().__init__()
        #
        #
        # =========================================================
    def get_input(self, go, piece_type):
        '''
        Get one input.
        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input.
        '''
        # Get Board
        # Check for best place
        # Return Position of piece
        placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    placements.append((i,j))
        self.board_size = go.size
        self.side = piece_type
        if go.game_end(piece_type):
            return
        if np.random.uniform(0, 1) <= self.epsilon[self.epsilon_counter]:
            move = random.choice(placements)
            row = move[0]
            column = move[1]
            m = row * self.board_size + column
            # print("in random \n")
            # print(m)
        else:
            self.board = go.board
            m = self.get_move(go, piece_type,placements)
            row, column = m[0], m[1]
            # print("in calculated \n")
            # print(m)
        test_go = go.copy_board()
        test_go.place_chess(row, column, piece_type)
        self.move_history.append((self.encode_state(test_go.board, test_go.size),m))
        return row, column
    def get_q(self, board_hash: int) -> [int]:
        """
        Returns the q values for the state with hash value of the  board : `board_hash`.
        :param board_hash: The hash value of the board state for which the q values should be returned
        :return: List of q values for the input state hash.
        """
        #
        # We build the Q table in a lazy manner, only adding a state when it is actually used for the first time
        #
        if board_hash in self.q:
            q_values = self.q[board_hash]
        else:
            # q_values = np.full(self.board_size*self.board_size, self.q_init_val)
            q_values = 0
        return q_values
    def get_move(self, go,piece_type, possible_moves) -> int:
        """
        Return the next move given the board `board` based on the current Q values
        :param board: The current board state
        :return: The next move based on the current Q values for the input state
        """
        max_value = -1000
        m = None
        for i, j in possible_moves:
            temp_board = go.copy_board()
            temp_board.place_chess(i, j, piece_type)
            board_hash = self.encode_state(temp_board.board, temp_board.size)  # type: int
            q_value = self.get_q( board_hash)
            if q_value > max_value:
                m = (i,j)
                max_value = q_value
        # pass
        return m
    def learn(self, result, go):
        """ when the game ends, this method will be called to learn from the previous game i.e. update QValues
            see `play()` method in TicTacToe.py
        Parameters: board
        """
        # =========================================================
        #
        #
        if self.side == 1:
            if result == 1:
                reward = 1
            elif result == 2:
                reward = -1
            else:
                reward = 0
        elif self.side == 2:
            if result == 1:
                reward = -1
            elif result == 2:
                reward = 1
            else:
                reward = 0
        self.move_history.reverse()
        next_max = True  # type: float
        for h in self.move_history:
            self.q[h[0]] = self.get_q(h[0], )
            if next_max:  # First time through the loop
                self.q[h[0]] = reward
                next_max = False
            else:
                temp_q_value = self.q[h[0]] * (
                        1.0 - self.learning_rate) + self.learning_rate *(reward +  self.value_discount * self.q[h[0]] )
                self.q[h[0]] = temp_q_value
            reward = self.q[h[0]]
        self.reset()
        self.game_counter += 1
        if self.game_counter == 99:
            self.converted_file()
            self.game_counter = 0
            print(len(self.q.keys()))
        # =========================================================
    def reset(self):
        self.move_history = []
    def encode_state(self, state, board_size):
        """ Encode the current state of the board as a string
        """
        return ''.join([str(state[i][j]) for i in range(board_size) for j in range(board_size)])
    def converted_file(self):
        f = open('dict_5X5.txt', "w")
        f.write(str(self.q))
        f.close()
    def reading(self):
        try:
            self.q = eval(open('dict_5X5.txt', 'r').read())
        except FileNotFoundError:
            self.q = {}
    def row_column(self, m):
        row = int(m / self.board_size)
        column = int(m % self.board_size)
        return row, column