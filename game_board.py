import numpy as np
import re
from collections import deque
from GUI_v1_4 import GUI

class Board(object):
    '''
    Board for the game (e.g., Gomoku)
    '''
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))
        self.height = int(kwargs.get('height', 15))
        self.states = {}
        # Board states stored as a dict:
        # key: move as location on the board,
        # value: player as piece type
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        # Number of pieces in a row needed to win
        self.players = [1, 2]
        # Player1 and Player2

        self.feature_planes = 8
        # Number of binary feature planes used
        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1, -1]] * self.feature_planes)
        # Use deque to store last 8 moves, filled with [-1, -1] at game start

    def init_board(self, start_player=0):
        '''
        Initialize the board and set initial variables
        '''
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('Board width and height cannot be less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # Starting player
        self.availables = list(range(self.width * self.height))
        # List of available moves
        self.states = {}
        self.last_move = -1

        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1, -1]] * self.feature_planes)

    def move_to_location(self, move):
        '''
        Convert move number to (row, column) coordinates
        '''
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        '''
        Convert (row, column) coordinates to move number
        '''
        if len(location) != 2:
            return -1
        h, w = location
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        '''
        Return the board state from the perspective of the current player.
        State shape: (self.feature_planes + 1) x width x height
        '''
        square_state = np.zeros((self.feature_planes + 1, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # Separate current player's moves and opponent's moves
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]

            # Construct binary feature planes
            for i in range(self.feature_planes):
                if i % 2 == 0:
                    # Even planes: opponent's moves
                    square_state[i][move_oppo // self.width, move_oppo % self.height] = 1.0
                else:
                    # Odd planes: current player's moves
                    square_state[i][move_curr // self.width, move_curr % self.height] = 1.0

            # Remove older moves to maintain history
            for i in range(0, len(self.states_sequence) - 2, 2):
                move, player = self.states_sequence[i]
                if player != -1:
                    if i < self.feature_planes:
                        # Ensure the move is correctly zeroed out
                        if i % 2 == 0:
                            square_state[i][move // self.width, move % self.height] = 0.
                        else:
                            square_state[i][move // self.width, move % self.height] = 0.

        if len(self.states) % 2 == 0:
            # Assign 1 to the color plane if it's player1's turn, else 0
            square_state[self.feature_planes][:, :] = 1.0

        # Reverse the board for correct orientation
        return square_state[:, ::-1, :]

    def do_move(self, move):
        '''
        Update the board with the given move
        '''
        self.states[move] = self.current_player
        # Save the move in states
        self.states_sequence.appendleft([move, self.current_player])
        # Save the last moves in deque for feature planes
        self.availables.remove(move)
        # Remove the played move from available moves
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        # Switch the current player
        self.last_move = move

    def has_a_winner(self):
        '''
        Determine if there's a winner or a forbidden move violation
        '''
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        # Moves that have been played

        if len(moved) < self.n_in_row:
            # Not enough moves to have a winner
            return False, -1

        # Check for n_in_row in all directions
        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            # Horizontal
            if w <= width - n:
                if all(states.get(m + i, -1) == player for i in range(n)):
                    return True, player

            # Vertical
            if h <= height - n:
                if all(states.get(m + i * width, -1) == player for i in range(n)):
                    return True, player

            # Diagonal \
            if w <= width - n and h <= height - n:
                if all(states.get(m + i * (width + 1), -1) == player for i in range(n)):
                    return True, player

            # Diagonal /
            if w >= n - 1 and h <= height - n:
                if all(states.get(m + i * (width - 1), -1) == player for i in range(n)):
                    return True, player

        # Check for forbidden moves if the last move was by player1
        if self.last_move != -1:
            player = self.states[self.last_move]
            if player == self.players[0]:
                if self.check_forbidden_move(self.last_move, player):
                    # Player1 violated forbidden rules, Player2 wins
                    return True, self.players[1]

        return False, -1

    def check_forbidden_move(self, move, player):
        '''
        Check if the player violates forbidden move rules (double three, double four, overline)
        '''
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        open_three = 0
        open_four = 0

        target = 'X' if player == self.players[0] else 'O'

        for dx, dy in directions:
            line = self.get_line(move, dx, dy)

            # Check for overline (six or more in a row)
            if target * (self.n_in_row + 1) in line:
                return True  # Overline violates forbidden move

            # Check for open four (.XXXX.)
            open_four += self.count_pattern(line, r'\.' + target * 4 + r'\.')

            # Check for open three (.XXX.)
            open_three += self.count_pattern(line, r'\.' + target * 3 + r'\.')

            # Check for broken open three (.XX.X. or .X.XX.)
            open_three += self.count_pattern(line, r'\.' + target * 2 + r'\.' + target + r'\.')
            open_three += self.count_pattern(line, r'\.' + target + r'\.' + target * 2 + r'\.')

        if open_four >= 2:
            return True  # Double four violates forbidden move
        if open_three >= 2:
            return True  # Double three violates forbidden move

        return False

    def count_pattern(self, line, pattern):
        '''
        Count non-overlapping occurrences of a pattern in a line
        '''
        return len(re.findall(pattern, line))

    def get_line(self, move, dx, dy):
        '''
        From the last move, get the line of pieces in the direction (dx, dy)
        '''
        x = move % self.width
        y = move // self.width
        line = ''

        # Extend in the negative direction
        nx, ny = x - dx, y - dy
        while 0 <= nx < self.width and 0 <= ny < self.height:
            pos = ny * self.width + nx
            if pos in self.states:
                line = ('X' if self.states[pos] == self.players[0] else 'O') + line
            else:
                line = '.' + line
            nx -= dx
            ny -= dy

        # Add the current move once to avoid duplication
        line += 'X' if self.states[move] == self.players[0] else 'O'

        # Extend in the positive direction
        nx, ny = x + dx, y + dy
        while 0 <= nx < self.width and 0 <= ny < self.height:
            pos = ny * self.width + nx
            if pos in self.states:
                line += 'X' if self.states[pos] == self.players[0] else 'O'
            else:
                line += '.'
            nx += dx
            ny += dy

        return line

    def has_exact_count(self, line, pattern):
        '''
        Check if the line contains an exact match of the pattern
        '''
        regex = r'(?<!X)' + pattern + r'(?!X)'
        return len(re.findall(regex, line)) > 0

    def game_end(self):
        '''
        Check whether the game has ended
        '''
        end, winner = self.has_a_winner()
        if end:
            # If there is a winner, return the winner
            return True, winner
        elif not len(self.availables):
            # If the board is full and no winner, it's a tie
            return True, -1
        return False, -1

    def get_current_player(self):
        '''
        Return the current player
        '''
        return self.current_player

class Game(object):
    '''
    Game server
    '''
    def __init__(self, board, **kwargs):
        '''
        Initialize a board
        '''
        self.board = board

    def graphic(self, board, player1, player2):
        '''
        Draw the board and show game info
        '''
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print(board.states)
        print()
        print(' ' * 2, end='')
        # Print column numbers
        for x in range(width):
            print("{0:4}".format(x), end='')
        print('\r')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(4), end='')
                elif p == player2:
                    print('O'.center(4), end='')
                else:
                    print('-'.center(4), end='')
            print('\r')

    def start_play(self, player1, player2, start_player=0, is_shown=1, print_prob=True):
        '''
        Start a game between two players
        '''
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        # Assign player indices
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move, move_probs = player_in_turn.get_action(self.board, is_selfplay=False, print_probs_value=print_prob)

            self.board.do_move(move)

            if is_shown:
                print('player %r move : %r' % (current_player, [move // self.board.width, move % self.board.width]))
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()

            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_play_with_UI(self, AI, start_player=0):
        '''
        A GUI for playing
        '''
        AI.reset_player()
        self.board.init_board()
        current_player = SP = start_player
        UI = GUI(self.board.width)
        end = False
        while True:
            print('current_player', current_player)

            if current_player == 0:
                UI.show_messages('Your turn')
            else:
                UI.show_messages('AI\'s turn')

            if current_player == 1 and not end:
                move, move_probs = AI.get_action(self.board, is_selfplay=False, print_probs_value=1)
            else:
                inp = UI.get_input()
                if inp[0] == 'move' and not end:
                    if type(inp[1]) != int:
                        move = UI.loc_2_move(inp[1])
                    else:
                        move = inp[1]
                elif inp[0] == 'RestartGame':
                    end = False
                    current_player = SP
                    self.board.init_board()
                    UI.restart_game()
                    AI.reset_player()
                    continue
                elif inp[0] == 'ResetScore':
                    UI.reset_score()
                    continue
                elif inp[0] == 'quit':
                    exit()
                    continue
                elif inp[0] == 'SwitchPlayer':
                    end = False
                    self.board.init_board()
                    UI.restart_game(False)
                    UI.reset_score()
                    AI.reset_player()
                    SP = (SP+1) % 2
                    current_player = SP
                    continue
                else:
                    # Ignore unrecognized input
                    continue
            # Perform the move if not ended
            if not end:
                UI.render_step(move, self.board.current_player)
                self.board.do_move(move)
                current_player = (current_player + 1) % 2
                end, winner = self.board.game_end()
                if end:
                    if winner != -1:
                        print("Game end. Winner is player", winner)
                        UI.add_score(winner)
                    else:
                        print("Game end. Tie")
                    print(UI.score)
                    print()

    def start_self_play(self, player, is_shown=0):
        '''
        Start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        '''
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 is_selfplay=True,
                                                 print_probs_value=False)
            # Store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # Perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # Winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # Reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)