import numpy as np
import tkinter

class Board(object):
    """
    board for the game
    """

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {} # board states, key:move as location on the board, value:player as pieces type
        self.n_in_row = int(kwargs.get('n_in_row', 5)) # need how many pieces in a row to win
        self.players = [1, 2] # player1 and player2
        
    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)
        self.current_player = self.players[start_player]  # start player        
        self.availables = list(range(self.width * self.height)) # available moves 
        self.states = {} # board states, key:move as location on the board, value:player as pieces type
        self.last_move = -1

    def move_to_location(self, move):
        """       
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move  // self.width
        w = move  %  self.width
        return [h, w]

    def location_to_move(self, location):
        if(len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if(move not in range(self.width * self.height)):
            return -1
        return move

    def current_state(self): 
        """return the board state from the perspective of the current player
        shape: 4*width*height"""
        
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]                           
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0   
            square_state[2][self.last_move //self.width, self.last_move % self.height] = 1.0 # last move indication   
        if len(self.states)%2 == 0:
            square_state[3][:,:] = 1.0

        return square_state[:,::-1,:]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1] 
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if(len(moved) < self.n_in_row*2 - 1):
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):#            
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

class Point:

    def __init__(self, x, y):
        self.x = x;
        self.y = y;
        self.pixel_x = 30 + 30 * self.x
        self.pixel_y = 30 + 30 * self.y

class Game(object):
    """
    game server
    """
    def __init__(self, board, **kwargs):
        self.board = board
    
    def click1(self, event): #click1 because keyword repetition

        current_player = self.board.get_current_player()
        if current_player == 1:
            i = (event.x) // 30
            j = (event.y) // 30
            ri = (event.x) % 30
            rj = (event.y) % 30
            i = i-1 if ri<15 else i
            j = j-1 if rj<15 else j
            move = self.board.location_to_move((i, j))
            if move in self.board.availables:
                self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='black')
                self.board.do_move(move)

    def run(self):
        current_player = self.board.get_current_player()
        
        end, winner = self.board.game_end()
        
        if current_player == 2 and not end:
            player_in_turn = self.players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            i, j = self.board.move_to_location(move)
            self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='white')
                
        end, winner = self.board.game_end()
        
        if end:
            if winner != -1:
                self.cv.create_text(self.board.width*15+15, self.board.height*30+30, text="Game over. Winner is {}".format(self.players[winner]))
                self.cv.unbind('<Button-1>')
            else:
                self.cv.create_text(self.board.width*15+15, self.board.height*30+30, text="Game end. Tie")

            return winner
        else:
            self.cv.after(100, self.run)
        
    def graphic(self, board, player1, player2):
        """
        Draw the board and show game info
        """
        width = board.width
        height = board.height
        
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        self.players = {p1: player1, p2:player2}
        
        window = tkinter.Tk()
        self.cv = tkinter.Canvas(window, height=height*30+60, width=width*30 + 30, bg = 'white')
        self.chess_board_points = [[None for i in range(height)] for j in range(width)]
        
        for i in range(width):
            for j in range(height):
                self.chess_board_points[i][j] = Point(i, j);
        for i in range(width):  #vertical line
            self.cv.create_line(self.chess_board_points[i][0].pixel_x, self.chess_board_points[i][0].pixel_y, self.chess_board_points[i][width-1].pixel_x, self.chess_board_points[i][width-1].pixel_y)
        
        for j in range(height):  #rizontal line
            self.cv.create_line(self.chess_board_points[0][j].pixel_x, self.chess_board_points[0][j].pixel_y, self.chess_board_points[height-1][j].pixel_x, self.chess_board_points[height-1][j].pixel_y)        
        
        self.button = tkinter.Button(window, text="start game!", command=self.run)
        self.cv.bind('<Button-1>', self.click1)
        self.cv.pack()
        self.button.pack()
        window.mainloop()
               
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        start a game between two players
        """
        if start_player not in (0,1):
            raise Exception('start_player should be 0 (player1 first) or 1 (player2 first)')
        self.board.init_board(start_player)

        if is_shown:
            self.graphic(self.board, player1, player2)
        else:
            p1, p2 = self.board.players
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)
            players = {p1: player1, p2:player2}
            while(1):
                current_player = self.board.get_current_player()
                print(current_player)
                player_in_turn = players[current_player]
                move = player_in_turn.get_action(self.board)
                self.board.do_move(move)
                if is_shown:
                    self.graphic(self.board, player1.player, player2.player)
                end, winner = self.board.game_end()
                if end:
                    return winner   

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree
        store the self-play data: (state, mcts_probs, z)
        """
        self.board.init_board()        
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []        
        while(1):
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move   
            self.board.do_move(move)      
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))  
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                #reset MCTS root node
                player.reset_player() 
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
            