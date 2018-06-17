
from game import Board, Game

from tf_policy_value_net import PolicyValueNet
from mcts_alphaZero import MCTSPlayer


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None
    
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):
                location = [int(n, 10) for n in location.split(",")]  # for python3
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n_row = 5
    width, height = 11, 11

    try:
        board = Board(width=width, height=height, n_in_row=n_row)
        game = Game(board)      
        
        ################ human VS AI ###################        
      
        best_policy = PolicyValueNet(width, height, n_row)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)  # set larger n_playout for better performance
        
        human = Human()                   
        
        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':    
    run()
   

