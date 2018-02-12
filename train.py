
import random
import numpy as np

from collections import defaultdict
from collections import deque

from game import Board, Game
from tf_policy_value_net import PolicyValueNet
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer


class TrainPipeline():
    def __init__(self):
        # params of the board and the game
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params 

        self.temp = 1.0 # the temperature param
        self.n_playout = 400 # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512 # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)        
        self.play_batch_size = 1 
        self.epochs = 3 # num of train_steps for each update
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        
        # start training from a new policy-value net
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height) 
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

    def get_equi_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]"""
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1,2,3,4]:
                # rotate counterclockwise 
                equi_state = np.array([np.rot90(s,i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data
                
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            # augment the data
            play_data = self.get_equi_data(play_data) 
            self.episode_len = len(play_data) / 8
            self.data_buffer.extend(play_data)
                        
    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]            

        for i in range(self.epochs): 
            self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch)

        loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, show_loss=1)
        print(" loss:{:.3f}\n entropy:{:.3f}\n ".format(
                loss, entropy))
        
    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing games against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i%2, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1])/n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio
    
    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):                
                self.collect_selfplay_data(self.play_batch_size)
             
                if len(self.data_buffer) > self.batch_size:
                    print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                    self.policy_update()                    
                # check the performance of the current model，and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    self.policy_value_net.saver.save(self.policy_value_net.sess, self.policy_value_net.model_file)
                    win_ratio = self.policy_evaluate()
                    
                    if win_ratio > self.best_win_ratio: 
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
#                        self.policy_value_net.saver.save(self.policy_value_net.sess, self.policy_value_net.model_file) # update the best_policy
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 100
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            self.policy_value_net.saver.save(self.policy_value_net.sess, self.policy_value_net.model_file)
            print('\n\rquit')
    
if __name__ == '__main__':
    
    training_pipeline = TrainPipeline()
    training_pipeline.run()    
