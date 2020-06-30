from blackjack import Player, Dealer, Arena
from utils import str_key, set_dict, get_dict
from utils import draw_policy, draw_value
from utils import epsilon_greedy_policy
import math

class MC_player(Player):
    '''
    具备蒙特卡洛控制玩家
    '''
    def __init__(self, name = "", A = None, display = False):
        super(MC_player, self).__init__(name, A, display)
        self.Q = {}  # 某一状态行为对的价值，策略迭代使用
        self.Nsa = {}  # Nsa的计数：某一状态行为对出现的次数
        self.total_learning_times = 0
        self.policy = self.episilon_greedy_policy
        self.learning_method = self.learn_Q

