import numpy as np

num_states = 7
# {"0":"C1", "1":"C2", "2":"C3", "3":"pass", "4":"pub", "5":"FB", "6":"sleep"}

i_to_n = {"0": "C1", "1": "C2", "2": "C3", "3": "Pass", "4": "Pub", "5": "FB", "6": "Sleep"}

n_to_i = {}# 名字到索引的字典
for i, name in zip(i_to_n.keys(), i_to_n.values()):
    n_to_i[name] = int(i)

Pss = [
        [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0 ],
        [ 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2 ],
        [ 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
        [ 0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0 ],
        [ 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0 ],
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
]

Pss = np.array(Pss)

rewards = [-2, -2, -2, 10, 1, -1, 0]
gamma = 0.5

chains =[
    ["C1", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "Sleep"],
    ["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB", "FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

def compute_return(start_index = 0, chain = None, gamma = 0.5) -> float:
    '''计算一个马尔科夫奖励过程中某状态的收获值
       Args:
           start_index 要计算的状态在链中的位置
           chain 要计算的马尔科夫过程
           gamma 衰减系数
       Returns：
           retrn 收获值
       '''
    retrn, power, gamma = 0.0, 0, gamma
    for i in range(start_index, len(chain)):
        retrn += np.power(gamma, power) * rewards[n_to_i[chain[i]]]
        power += 1
    return retrn

def compute_value(Pss, rewards, gamma = 0.05):
    '''通过求解矩阵方程的形式直接计算状态的价值
    Args：
        P 状态转移概率矩阵 shape(7, 7)
        rewards 即时奖励 list
        gamma 衰减系数
    Return
        values 各状态的价值
    '''
    #assert(gamma >= 0 and gamma < 1.0)
    #assert(len(P.shape) == 2 and P.shape[0] == P.shape[1])
    rewards = np.array(rewards).reshape((-1,1))
    values = np.dot(np.linalg.inv(np.eye(7,7) - gamma * Pss), rewards)
    return values

values = compute_value(Pss, rewards, gamma = 0.99999)
print(values)