from utils import str_key, display_dict
from utils import set_prob, set_reward, get_prob, get_reward
from utils import set_value, set_pi, get_value, get_pi

S = ['浏览手机中', '第一节课', '第二节课', '第三节课', '休息中']
A = ['浏览手机', '学习', '离开浏览', '泡吧', '退出学习']
R = {}  # 奖励Rsa
P = {}  # 状态转移概率Pss'a
gamma = 1.0  # 衰减因子

# 根据学生的马尔可夫决策过程示例的数据设置状态转移概率，默认概率为1
set_prob(P, S[0], A[0], S[0])  # 浏览手机中 - 浏览手机 -> 浏览手机中
set_prob(P, S[0], A[2], S[1])  # 浏览手机中 - 离开浏览 -> 第一节课
set_prob(P, S[1], A[0], S[0])  # 第一节课 - 浏览手机 -> 浏览手机中
set_prob(P, S[1], A[1], S[2])  # 第一节课 - 学习 -> 第二节课
set_prob(P, S[2], A[1], S[3])  # 第二节课 - 学习 -> 第三节课
set_prob(P, S[2], A[4], S[4])  # 第二节课 - 退出学习 -> 退出休息
set_prob(P, S[3], A[1], S[4])  # 第三节课 - 学习 -> 退出休息
set_prob(P, S[3], A[3], S[1], p = 0.2)  # 第三节课 - 泡吧 -> 第一节课
set_prob(P, S[3], A[3], S[2], p = 0.4)  # 第三节课 - 泡吧 -> 第一节课
set_prob(P, S[3], A[3], S[3], p = 0.4)  # 第三节课 - 泡吧 -> 第一节课

set_reward(R, S[0], A[0], -1)  # 浏览手机中 - 浏览手机 -> -1
set_reward(R, S[0], A[2],  0)  # 浏览手机中 - 离开浏览 -> 0
set_reward(R, S[1], A[0], -1)  # 第一节课 - 浏览手机 -> -1
set_reward(R, S[1], A[1], -2)  # 第一节课 - 学习 -> -2
set_reward(R, S[2], A[1], -2)  # 第二节课 - 学习 -> -2
set_reward(R, S[2], A[4],  0)  # 第二节课 - 退出学习 -> 0
set_reward(R, S[3], A[1], 10)  # 第三节课 - 学习 -> 10
set_reward(R, S[3], A[3], +1)  # 第三节课 - 泡吧 -> -1

MDP = (S, A, R, P, gamma)

# print("----状态转移概率字典（矩阵）信息:----")
# display_dict(P)
# print("----奖励字典（函数）信息:----")
# display_dict(R)

Pi = {}
set_pi(Pi, S[0], A[0], 0.5) # 浏览手机中 - 浏览手机
set_pi(Pi, S[0], A[2], 0.5) # 浏览手机中 - 离开浏览
set_pi(Pi, S[1], A[0], 0.5) # 第一节课 - 浏览手机
set_pi(Pi, S[1], A[1], 0.5) # 第一节课 - 学习
set_pi(Pi, S[2], A[1], 0.5) # 第二节课 - 学习
set_pi(Pi, S[2], A[4], 0.5) # 第二节课 - 退出学习
set_pi(Pi, S[3], A[1], 0.5) # 第三节课 - 学习
set_pi(Pi, S[3], A[3], 0.5) # 第三节课 - 泡吧

print("----状态转移概率字典（矩阵）信息:----")
display_dict(Pi)
# 初始时价值为空，访问时会返回0
print("----状态转移概率字典（矩阵）信息:----")
V = {}
display_dict(V)

def compute_q(MDP, V, s, a):
    '''根据给定的MDP，价值函数V，计算状态行为对s,a的价值qsa
    '''
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s,a,s_prime) * get_value(V, s_prime)
    q_sa = get_reward(R, s,a) + gamma * q_sa
    return q_sa

def compute_v(MDP, V, Pi, s):
    '''给定MDP下依据某一策略Pi和当前状态价值函数V计算某状态s的价值
    '''
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s,a) * compute_q(MDP, V, s, a)
    return v_s

# # 根据当前策略使用回溯法来更新状态价值，本章不做要求
# def update_V(MDP, V, Pi):
#     '''给定一个MDP和一个策略，更新该策略下的价值函数V
#     '''
#     S, _, _, _, _ = MDP
#     V_prime = V.copy()
#     for s in S:
#         #set_value(V_prime, s, V_S(MDP, V_prime, Pi, s))
#         V_prime[str_key(s)] = compute_v(MDP, V_prime, Pi, s)
#     return V_prime
#
#
# # 策略评估，得到该策略下最终的状态价值。本章不做要求
# def policy_evaluate(MDP, V, Pi, n):
#     '''使用n次迭代计算来评估一个MDP在给定策略Pi下的状态价值，初始时价值为V
#     '''
#     for i in range(n):
#         V = update_V(MDP, V, Pi)
#         #display_dict(V)
#     return V

# V = policy_evaluate(MDP, V, Pi, 100)
# display_dict(V)
# # 验证状态在某策略下的价值

v = compute_v(MDP, V, Pi, "第三节课")
print("第三节课在当前策略下的价值为:{:.2f}".format(v))