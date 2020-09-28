  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:23:49 2020

@author: maggiechen
"""

import numpy as np
import matplotlib.pyplot as plt
from minesweeper_simulation import Minesweeper


from rl_algorithms import sarsa
from rl_algorithms import q_learning
from rl_algorithms import monte_carlo_iterative_optimisation



num_series = 100
series_size = 5
#def main():
def main():
    width = 3
    height = 3
    num_mines = 1
    
    sim = Minesweeper(num_mines, width, height)
    features = sim.reset()

    action = np.random.choice([0,1])
    qweights = np.zeros((2,80))
    for i in range(80):
        qweights[1] = 1
    #print("\tAction choice is: %d (meaning %s)" \
    #    % (action, sim.action_names[action],) )
    #rewards_arr = []
    sim = Minesweeper(num_mines, width, height)
    features = sim.reset()
    action = np.random.choice([0,1])
    while sim.is_terminal() == False:
        features, reward, num_revealed = sim.next(action)
    return reward

# setting parameters, gamma - discount factor, alpha - learning rate
# epsilonn - exploration probability
gamma = 0.95
alpha = 0.05
epsilon = 0.1    

"""
we start by returning the rewards and steps from each episode using:
sarsa, q learning and monte carlo iterative optimisation
    
"""
def run_sarsa():
    width = 3
    height = 3
    num_mines = 1
    sim = Minesweeper(num_mines, width, height)
    
    # initial Q table
    Q = np.ones((sim.num_states, sim.num_actions))
    
    # running simulatino with sarsa
    policy, Q, steps, reward, r, num_revealed = sarsa(
            sim, gamma=gamma,alpha = alpha, epsilon=epsilon,
            num_episodes=1, initial_Q=Q)
    return reward
        
def steps_sarsa():
    width = 3
    height = 3
    num_mines = 1
    sim = Minesweeper(num_mines, width, height)
    
    # initial Q table
    Q = np.ones((sim.num_states, sim.num_actions))

    # running simulatino with sarsa
    policy, Q, steps, reward, r, num = sarsa(
            sim, gamma=gamma,alpha = alpha, epsilon=epsilon,
            num_episodes=1, initial_Q=Q)
    return steps


    
def run_q():
    width = 3
    height = 3
    num_mines = 1
    sim = Minesweeper(num_mines, width, height)
    # intial Q table
    Q = np.ones((sim.num_states, sim.num_actions))
    
    # running simulatino with q learning
    policy, Q, steps, reward, r, num_revealed = q_learning(
            sim, gamma=gamma,alpha = alpha, epsilon=epsilon,
            num_episodes=1, initial_Q=Q)
    return reward
        
def steps_q():
    width = 3
    height = 3
    num_mines = 1
    sim = Minesweeper(num_mines, width, height)

    # initial Q table
    Q = np.ones((sim.num_states, sim.num_actions))
    
    
    policy, Q, steps, reward, r , num_revealed= q_learning(
            sim, gamma=gamma,alpha = alpha, epsilon=epsilon,
            num_episodes=1, initial_Q=Q)
    return steps

    
def run_mc():
    width = 3
    height = 3
    num_mines = 1
    sim = Minesweeper(num_mines, width, height)
    #features = sim.reset()
    Q = np.ones((sim.num_states, sim.num_actions))
    
    
    policy, Q, steps,reward , num_rev = monte_carlo_iterative_optimisation(
    sim, gamma, epsilon, alpha, num_episodes = 1, max_steps=40,
    initial_Q=Q, default_value=0)
    return reward

def steps_mc():
    width = 3
    height = 3
    num_mines = 1
    sim = Minesweeper(num_mines, width, height)
    #features = sim.reset()
    Q = np.ones((sim.num_states, sim.num_actions))
    
    policy, Q, steps,reward, num_rev  = monte_carlo_iterative_optimisation(
    sim, gamma, epsilon, alpha, num_episodes = 1, max_steps=40,
    initial_Q=Q, default_value=0)
    return steps


      

figgame, axgame = plt.subplots()
#figstep, axstep = plt.subplots() 
#figrev, axrev = plt.subplots() 
#figbar, axbar = plt.subplots()


wins_random = 0
losses_random = 0
wins_arr_random = [0]
losses_arr_random = [0]
wins_ratio_random = []

wins_sarsa = 0
losses_sarsa = 0     
wins_arr_sarsa = [0]
losses_arr_sarsa = [0]
wins_ratio_sarsa = []
steps_arr_sarsa = [0]
rewards_arr_sarsa = [0]
revealed_arr_sarsa = []

wins_q = 0
losses_q = 0     
wins_arr_q= [0]
losses_arr_q = [0]
wins_ratio_q = []
steps_arr_q = [0]
rewards_arr_q = [0]
revealed_arr_q = []

wins_mc= 0
losses_mc = 0     
wins_arr_mc= [0]
losses_arr_mc = [0]
wins_ratio_mc = []
steps_arr_mc = [0]
rewards_arr_mc = [0]
rewards_arr = [0]
revealed_arr_mc = []


"""
revealed_arr_random = []
for series in range(num_series):
    for episode in range(series_size):
        random()
        revealed_arr_random.append(random())
"""
#print(revealed_arr_random)

#revealed_arr_random = np.array(revealed_arr_random)


for series in range(num_series):
    for episode in range(series_size):
        rew = main()
        if rew >= 10:
            wins_random += 1
        elif rew < 10:
            losses_random += 1
        losses_arr_random.append((losses_random)/(wins_random+losses_random))
        wins_arr_random.append((wins_random)/(wins_random+losses_random))
        rewards_arr.append(rew)


wins_ratio_random = np.array(wins_arr_random)  
rewards_arr = np.array(rewards_arr)
#print(losses_arr)
#print(rewards_arr)

for series in range(num_series):
    for episode in range(series_size):
        rew = run_sarsa()
        steps_sarsa()
        #rewards_sarsa()
        if rew >= 10:
            wins_sarsa += 1   
        elif rew < 10:
            losses_sarsa += 1
        losses_arr_sarsa.append((losses_sarsa)/(wins_sarsa+losses_sarsa))
        wins_arr_sarsa.append((wins_sarsa)/(wins_sarsa+losses_sarsa))
        steps_arr_sarsa.append(steps_sarsa())
        rewards_arr_sarsa.append(rew)
        

#print(losses_arr_sarsa)

for series in range(num_series):
    for episode in range(series_size):
        rew = run_q()
        steps_q()
        #rewards_q()
        if rew >= 10:
            wins_q += 1    
        elif rew < 10:
            losses_q += 1
        losses_arr_q.append((losses_q)/(wins_q+losses_q))
        wins_arr_q.append((wins_q)/(wins_q+losses_q))
        steps_arr_q.append(steps_q())
        rewards_arr_q.append(rew)
        #revealed_arr_q.append(revealed_q())
        


#print(wins_arr_q) 
#print(wins_arr_sarsa)
#print(revealed_arr_sarsa)
#print(revealed_arr_q)
#print(revealed_arr_mc)

for series in range(num_series):
    for episode in range(series_size):
        rew = run_mc()
        steps_mc()
        #rewards_mc()
        #steps_sarsa()
        if rew >= 10:
            wins_mc += 1
        elif rew < 10:
            losses_mc += 1
        losses_arr_mc.append((losses_mc)/(wins_mc+losses_mc))
        wins_arr_mc.append((wins_mc)/(wins_mc+losses_mc))
        steps_arr_mc.append(steps_mc())
        rewards_arr_mc.append(rew)
        #steps_arr_sarsa.append(steps_sarsa())
        
print("num of wins : random - ", wins_random, "q-", wins_q, "sarsa-", wins_sarsa,"mc-", wins_mc)

wins_ratio_q = np.array(wins_arr_q)   
steps_arr_q = np.array(steps_arr_q)   
rewards_arr_q = np.array(rewards_arr_q)

wins_ratio_sarsa = np.array(wins_arr_sarsa)       
steps_arr_sarsa = np.array(steps_arr_sarsa)
rewards_arr_sarsa = np.array(rewards_arr_sarsa)

#print(wins_sarsa, losses_sarsa)

wins_ratio_mc = np.array(wins_arr_mc)   
steps_arr_mc = np.array(steps_arr_mc)   
rewards_arr_mc = np.array(rewards_arr_mc)


def win_consistency(win_ratio, tolerance):
    i = 5
    d_win_ratio = win_ratio[i+1] - win_ratio[i]
    while abs(d_win_ratio) > tolerance and i < len(win_ratio) - 2:
        i += 1
        d_win_ratio = win_ratio[i+1] - win_ratio[i]
    return i

#print(win_consistency(win_ratio_q, tolerance = 0.01))

print('Monte Carlo win rates become consistent after',win_consistency(wins_ratio_mc, 0.005),'episodes.')
print('Sarsa win rates become consistent after',win_consistency(wins_ratio_sarsa, 0.005),'episodes.')
print('Q learning win rates become consistent after',win_consistency(wins_ratio_q, 0.005),'episodes.')


#print("Q" ,  rewards_arr_q)
#print("Sarsa : ",  rewards_arr_sarsa)
#print("MC : " , rewards_arr_mc)

plt.figure()
algs = ("Random AI", "Monte-Carlo","Sarsa", "Q-Learning" )
y_pos = np.arange(len(algs))
total_wins = [wins_random, wins_mc, wins_sarsa, wins_q]
plt.bar(y_pos, total_wins, align = 'center', alpha = 0.5)
plt.xticks(y_pos,( algs))
plt.ylabel("total wins for 500 episodes")

plt.figure()
algs1 = ("Monte-Carlo", "Sarsa", "Q-Learning" )
y_pos1 = np.arange(len(algs1))
total_reward = [np.sum(rewards_arr_mc), np.sum(rewards_arr_sarsa), np.sum(rewards_arr_q)]
plt.bar(y_pos1, total_reward, align = 'center', alpha = 0.5)
plt.xticks(y_pos1,(algs1))
plt.ylabel("total reward for 500 episodes")




axgame.plot(np.arange(0, num_series * series_size + 1, series_size), wins_ratio_random[::series_size], label='random ai') 
axgame.plot(np.arange(0, num_series * series_size + 1, series_size), wins_ratio_sarsa[::series_size], label='sarsa') 
axgame.plot(np.arange(0, num_series * series_size + 1, series_size), wins_ratio_q[::series_size], label='q-learning') 
axgame.plot(np.arange(0, num_series * series_size + 1, series_size), wins_ratio_mc[::series_size], label='monte carlo') 
axgame.set_xlabel("episodes")
axgame.set_ylabel("wins ratio")
axgame.set_title("wins ratio")
axgame.legend()



#axrev.bar()
"""
axstep.plot(np.arange(0, num_series * series_size + 1, series_size), steps_arr_sarsa[::series_size], label='sarsa')
axstep.plot(np.arange(0, num_series * series_size + 1, series_size), steps_arr_q[::series_size], label='q_learning')
axstep.plot(np.arange(0, num_series * series_size + 1, series_size), steps_arr_mc[::series_size], label='mc')
axstep.set_xlabel("episodes")
axstep.set_ylabel("steps per episode")
axstep.legend()

axrew.plot(np.arange(0, num_series * series_size + 1, series_size), rewards_arr_sarsa[::series_size], label='sarsa')
axrew.plot(np.arange(0, num_series * series_size + 1, series_size), rewards_arr_q[::series_size], label='q-learning')
axrew.plot(np.arange(0, num_series * series_size + 1, series_size), rewards_arr_mc[::series_size], label='mc')
axrew.set_xlabel("episodes")
axrew.set_ylabel("reward per episode")
axrew.legend()
"""

"""
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
p1 = figure(tools=TOOLS, toolbar_location="above",
    title="Q-learning (GAMES_COUNT: ) "+str(num_series * series_size)+" . " +str(wins_q) +" Wins.",
    logo="grey",background_fill_color="#E8DDCB")
hist, edges = np.histogram(np.asarray(steps_arr_q), density=False, bins=np.max(steps_arr_q))
p1.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],fill_color="#036564", line_color="#033649")
p1.legend.location = "center_right"
p1.legend.background_fill_color = "darkgrey"
show(p1)
"""
#axrew.legend()
plt.tight_layout()
                