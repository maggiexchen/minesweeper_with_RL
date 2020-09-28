#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:18:07 2020

@author: Daniela
"""


import numpy as np
import matplotlib.pyplot as plt
from minesweeper_simulation import Minesweeper


from rl_algorithms import sarsa
from rl_algorithms import q_learning
from rl_algorithms import monte_carlo_iterative_optimisation



num_series = 100
series_size = 10


gamma = 0.95
alpha = 0.01
epsilon = 0.1    

def random():
    width = 3
    height = 3
    num_mines = 1
    
    sim = Minesweeper(num_mines, width, height)
    
    
    features = sim.reset()
    action = np.random.choice([0,1])
    qweights = np.zeros((2,80))
    for i in range(80):
        qweights[1] = 1
    sim = Minesweeper(num_mines, width, height)
    action = np.random.choice([0,1])
    while sim.is_terminal() == False:
        features, reward, num_revealed = sim.next(action)
        
    board_covered = num_revealed / (width * height)
    
    return board_covered

    
        #print(wins, losses)
     
    #for i in range(5):
    #    running_minesweeper()


"""
we start by returning the % of the board cleared from each episode using:
sarsa, q learning and monte carlo iterative optimisation
    
"""

def revealed_sarsa():
    width = 3
    height = 3
    num_mines = 1
    sim = Minesweeper(num_mines, width, height)
    #features = sim.reset()
    Q = np.ones((sim.num_states, sim.num_actions))
    
    policy, Q, steps, reward, r, num_revealed = sarsa(
            sim, gamma=gamma,alpha = alpha, epsilon=epsilon,
            num_episodes=1, initial_Q=Q)
    board_covered = num_revealed / (width * height)
    return board_covered


def revealed_q():
    width = 3
    height = 3
    num_mines = 1
    sim = Minesweeper(num_mines, width, height)
    #features = sim.reset()
    Q = np.ones((sim.num_states, sim.num_actions))

    policy, Q, steps, reward, r, num_revealed = q_learning(
            sim, gamma=gamma,alpha = alpha, epsilon=epsilon,
            num_episodes=1, initial_Q=Q)
    board_covered = num_revealed / (width * height)
    return board_covered

def revealed_mc():
    width = 3
    height = 3
    num_mines = 1
    sim = Minesweeper(num_mines, width, height)
    #features = sim.reset()
    Q = np.ones((sim.num_states, sim.num_actions)) 
    policy, Q, steps,reward, num_rev = monte_carlo_iterative_optimisation(
    sim, gamma, epsilon, alpha, num_episodes = 1, max_steps=20,
    initial_Q=Q, default_value=0)
    #print(r, reward)
    board_covered = num_rev / (width * height)
    return board_covered 

       


#revealed_arr_random = []
revealed_arr_sarsa = []
revealed_arr_q = []
revealed_arr_mc = []

#revealed_arr_random = np.array(revealed_arr_random)


for series in range(num_series):
    for episode in range(series_size):
        revealed_sarsa()
        revealed_arr_sarsa.append(revealed_sarsa())
        revealed_mc()
        revealed_arr_mc.append(revealed_mc())
        revealed_q()
        revealed_arr_q.append(revealed_q())
             
        #steps_arr_sarsa.append(steps_sarsa())

#revealed_arr_q.sort()        
revealed_arr_sarsa = np.array(revealed_arr_sarsa)
revealed_arr_mc = np.array(revealed_arr_mc)
revealed_arr_q = np.array(revealed_arr_q)

figure = plt.figure()
subplot = figure.add_subplot()
subplot1 = figure.add_subplot()
subplot2 = figure.add_subplot()

over_q = []
under_q = []
over_sarsa = []
under_sarsa = []
over_mc = []
under_mc = []

for i in range(len(revealed_arr_q) - 1):
    if revealed_arr_q[i] < 0.5:
        under_q.append(revealed_arr_q[i])
    else:
        over_q.append(revealed_arr_q[i])
        
for i in range(len(revealed_arr_sarsa)):
    if revealed_arr_sarsa[i] < 0.5:
        under_sarsa.append(revealed_arr_sarsa[i])
    else:
        over_sarsa.append(revealed_arr_sarsa[i])
        

for i in range(len(revealed_arr_mc) - 1):
    if revealed_arr_mc[i] < 0.5:
        under_mc.append(revealed_arr_mc[i])
    else:
        over_mc.append(revealed_arr_mc[i])

fig, ax = plt.subplots()
width = 1.2
algorithms = ("Q - Learning", "Sarsa", "Monte-Carlo ")
over_vals = [len(over_q), len(over_sarsa), len(over_mc)]
under_vals = [len(under_q),len(under_sarsa), len(under_mc)]
x = [0, 4, 8]
x = np.array(x)
y = [1.2, 5.2, 9.2]


plt.bar(x, over_vals, width, alpha = 1, color = (0.2, 0.4, 0.8, 0.6), label =  "over 50%")
plt.bar(y,under_vals, width, alpha = 1,color = "orange", label = "under 50%")
plt.ylabel('num of episodes')
plt.title('% of the board opened by the agent')
plt.xticks([0.5, 4.5, 8.5 ],("Q - Learning", "Sarsa", "Monte-Carlo "))
plt.legend()


#plt.title("proportion of the board opened by the agent")
plt.tight_layout()
plt.show()