#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:14:10 2020

@author: maggiechen
"""

import random
import numpy as np

from simulation import Simulation
from states_minesweeper import vfa


class Minesweeper(Simulation):
    """
    A simulation of a minesweeper game on a 3x3 board.
    Reset, next action, terminal conditions and stepsize of
    action are defined.
    """
    
    def __init__(self, num_mines, width, height):
        self.num_mines = num_mines
        self.width = width
        self.height = height
        
        # all the possible numbers taken by the 8 cells surrounding a cell
        possible_mine_numbers = np.linspace(0,10,10)
        self.state_names = []
        for cell_loc in range(8):
            for mine_number in range(10):
                self.state_names.append([cell_loc, possible_mine_numbers[mine_number]])

        self.num_states = 80
        self.state_lookup = {}
        for i, (cell_loc, mine_number) in enumerate(self.state_names):
            self.state_lookup[(cell_loc, mine_number)] = i
        
        # two action choices
        self.action_names = ['move on', 'open the cell']
        self.num_actions = len(self.action_names)
        
        self.revealed = [['?' for j in range(self.height)] for i in range(self.width)]
        self.board = [[0 for j in range(self.height)] for i in range(self.width)]
        
        self.not_bombs = int(self.height * self.width - 1)
        self.num_revealed = 0
        self.agent_loc = [0,0]
                    
        self.wins = 0
        self.losses = 0
        self.reward = 0
        super().__init__(self.num_states, self.num_actions, self.state_names, self.action_names) 
        
        
    def Game_Board(self):
        """
        Places num_mines number of bomb in random locations, bombs are marked as '*'.
        For all safe cell, calculate the number of adjacent bombs. 
        Return the game board self.board.
        """
        
        r = random.randint(0, self.height-1)
        c = random.randint(0, self.width-1)
        self.board[r][c] = "*"
        
        for r in range(self.height):
            for c in range(self.width):
                if self.board[r][c] != "*":
                    if r > 0:
                        if self.board[r - 1][c] == "*":
                            self.board[r][c] += 1
                        if c > 0:
                            if self.board[r - 1][c - 1] == "*":
                                self.board[r][c] += 1
                        if c < (self.width - 1):
                            if self.board[r - 1][c + 1] == "*":
                                self.board[r][c] += 1
                    if r < (self.height - 1):
                        if self.board[r + 1][c] == "*":
                            self.board[r][c] += 1
                        if c > 0:
                            if self.board[r + 1][c - 1] == "*":
                                self.board[r][c] += 1
                        if c < (self.width - 1):
                            if self.board[r + 1][c + 1] == "*":
                                self.board[r][c] += 1
                    if c < (self.width - 1):
                        if self.board[r][c + 1] == "*":
                            self.board[r][c] += 1
                    if c > 0:
                        if self.board[r][c - 1] == "*":
                            self.board[r][c] += 1
                            
        return self.board
        
    def reset(self):
        self.reset_counts()
        # initialise board for game
        self.board = self.Game_Board()
        
        # need to define initial board, only cell [0,0] is revealed
        r = 0
        c = 0 
        self.agent_loc = [r,c]
        self.revealed = [['?' for j in range(self.height)] for i in range(self.width)]
        self.revealed[r][c] = self.board[r][c]
        self.num_revealed = 1
        # initialise reward
        self.reward = 0
        return self.state_representation()


    def next(self, action):
        """
        Action: choose a random cell that is not the cell has not been chosen before.
        Once an action is taken, assign reward according to next state. 
        Returns the representation of the next state, and the reward received.
        
        action: 0 - move on
                1 - reveal
        """
        
        #new_agent_loc = random.choice(possible_coords.remove([0,0]))
        #now our agent will scan the whole board
        if self.agent_loc[0] < self.height - 1:
            new_agent_loc = [self.agent_loc[0] + 1, self.agent_loc[1]]
        elif self.agent_loc[0] == self.height - 1 and self.agent_loc[1] != self.width - 1:
            new_agent_loc = [0, self.agent_loc[1] + 1]
        elif self.agent_loc[0] == self.height - 1 and self.agent_loc[1] == self.width - 1:
            new_agent_loc = [0, 0]
        
        
        new_r = new_agent_loc[0]
        new_c = new_agent_loc[1]
        self.reward  = 1
        
        # for each possible action, reveal cell, and define reward
        # agent chooses to reveal or move on at random
        if self.revealed[new_r][new_c] != "?" and self.revealed[new_r][new_c] != "*":
            action == 0
        else:   
            action = np.random.choice([0, 1])
        
        if action == 1:
            if self.board[new_r][new_c] == '*':
                self.revealed[new_r][new_c] = self.board[new_r][new_c]
                self.agent_loc = new_agent_loc
                self.reward = -10
                self.losses += 1
                

            if self.board[new_r][new_c] != '*':
                self.revealed[new_r][new_c] = self.board[new_r][new_c]
                self.agent_loc = new_agent_loc
                self.num_revealed += 1
                # check for winning condition
                if self.num_revealed == self.not_bombs:
                    self.reward = 10
                    self.wins += 1
                # safely reveal cell reward
                else:    
                    self.reward = 1
        # move on reward
        if action == 0: 
            self.agent_loc = new_agent_loc
            self.reward  = 0
            if self.num_revealed == self.not_bombs :
                    self.reward = 10
                    self.wins += 1
        
        # count the reward 
        #self.increment_counts(self.reward)
        
        
        next_state = self.state_representation()
    
            
        return next_state, self.reward, self.num_revealed
        
        
            
    def is_terminal(self):
        """
        defining the terminal condition:
        the simulation terminates if either:
        the agent wins,i.e. reveals all cells which don't contain a mine 
        and gets a reward of 10
        or it loses (reveals a mine) and it gets a reward of -10 
        
        
        """
        if self.reward == 10 or self.reward == -10:
            return True
        else:
            return False
            
        # True when the revealed safe cells == all the safe cells
        #return self.reward == 10 or self.reward == -10
    
    
    def state_representation(self):
        
        # representing the revealed cell as one-hot vectors of surround cells
        feature = np.zeros(80)
        c = self.agent_loc[0]
        r = self.agent_loc[1]
        """
         cell labels    0 1 2 
                        7 x 3
                        6 5 4
         their location labels are indicated by the row number of state matrix
         their bomb number is indicated by the column number of 1 in the vector
        
         if the number is 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         if the number is 1, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         if the number is 2, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
         if the number is 3, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
         if the number is 4, [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
                               etc. 
         if the cell is unknown [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        """
        if r > 0:
            if self.revealed[r - 1][c] == "?":
                feature[10+9] = 1
            if self.revealed[r - 1][c] != "?" and self.revealed[r - 1][c] != '*':
                feature[10:10+int(self.revealed[r - 1][c])] = 1
            if c > 0:
                if self.revealed[r - 1][c - 1] == "?":
                    feature[0+9] = 1
                if self.revealed[r - 1][c - 1] != "?" and self.revealed[r - 1][c - 1] != '*':
                    feature[0+int(self.revealed[r - 1][c - 1])] = 1
            if c < (self.height - 1):
                if self.revealed[r - 1][c + 1] == "?":
                    feature[20+9] = 1
                if self.revealed[r - 1][c + 1] != "?" and self.revealed[r - 1][c + 1] != '*':
                    feature[20:20+int(self.revealed[r - 1][c + 1])] = 1
                    
        if c < (self.width - 1):
            if self.revealed[r][c + 1] == "?":
                feature[30+9] = 1
            if self.revealed[r][c + 1] != "?" and self.revealed[r][c + 1] != '*':
                feature[30:30+int(self.revealed[r][c + 1])] = 1 
                    
        if r < (self.height - 1):
            if self.revealed[r + 1][c] == "?":
                feature[50+9] = 1
            if self.revealed[r + 1][c] != "?" and self.revealed[r + 1][c] != '*':
                feature[50:50+int(self.revealed[r +1][c])] = 1
            if c > 0:
                if self.revealed[r + 1][c - 1] == "?":
                    feature[60+9] = 1
                if self.revealed[r + 1][c - 1] != "?" and self.revealed[r + 1][c - 1] != '*':
                    feature[60:60+int(self.revealed[r + 1][c - 1])] = 1
            if c < (self.height - 1):
                if self.revealed[r + 1][c + 1] == "?":
                    feature[40+9] = 1
                if self.revealed[r + 1][c + 1] != "?" and self.revealed[r + 1][c + 1] != '*':
                    feature[40:40+int(self.revealed[r + 1][c + 1])] = 1
        if c > 0:
            if self.revealed[r][c - 1] == "?":
                feature[70+9] = 1
            if self.revealed[r][c - 1] != "?" and self.revealed[r][c - 1] != '*':
                feature[70:70+int(self.revealed[r][c - 1])] = 1
                
        return feature