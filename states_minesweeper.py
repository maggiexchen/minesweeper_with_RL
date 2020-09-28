#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:30:24 2020

@author: Daniela
"""
"""
policies and states

"""
import numpy as np
import copy
import matplotlib.pyplot as plt

"""
  Some functions in this file have been adapted from code provided by Luke Dickens on the
  UCL module INST0060: Foundations of Machine Learning and Data Science
"""

def choose_from_policy(policy, state):
    num_actions = 2
    return np.random.choice(num_actions, p=policy[state,:])

def get_epsilon_greedy_policy(epsilon, Q, absorbing=None):
    """
    Returns an epsilon-greedy policy from a Q-function estimate

    parameters
    ----------
    epsilon - should be 0<epsilon<0.5. This is the variable that controls the
        degree of randomness in the epsilon-greedy policy.
    Q - (num_states x num_actions) matrix of Q-function values
    absorbing (optional) - A vector of booleans, indicating which states are
        absorbing (and hence do not need action decisions). if specified then
        the rows of the output policy will not specify a probability vector
        
    returns
    -------
    policy - (num_states x num_actions) matrix of state dependent action
        probabilities.
    """
    num_actions = 2
    greedy_policy = get_greedy_policy(Q, absorbing=absorbing)
    policy = (1-epsilon)*greedy_policy + epsilon*np.ones(Q.shape)/num_actions
    return policy
    """
    if np.random.rand() < epsilon:
        action = np.argmax(Q,axis=1)
    else:
        action = np.random.randint(0, num_actions,dtype=int)
    return action
    """

def get_greedy_policy(Q, absorbing):
    """
    Returns the greedy policy from a Q-function estimate

    parameters
    ----------
    Q - (num_states x num_actions) matrix of Q-function values
    absorbing (optional) - A vector of booleans, indicating which states are
        absorbing (and hence do not need action decisions). if specified then
        the rows of the output policy will not specify a probability vector
        
    returns
    -------
    policy - (num_states x num_actions) matrix of state dependent action
        probabilities. However this will contain just one 1 per row with
        all other values zero. If a vector specifying absorbing states is 
        pass in then the corresponding rows will not be a valid probability
        vector
    """
    num_states = 80
    num_actions= 2
    dominant_actions = np.argmax(Q, axis=1)
    policy = np.zeros((num_states, num_actions))
    policy[np.arange(num_states), dominant_actions] = 1.
    return policy


def vfa(s_rep, a, qweights):
    """
    parameters
    ----------
    s_rep - is the 1d numpy array of the state feature
    a - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    the q_value approximation for the state and action input
    """
    qweights_a = qweights[a]
    return np.dot(qweights_a.reshape(1,80), s_rep.reshape(80,1))


def sample_from_epsilon_greedy(s_rep, qweights, epsilon):
    """
    A method to sample from the epsilon greedy policy associated with a
    set of q_weights which captures a linear state-action value-function

    parameters
    ----------
    s_rep - is the 1d numpy array of the state feature
    a - is the index of the action
    qweights - a list of weight vectors, one per action
        qweights[i] is the weights for the ith action    

    returns
    -------
    """
    qvalues = np.empty(qweights.shape)
    for a in range(qweights.shape[0]):
        qvalues[a] = np.dot(qweights[a].reshape(1,80), s_rep)
        #qvalues[a] = vfa(s_rep, a, qweights)
    if np.random.random() > epsilon:
        if np.amax(qvalues[0]) < np.amax(qvalues[1]):
            return 1
        return 0
    return np.random.randint(qvalues.shape[0])
