import numpy as np

from states_minesweeper import sample_from_epsilon_greedy

"""
  This class/function has been adapted from code provided by Luke Dickens on the
  UCL module INST0060: Foundations of Machine Learning and Data Science
"""

class Trace(object):
    """
    A trace object stores a sequence of states, actions and rewards for
    an RL style agent. Can calculate returns for the given trace.
    """

    def __init__(self, initial_state, state_names=None, action_names=None):
        """
        Construct the trace with the initial state, and state/action names
        for nice output.

        parameters
        ----------
        """
        self.states = [initial_state]
        self.actions = []
        self.rewards = []
        
    def record(self, action, reward, state_vfa):
        """
        Record the chosen action and the subsequent reward and state.
        """
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(state_vfa)

    def trace_return(self, gamma, t=0):
        """
        Gets the geometrically discounted return from a given time index
        """ 
        return np.sum(gamma**k * r for k, r in enumerate(self.rewards[t:]))

    def first_visit_state_returns(self, gamma):
        """
        Given a geometric discount gets a state indexed return for
        each unique state, corresponding to the first visit return.
        """
        # a dictionary stores the returns
        first_visit_returns = []
        # iterate over prior state and subsequent reward
        # calculate returns and inserting them into the output dictionary
        for t, (s, r) in enumerate(zip(self.states, self.rewards)):
            # check whether state has been seen already
            if not (s, r) in first_visit_returns:
                 first_visit_returns.append(self.trace_return(gamma,t))
        return list(first_visit_returns.items())

    def every_visit_state_returns(self, gamma):
        """
        Given a geometric discount gets a state indexed return for
        each state appearing in the trace.
        """
        every_visit_returns = []
        # iterate over prior state and subsequent reward
        # calculate returns and inserting them into the output dictionary
        for t, (s, r) in enumerate(zip(self.states, self.rewards)):
             every_visit_returns.append((s, self.trace_return(gamma,t)))
        return every_visit_returns

    def first_visit_state_action_returns(self, gamma):
        """
        Given a geometric discount gets a (state, action) indexed return for
        each unique (state, action), corresponding to the first visit return.
        """
        # a dictionary stores the returns to keep track of first visits

        first_visit_returns = {}

        # iterate over prior state and subsequent reward
        # calculate returns and inserting them into the output dictionary
        for t, (s, a, r) in enumerate(
                zip(self.states, self.actions, self.rewards)):
            # check whether state has been seen already
            if not (s, a) in first_visit_returns:
                 first_visit_returns[(s,a)] = self.trace_return(gamma,t)
        return list(first_visit_returns.items())

    def every_visit_state_action_returns(self, gamma):
        """
        Given a geometric discount gets a (state, action) indexed return for
        each (state, action) appearing in the trace.
        """
        every_visit_returns = []
        # iterate over prior state and subsequent reward
        # calculate returns and inserting them into the output dictionary
        for t, (s, a, r) in enumerate(
                zip(self.states, self.actions, self.rewards)):
            every_visit_returns.append( ((s, a, r), self.trace_return(gamma,t)))
        return every_visit_returns


"""
  This class/function has been adapted from code provided by Luke Dickens on the
  UCL module INST0060: Foundations of Machine Learning and Data Science
"""

class Simulation(object):
    """
    A general simulation class for discrete state and discrete actions,
    any inheriting class must define reset(), next(action),
    is_terminal() and step(action). See MDPSimulation for examples of these.
    """
    def __init__(self, num_states, num_actions, state_names, action_names):
        self.num_states = num_states
        self.num_actions = num_actions
        self.state_names = state_names
        self.action_names = action_names

    def run(self, policy, max_steps=None):
        """
        parameters
        ----------
        max_steps - maximum number of steps per epsiode, max_steps;
        policy - control policy, a (num_states x num_actions) matrix, where 
            each row is a probability vector over actions, given the
            corresponding state

        returns
        -------
        trace - a simulated trace/episode in the sim following the policy
           a tuple of (trace_states, trace_actions, trace_rewards)
        """
        if max_steps is None:
            max_steps = 30
        step = 0
        # get the initial state representation 
        state = self.reset()
        
        # initial weights
        qweights = np.zeros((2,80))
        for i in range(80):
            qweights[1] = 1
        # settint epsilon
        epsilon = 0.9
        # generating a trace
        trace = Trace(state, self.state_names, self.action_names)
        
        # generate episode until terminal
        while self.is_terminal() == False and step <= max_steps:
            step += 1
            # if the trace has not terminated then choose another action
            action = sample_from_epsilon_greedy(state.reshape(80,1), qweights, epsilon)
            # the posterior state and reward are drawn from next()
            next_state, reward, num_revealed = self.next(action)
            # store the action, state and reward
            trace.record(action, reward, state)
            # update the next state
            state = next_state
        return trace

    def reset_counts(self):
        """
        The default reset behaviour, you will need to extend this with state
        initialisation for any inheriting class.

        In particular, this helps to monitor performance, by providing a record
        of  total reward and total steps per episode. Note that total reward
        is not discounted.
        """
        self.reward_this_episode = 0
        self.steps_this_episode = 0
        return None
    

    def step(self, action):
        """
        The step function mimics the environment step function from the OpenAI
        Gym interface. Note that this returns a 4th value, but for our purposes
        this will always be None

        returns
        -------
        next_state - the next observed state
        reward - the reward for the transition
        done - if the environment is terminal
        None - a blank return value (you can safely ignore this)
        """
        next_state, reward = self.next(action)
        done = self.is_terminal()
        return next_state, reward, done, None

