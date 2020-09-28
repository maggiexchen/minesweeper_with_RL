import numpy as np

from states_minesweeper import sample_from_epsilon_greedy
from states_minesweeper import choose_from_policy
from states_minesweeper import get_epsilon_greedy_policy
from states_minesweeper import vfa


def sarsa(
        env, gamma, alpha, epsilon, num_episodes, max_steps=40,
        initial_Q=None):
    """
      Estimates optimal policy by interacting with an environment using
      a td-learning approach

      parameters
      ----------
      env - an environment that can be reset and interacted with via step
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate
      epsilon - the epsilon to use with epsilon greedy policies 
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      policy - an estimate for the optimal policy
      Q - a Q-function estimate of the output policy
      steps -  the number of steps in an episode
      reward - the total reward in an episode
      r - the reward assigned after each step
    """
    num_states = env.num_states
    num_actions = env.num_actions
    
    # setting up Q table
    if initial_Q is None:
        Q = np.zeros((num_states, num_actions))
    else:
        Q = initial_Q
    
    # initial weights
    qweights = np.zeros((2,80))
    for i in range(80):
        qweights[1] = 1
    
    # initial policy
    policy = get_epsilon_greedy_policy(epsilon, Q, env.is_terminal)
    
    # run episodes for num_episodes number of time
    for _ in range(num_episodes):
        # reset state
        s = env.reset()
        # reset reward
        reward = 0
        # reset number of steps
        steps = 0
        # choose initial action
        a = sample_from_epsilon_greedy(s.reshape(80,1), qweights, epsilon)

        # run each episode until terminal 
        while not env.is_terminal() and steps < max_steps:
            # generate the next state, reward and the number of cells revealed
            next_s, r, num_revealed = env.next(a)
            # choose the next action
            next_a = choose_from_policy(policy, int(np.sum(s)))
            
            # updates weights using Sarsa
            q_current = vfa(s, a, qweights)
            q_next = vfa(next_s, next_a, qweights)
            DeltaW = alpha*(r +gamma*q_next - q_current)*s

            for i in range(80):
                qweights[a] = qweights[a].reshape(1, 80)
                qweights[a][i] = qweights[a][i] + DeltaW[0][i]
                
            # updating Q table
            Q[int(np.sum(s))][a] = vfa(s, a, qweights)
            
            # estimate for the optimal policy
            policy = get_epsilon_greedy_policy(epsilon, Q, absorbing = env.is_terminal)
            
            # set next state and action to current state and action
            s = next_s
            a = next_a
            # increment the number of steps and rewards
            steps += 1
            reward += r

    return policy, Q, steps, reward, r, num_revealed

def q_learning(
        env, gamma, alpha, epsilon, num_episodes, max_steps=40,
        initial_Q=None):
    """
      Estimates optimal policy by interacting with an environment using
      a td-learning approach

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - the learning rate
      epsilon - the epsilon to use with epsilon greedy policies 
      num_episodes - number of episode to run
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)

      returns
      -------
      policy - an estimate for the optimal policy
      Q - a Q-function estimate of the optimal epsilon-soft policy
      steps -  the number of steps in an episode
      reward - the total reward in an episode
      r - the reward assigned after each step
    """
    num_states = env.num_states
    num_actions = env.num_actions
    
    # inital weights
    qweights = np.zeros((2,80))
    reward = 0
    for i in range(80):
        qweights[1] = 1
        
    # setting up Q table
    if initial_Q is None:
        Q = np.zeros((num_states, num_actions))
    else:
        Q = initial_Q
        
    # initial policy
    policy = get_epsilon_greedy_policy(epsilon, Q, absorbing=env.is_terminal)
    
    # run episodes for num_episodes number of time
    for _ in range(num_episodes):
        # reset state
        s = env.reset()
        # reset reward 
        reward = 0
        # reset steps
        steps = 0
        
        # run each episode until terminal  
        while not env.is_terminal() and steps < max_steps:
            # choose the action
            a = choose_from_policy(policy, int(np.sum(s)))
            # generate the next state, reward and the number of cells revelaed
            next_s, r, num_revealed = env.next(a)
            
            # generate the best action using greedy policy
            best_a = np.argmax(Q[int(np.sum(next_s))])
            # updating weights using Q-learning
            q_current = vfa(s, a, qweights)
            q_next = vfa(next_s, best_a, qweights)
            DeltaW = alpha*(r + gamma*q_next - q_current)*s
            
            for i in range(80):
                qweights[a] = qweights[a].reshape(1,80)
                qweights[a][i] = qweights[a][i] + DeltaW[0][i]
                
            # update the Q table 
            Q[int(np.sum(s))][a] = vfa(s, a, qweights)
        
            # estimate for the optimal policy
            policy = get_epsilon_greedy_policy(epsilon, Q, absorbing = env.is_terminal)
            # set next state and action to current state and action
            s = next_s
            # increment the number of steps and rewards
            steps += 1
            reward += r

    return policy, Q, steps, reward, r,  num_revealed


def monte_carlo_iterative_optimisation(
        env, gamma, epsilon, alpha, num_episodes, max_steps=20,
        initial_Q=None, default_value=0):
    """
      Estimates optimal policy based on monte-carlo estimates of the
      Q-function which are approximated using the iterative update method

      parameters
      ----------
      env - an environment that can be initialised and interacted with
          (typically this might be an MDPSimulation object)
      gamma - the geometric discount for calculating returns
      alpha - iterative update step size
      num_episodes - number of episodes to run in total
      epsilon - the epsilon to use with epsilon greedy policies 
      max_steps (optional) - maximum number of steps per trace (to avoid very
          long episodes)
      initial_Q (optional) - the initial values for the Q-function
      default_value (optional) - if initial values for the Q-function are not
          provided, this is the mean value of the initial Q-function values

      returns
      -------
      policy - (num_states x num_actions)-matrix of policy probabilities for
          estimated optimal policy (will be deterministic so each row will
          have one 1 and the other values will be 0)
      Q - a Q-function estimate of the output policy
      steps - the number of steps it took in an episode 
      num_rev - the number of cells revealed after an episode
    """
    num_states = env.num_states
    num_actions = env.num_actions
    
    # inintialise weights
    qweights = np.zeros((2,80))
    for i in range(80):
        qweights[1] = 1
        
    # setting up the Q table
    if initial_Q is None:
        # we initialise Q randomly around the default value
        Q = np.random.normal(loc=default_value, size=(num_states, num_actions))
    else:
        # an initial set of Q-values is provided (good for follow on learning)
        Q = initial_Q
        
    # run episodes for num_episodes number of time
    for _ in range(num_episodes):
        # reset steps, reward and the number of cells revealed
        steps = 0
        reward = 0
        num_rev = 0
        
        # run each episode until terminal
        while not env.is_terminal() and steps < max_steps:
            
        # the control policy is the epsilon greedy policy from the Q estimates
            control_policy = get_epsilon_greedy_policy(
                    epsilon, Q, absorbing=env.is_terminal)
            
        # get a trace by interacting with the environment
            trace = env.run(control_policy, max_steps=max_steps)
        # iterate over unique state-action pairs in the trace and store the
        # return following the first visit in the corresponding return list
        
        # since it is unlikely that this simulation will encounter two of the same states
        # we are using every_visit_state_action_returns instead of the first_visit version.
            
            for (s, a, r), ret in trace.every_visit_state_action_returns(gamma):
                
                # updating the weights using Monte Carlo
                q_current = vfa(s, a, qweights)
                DeltaW = alpha * (ret - q_current)*s
                if r == 1:
                    num_rev += 1   
            
                for i in range(80):
                    qweights[a] = qweights[a].reshape(1,80)
                    qweights[a][i] = qweights[a][i] + DeltaW[0][i]
                
                # updating the Q table
                Q[int(np.sum(s))][a] = vfa(s, a, qweights)
                
                # estimate for the optimal policy
                policy = get_epsilon_greedy_policy(epsilon,Q, absorbing=env.is_terminal)
                
                # increment the number of steps and rewards
                reward += r
                steps += 1
                
    return policy, Q, steps, reward, num_rev