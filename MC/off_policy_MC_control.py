import matplotlib
import numpy as np
import sys
import random

from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()
returns_sum = defaultdict(float)
returns_count = defaultdict(float)
ratios_count = defaultdict(float)


def create_random_policy(nA):
    """
  Creates a random policy function.

  Args:
      nA: Number of actions in the environment.

  Returns:
      A function that takes an observation as input and returns a vector
      of action probabilities
  """
    A = np.ones(nA, dtype=float) / nA

    def policy_fn():
        return random.randint(0, 1), A

    return policy_fn


def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.

    Args:
        Q: A dictionary that maps from state -> action values

    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """

    def policy_fn(observation):
        greedy_action = np.argmax(Q[observation])
        return greedy_action, greedy_prob(greedy_action)

    def greedy_prob(action):
        vals = {0: np.array([1, 0]),
                1: np.array([0, 1])
                }
        return vals[action]

    return policy_fn


def mc_prediction(Q, behavior_policy, target_policy, env, discount_factor=1.0):
    episode = []
    state = env.reset()

    for t in range(100):
        action_b, prob_b = behavior_policy
        action_t, prob_t = target_policy
        observation, reward, done, _ = env.step(action_b)
        ratio = prob_t[action_b] / prob_b[action_b]
        episode.append((state, action_b, reward, ratio))
        if done:
            break
        state = observation

    states_actions_ratios_in_episode = [(x[0], x[1], x[3]) for x in episode]

    for state, action, ratio in states_actions_ratios_in_episode:
        occurrences = [i for i, x in enumerate(states_actions_ratios_in_episode) if x == (state, action, ratio)]
        first_occurrence = occurrences[0]
        returns_count[(state, action)] += len(occurrences)
        #ratios_count[(state, action)] = ratio
        Gb = sum([ratio * x[2] * discount_factor ** i for i, x in enumerate(episode[first_occurrence:])])
        G = Gb / sum([ratios_count[i] for i, x in enumerate(episode[first_occurrence:])])
        returns_sum[state, action] += G
        Q[state][action] = returns_sum[state, action] / returns_count[state, action]

    return Q


def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
    """
  Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
  Finds an optimal greedy policy.

  Args:
      env: OpenAI gym environment.
      num_episodes: Number of episodes to sample.
      behavior_policy: The behavior to follow while generating episodes.
          A function that given an observation returns a vector of probabilities for each action.
      discount_factor: Gamma discount factor.

  Returns:
      A tuple (Q, policy).
      Q is a dictionary mapping state -> action values.
      policy is a function that takes an observation as an argument and returns
      action probabilities. This is the optimal greedy policy.
  """

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.nA))

    for ep in range(1, num_episodes + 1):
        target_policy = create_greedy_policy(Q)
        behavior_policy = create_random_policy(env.nA)
        Q = mc_prediction(Q, behavior_policy, target_policy, env)

    return Q, target_policy


if __name__ == "__main__":
    random_policy = create_random_policy(env.nA)
    Q, policy = mc_control_importance_sampling(env, num_episodes=100000, behavior_policy=random_policy)
    V = defaultdict(float)
    for state, action_values in Q.items():
        action_value = np.max(action_values)
        V[state] = action_value
    plotting.plot_value_function(V, title="Optimal Value Function")
