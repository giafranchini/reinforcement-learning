import random
import matplotlib
import numpy as np
import sys
from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')
env = BlackjackEnv()


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    k = 0

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))

    for ep in range(num_episodes):
        # acting epsilon-greedy at every step means improving the policy
        k += 1
        epsilon = 1 / k
        policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)
        # MC policy evaluation
        Q = mc_prediction(Q, policy, env, returns_sum, returns_count)

    return Q, policy


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        greedy_a = np.argmax(Q[observation])
        return epsilon_greedy_prob(greedy_a, epsilon)

    def epsilon_greedy_prob(greedy_a, epsilon):
        vals = {0: np.array([1 - epsilon / 2, epsilon / 2]),
                1: np.array([epsilon / 2, 1 - epsilon / 2])
                }
        return vals[greedy_a]

    return policy_fn


def mc_prediction(Q, policy, env, returns_sum, returns_count, num_episodes=1, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample. ### in this we update after every episode (num_episodes=1)
        discount_factor: Gamma discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    episode = []
    state = env.reset()

    for t in range(100):
        action_prob = 100 * policy(state)
        action = random.choices(range(env.nA), list(action_prob))[0]
        observation, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = observation

    states_actions_in_episode = [(x[0], x[1]) for x in episode]
    for state, action in states_actions_in_episode:
        first_occurence = states_actions_in_episode.index((state, action))
        returns_count[(state, action)] += 1
        G = sum([x[2] * discount_factor ** i for i, x in enumerate(episode[first_occurence:])])
        returns_sum[state, action] += G
        Q[state][action] = returns_sum[state, action] / returns_count[state, action]

    return Q


if __name__ == "__main__":
    Q, policy = mc_control_epsilon_greedy(env, num_episodes=100000, epsilon=1)
    V = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        V[state] = action_value
    plotting.plot_value_function(V, title="Optimal Value Function")
