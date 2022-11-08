import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
import random

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

env = gym.envs.make("MountainCar-v0")

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)


# Used to convert a state to a featurized representation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))


class Estimator():
        """
        Value Function approximator.
        """

        def __init__(self):
                # We create a separate model for each action in the environment's
                # action space. Alternatively we could somehow encode the action
                # into the features, but this way it's easier to code up.
                self.models = []
                for _ in range(env.action_space.n):
                        model = SGDRegressor(learning_rate="constant")
                        # We need to call partial_fit once to initialize the model
                        # or we get a NotFittedError when trying to make a prediction
                        # This is quite hacky.
                        model.partial_fit([self.featurize_state(env.reset()[0])], [0])
                        self.models.append(model)

        def featurize_state(self, state):
                """
                Returns the featurized representation for a state.
                """
                scaled = scaler.transform([state])
                featurized = featurizer.transform(scaled)
                return featurized[0]

        def predict(self, s, a=None):
                """
                Makes value function predictions.

                Args:
                    s: state to make a prediction for
                    a: (Optional) action to make a prediction for

                Returns
                    If an action a is given this returns a single number as the prediction.
                    If no action is given this returns a vector or predictions for all actions
                    in the environment where pred[i] is the prediction for action i.

                """
                # y prediction with SDG
                q = np.zeros(env.action_space.n)
                for action in range(env.action_space.n):
                    q[action] = self.models[action].predict([self.featurize_state(s)])
                return q[a] if a else q

        def update(self, s, a, y):
                """
                Updates the estimator parameters for a given state and action towards
                the target y.
                """
                # fit a new model
                self.models[a].fit([self.featurize_state(s)], [y])
                return None


def make_epsilon_greedy_policy(estimator, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

        Args:
            estimator: An estimator that returns q values for a given state
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.

        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.

        """

        def policy_fn(observation):
                A = np.ones(nA, dtype=float) * epsilon / nA
                q_values = estimator.predict(observation)
                best_action = np.argmax(q_values)
                A[best_action] += (1.0 - epsilon)
                return A

        return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float between 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    alpha = 1.

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print("\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes, last_reward), end="")
        sys.stdout.flush()

        # TODO: Implement this!
        s = env.reset()[0]
        action_prob = 100 * policy(s)
        a = random.choices(range(env.action_space.n), list(action_prob))[0]

        for t in range(1000):
            next_s, r, is_done, is_truncated, _ = env.step(a)
            q_max = np.max(estimator.predict(next_s))
            a_max = np.argmax(estimator.predict(next_s))
            target = r + discount_factor*q_max
            estimator.update(s, a_max, target)
            stats.episode_rewards[i_episode] += r
            stats.episode_lengths[i_episode] = t
            if is_done or is_truncated:
                break
            s = next_s
            a = a_max

        return stats


if __name__ == "__main__":
    estimator = Estimator()
    # Note: For the Mountain Car we don't actually need an epsilon > 0.0
    # because our initial estimate for all states is too "optimistic" which leads
    # to the exploration of all states.
    stats = q_learning(env, estimator, 100, epsilon=0.0)
    plotting.plot_cost_to_go_mountain_car(env, estimator)
    plotting.plot_episode_stats(stats, smoothing_window=25)
