import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()
random_policy = np.ones([env.nS, env.nA]) / env.nA
R = -1 * np.ones([env.nS, env.nA])


def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
  Evaluate a policy given an environment and a full description of the environment's dynamics.

  Args:
      policy: [S, A] shaped matrix representing the policy.
      env: OpenAI env. env.P represents the transition probabilities of the environment.
          env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
          env.nS is a number of states in the environment.
          env.nA is a number of actions in the environment.
      theta: We stop evaluation once our value function change is less than theta for all states.
      discount_factor: Gamma discount factor.

  Returns:
      Vector of length env.nS representing the value function.
  """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        V_new = np.zeros(env.nS)
        for state in range(env.nS):
            vs = list()
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    vs.append(random_policy[state, action] * (reward + discount_factor*prob*V[next_state]))
                #vs.append((random_policy[state, action] * (env.P[state][action][0][2] + discount_factor * (env.P[state][action][0][0] * V[env.P[state][action][0][1]]))))
            V_new[state] = sum(vs)
        delta = V - V_new
        if all(x < theta for x in abs(delta)):
            break
        V = V_new
    return np.array(V)


if __name__ == "__main__":
    v = policy_eval(random_policy, env)
    expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
    print(v)
