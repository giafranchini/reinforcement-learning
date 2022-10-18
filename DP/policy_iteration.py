import numpy as np
import sys

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()
random_policy = np.ones([env.nS, env.nA]) / env.nA

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    V = np.zeros(env.nS)
    while True:
        V_new = np.zeros(env.nS)
        for state in range(env.nS):
            vs = list()
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    vs.append(random_policy[state, action] * (reward + discount_factor*prob*V[next_state]))
            V_new[state] = sum(vs)
        delta = V - V_new
        if all(x < theta for x in abs(delta)):
            break
        V = V_new
    return np.array(V)


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    i = 0
    while True:
        # policy evaluation
        V = policy_eval_fn(policy, env)
        # acting greedy --> new policy
        for state in range(env.nS):
            chosen_action = np.argmax(policy[state])
            policy_stable = True
            v_ns = list()
            # one step look ahead
            for action in range(env.nA):
                for prob, next_state, reward, done in env.P[state][action]:
                    #v_ns.append(V[next_state])
                    # quella sopra non tiene conto del reward perchè è sempre uguale, quella sotto si
                    v_ns.append(random_policy[state, action] * (reward + discount_factor * prob * V[next_state]))
            best_action = np.argmax(np.array(v_ns))

            if best_action != chosen_action:
                # create the new policy
                policy_stable = False
            policy[state] = np.eye(env.nA)[best_action]

        if policy_stable:
            break
    return policy, V


if __name__ == "__main__":
    policy, v = policy_improvement(env)
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Value Function:")
    print(v)
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")
