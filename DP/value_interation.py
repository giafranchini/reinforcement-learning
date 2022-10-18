import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
   
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    max_i = 50
    i = 0
    
    while True: 
        for s in range(env.nS):
            vf_opt, a_opt = one_step_look_ahead(s, env, V)
            V[s] = vf_opt
            policy[s] = np.eye(env.nA)[best_action]

        i += 1
        if i == max_i:
            break

    return policy, V

def one_step_look_ahead(s, env, V, discount_factor=1):
    result = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_s, r, d in env.P[s][a]:
            result[a] = r + discount_factor*prob*V[next_s]
    return np.max(result), np.argmax(result)	


if __name__ == "__main__": 

    policy, v = value_iteration(env)
    
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



    # Test the value function
    expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

