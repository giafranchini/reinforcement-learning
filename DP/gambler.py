import numpy as np
import sys
import matplotlib.pyplot as plt
from random import randint

if "../" not in sys.path:
  sys.path.append("../") 

def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
        try first with 0.25 then 0.55
    """
    rewards = np.zeros(101)
    rewards[100] = 1
    
    states = np.array(range(101), int) 
    V = np.zeros(101)
    i = 0

    def one_step_lookahead(s, V, rewards, p_h):
        """
        Helper function to calculate the value for all action in a given state.
        
        Args:
            s: The gambler’s capital. Integer.
            V: The vector that contains values at each state. 
            rewards: The reward vector.
                        
        Returns:
            A vector containing the expected value of each action. 
            Its length equals to the number of actions.
        """
        # tutte le azioni possibili, in funzione dello stato analizzato
        max_a = [s, 100 - s]
        actions = range(min(max_a))
        A = np.zeros(len(actions))
        for a in actions:
            #TODO: qua c'è qualcosa di sbagliato, restituisce sempre 0
            ns_w = s + a
            ns_l = s - a
            A[a] = rewards[ns_w] + rewards[ns_l] + discount_factor*(p_h*ns_w*V[ns_w] + (1-p_h)*ns_l*V[ns_l]) 
        return A
   
    while True:
        
        for s in states:
            if s == 0:
                continue
            elif s == 100:
                continue
            else:
                A = one_step_lookahead(s, V, rewards, p_h)
                V[s] = np.amax(A)
        i += 1
        if i == 10:
            print(V)
            break
    policy = 0
    return policy, V


if __name__ == "__main__":
    p, v = value_iteration_for_gamblers(0.25)
