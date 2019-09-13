# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:57:02 2019

@author: kgj1234
"""
from matplotlib import pyplot as plt
import Subsidiary_Functions as SF
import numpy as np
epsilon_values=[0,.01,.1,.5]
def simulate_bandits(epsilon_values,T=1000):
    init_rewards=SF.Initialize_Rewards()
    
    Returned_Values=np.zeros((len(epsilon_values),2000,T))
    for i in range(len(epsilon_values)):
        for k in range(2000):
            Returned_Values[i,k,:]=SF.Learn_Bandit(init_rewards,epsilon=epsilon_values[i],
                                                    T=T)
            if k%50==0:
                print(k)
    return Returned_Values

Values=simulate_bandits(epsilon_values)
mean_values=np.mean(Values,1)
for i in range(len(epsilon_values)):
    plt.plot(mean_values[i,:])

plt.legend(['epsilon '+str(epsilon_values[i]) for i in range(len(epsilon_values))])
plt.show()
    
