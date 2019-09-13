# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:08:26 2019

@author: kgj1234
"""
import numpy as np
#Initialize_Bandit_Rewards, n indicates number of bandits
def Initialize_Rewards(n=10):
    return np.random.normal(size=(n))
def Calculate_Reward_Received(state,rewards):
    return rewards[state]+np.random.normal()
def Update_State_Value(state,times_explored,current_values,reward):
    return current_values[state]+1/times_explored[state]*(reward-current_values[state])
def Select_Action(current_values,epsilon):
    val=np.random.rand()
    if val>1-epsilon:
        return np.random.randint(0,high=len(current_values))
    else:
        return np.random.choice(np.flatnonzero(current_values == current_values.max()))
def Learn_Bandit(rewards,epsilon=0,T=1000):
    #Initialize states as being equal
    current_expected_values=np.zeros(len(rewards))
    current_explored=np.ones((len(rewards)))
    values_received=[]
    for i in range(T):
        new_state=Select_Action(current_expected_values,epsilon)
        curr_reward=Calculate_Reward_Received(new_state,rewards)
        values_received.append(curr_reward)
        current_expected_values[new_state]=Update_State_Value(new_state,
                                      current_explored,current_expected_values,
                                      curr_reward)
        current_explored[new_state]+=1
    return values_received
