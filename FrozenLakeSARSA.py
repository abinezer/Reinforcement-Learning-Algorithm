#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np


# In[2]:


env = gym.make('FrozenLake-v0')


# In[3]:


env.action_space.n


# In[4]:


env.observation_space


# In[5]:


alpha = 0.4


# In[6]:


gamma = 0.999


# In[7]:


q_table = dict([(x,[1,1,1,1]) for x in range(16) ])


# In[8]:


q_table


# In[9]:


def choose_action(observ):
    return np.argmax(q_table[observ])


# In[12]:


for i in range(1000):
    observ = env.reset()
    action = choose_action(observ)
    
    prev_observ = None
    prev_action = None
    t = 0
    for  i in range(2500):
        env.render()
        observ, reward, done, info = env.step(action)
        action = choose_action(observ)
        
        if not prev_observ is None:
            q_old = q_table[prev_observ][prev_action]
            q_new = q_old
            
            if done:
                q_new += alpha * (reward - q_old)
                
            else:
                q_new += alpha * (reward + gamma * q_table[observ][action] - q_old)
                
            new_table = q_table[prev_observ]
            new_table[prev_action] = q_new
            
            q_table[prev_observ] = new_table
            
        prev_observ = observ
        prev_action = action
        
        if done:
            print('Episode {} finished with {} timesteps with r = {}.'.format(i,t,reward))
            break


# In[13]:


new_table


# In[14]:


q_table


# In[ ]:




