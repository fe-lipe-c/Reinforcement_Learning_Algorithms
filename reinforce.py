#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tensorflow_probability as tfp

import numpy as np
import pandas as pd

import time

def discounted_rewards(rewards, gamma):
    
    rtg = np.zeros_like(rewards , dtype=np.float32)
    rtg[-1] = rewards[-1]
    for i in reversed(range(len(rewards)-1)):
        
        rtg[i] = rewards[i] + gamma * rtg[i+1]
        
    return rtg


class Policy(Model):
    
    def __init__(self, hidden_layers, hidden_size, output_size, activation, output_activation):
        
        super(Policy, self).__init__()
        self.hidden_layers = [Dense(hidden_size[i], activation=activation[i]) for i in range(hidden_layers)]
        self.output_layer = Dense(output_size, activation=output_activation)
        
    def call(self, state):
        
        x = state
        
        for layer in self.hidden_layers:
            
            x = layer(x)
        
        return self.output_layer(x)


class Buffer():
    
    def __init__(self, gamma):
        
        self.gamma = gamma
        self.obs = []
        self.actions = []
        self.returns = []
    
    def store(self, temp_traj):
        
        if len(temp_traj) > 0:
            self.obs.extend(temp_traj[:,0])
            ret = discounted_rewards(temp_traj[:,1], self.gamma)
            self.returns.extend(ret)
            self.actions.extend(temp_traj[:,2])
        
    def get_batch(self):
        
        return np.array(self.obs,dtype=np.float32), self.actions, self.returns
    
    def __len__(self):
        
        assert(len(self.obs) == len(self.actions) == len(self.returns))
        return len(self.obs)


def REINFORCE(env_name, hidden_layers, hidden_size, activation, output_activation, 
              alpha, num_epochs, gamma, steps_per_epoch):
    
    env = gym.make(env_name)
    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    policy = Policy(hidden_layers, hidden_size, act_dim, activation, output_activation)
    
    obs = env.reset()
    _ = policy.predict(np.array([obs]))
    
    policy.compile(optimizer = tf.keras.optimizers.Adam(alpha))
    
    step_count = 0
    train_rewards = []
    train_ep_len = []
    plot_mean_rew = []
    plot_steps = []
    plot_std = []
    
    timer = time.time()
    
    for epoch in range(num_epochs):
        
        
        obs = np.array([env.reset()])
        
        buffer = Buffer(gamma)
        env_buffer = []
        epoch_rewards = []
        
        done = False
        while len(buffer) < steps_per_epoch:
           
            
            #policy_actions = policy.predict(obs)
            actions_prob = policy.predict(obs)
            actions_dist = tfp.distributions.Categorical(probs=actions_prob, dtype=tf.float32)
            action = int(actions_dist.sample().numpy()[0])
            
            #action = tf.squeeze(tf.random.categorical(policy_actions,1))
            
            #next_obs, reward, done, _ = env.step(np.squeeze(action))
            next_obs, reward, done, _ = env.step(action)
            
            env_buffer.append([obs.copy(), reward, action])
            
            obs = np.array([next_obs.copy()])
            step_count += 1
            epoch_rewards.append(reward)
            
            if done: 
                
                buffer.store(np.array(env_buffer))
                env_buffer = []
                
                train_rewards.append((np.sum(epoch_rewards)))
                train_ep_len.append(len(epoch_rewards))
                
                obs = np.array([env.reset()])
                epoch_rewards = []
                
        # Policy Optimization
        
        obs_batch, action_batch, return_batch = buffer.get_batch()
        
        with tf.GradientTape() as tape:
            
            one_hot_actions = tf.keras.utils.to_categorical(action_batch, act_dim, dtype=np.float32)
            
            pi_logits = policy(obs_batch, training=True)
            
            pi_log = tf.reduce_sum(tf.multiply(one_hot_actions.reshape(one_hot_actions.shape[0],1,one_hot_actions.shape[1]),
                                               tf.math.log(pi_logits)), axis=2)
            
            return_batch_array = np.array(return_batch).reshape(len(return_batch),1)
            pi_loss = -tf.reduce_mean(pi_log * return_batch_array)
            
            
            model_gradients = tape.gradient(pi_loss, policy.trainable_variables)
            policy.optimizer.apply_gradients(zip(model_gradients, policy.trainable_variables))
        
        # Statistics
        
        if epoch % 10 == 0:
            
            print('Ep:%d MnRew:%.2f StdRew:%.1f EpLen:%.1f Buffer:%d -- Step:%d -- Time:%d' % 
                  (epoch, np.mean(train_rewards), np.std(train_rewards), np.mean(train_ep_len), 
                   len(buffer), step_count,time.time()-timer))
            
            plot_mean_rew.append(np.mean(train_rewards))
            plot_steps.append(step_count)
            plot_std.append(np.std(train_rewards))
            
            train_rewards = []
            train_ep_len = []
            
            policy.save_weights('./saved_models/enforce_nn')
    
    env.close
    return policy, plot_mean_rew, plot_steps,plot_std
    


