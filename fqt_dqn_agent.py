# Author        : JS @breaktoprotect
# Date created  : 16 SEP 2020
# Description   : Using Fixed Q Targets Deep Q Network (DQN) method to train the agent

# Supress warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import tensorflow as tf
#tf.get_logger().setLevel('INFO')

import gym
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import collections
import keyboard
import time
import random

class FQT_DQN_Agent:
    def __init__(self, epsilon=0.99, epsilon_decay=0.9995, epsilon_min=0.1, learning_rate=0.001, discount=0.95, network_update_interval=10):
        # Epsilon Greedy Hyper-parameters
        self.original_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min
        self.learning_rate=learning_rate
        self.discount = discount # Future reward discount

        self.local_q_network = self.create_q_network(4, 16, 16, 2)
        self.target_q_network = self.create_q_network(4, 16, 16, 2)
        self.update_target_q_network()
        self.network_update_interval = network_update_interval      # Every N episodes
        self.current_update_counter = 0
        self.replay_buffer = collections.deque([],maxlen=100_000)    # [Current state, action, reward, future state] 
        self.minimum_size_to_fit = 5000
        self.sample_batch_size = 64
    
    def create_q_network(self, num_inputs, num_hidden_1, num_hidden_2, num_outputs):
        model = Sequential()
        model.add(Dense(num_hidden_1, activation='relu', input_shape=(num_inputs,))) 
        model.add(Dense(num_hidden_2, activation='relu'))
        model.add(Dense(num_outputs, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])       

        return model

    #* Copy weights of local q network to target q network
    def update_target_q_network(self):
        self.target_q_network.set_weights(self.local_q_network.get_weights())
        self.current_update_counter = 0 # Reset counter to 0
        return

    #* Load saved model
    def load_model(self, model_weights_filepath):
        try:
            model_weights = np.load(model_weights_filepath, allow_pickle=True)
        except Exception as e:
            print("[-] Error loading {FILE} due to: {E}".format(FILE=model_weights_filepath, E=e))
            return
        self.local_q_network.set_weights(model_weights)
        self.target_q_network.set_weights(model_weights)
        return

    def save_model(self, model_weights_filepath):
        np.save(model_weights_filepath, self.local_q_network.get_weights())

        return

    #* Reset epsilon
    def reset_epsilon(self):
        self.epsilon = self.original_epsilon

    #* Minimize epsilon
    def minimize_epsilon(self):
        self.epsilon = self.epsilon_min

    #* Start Fixed Q-targets DQN Reinforcment Learning
    def learn(self): 
        #* OpenAI Gym Enviroment - CartPole v0
        env = gym.make("CartPole-v0")
        prev_observation = env.reset()
        done = 0
        score = 0

        while not done:
            if np.random.random() < self.epsilon:
                action = env.action_space.sample() # 0 is left; 1 is right
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min) # Decay epsilon until min value
            else:
                action = np.argmax(self.target_q_network.predict(np.array(prev_observation).reshape(1,4)))

            observation, reward, done, info = env.step(action)

            # Append to Training Data Buffer
            self.replay_buffer.append([prev_observation, action, reward, observation, done])

            # Train and update model (or not, depending on conditions)
            self.train_and_update_model(done)

            prev_observation = observation
            score += reward

            if done:
                if score >= 195:
                    print("[+] Agent has completed the game once with {SCORE} score!".format(SCORE=score)) # Reward = 1.0
                break

        return score

    # Perform local q network model fit
    def train_and_update_model(self, is_done):
        #? Don't train if replay buffer is less than certain size
        if len(self.replay_buffer) < self.minimum_size_to_fit:
            return
        
        #debug
        #print("debug: enter training.")

        # Training batch
        training_batch = random.sample(self.replay_buffer, self.sample_batch_size)

        #debug
        #print("debug:", training_batch[:5])

        current_states_list = np.array([i[0] for i in training_batch]) # [current_state, action, reward, future_state, done]
        current_qs_list = self.local_q_network.predict(current_states_list)

        future_states_list = np.array([i[3] for i in training_batch])
        future_qs_list = self.target_q_network.predict(future_states_list)

        X = []
        y = []

        for index, transition in enumerate(training_batch):
            # Transition is [current_state, action, reward, future_state, done]
            if not transition[4]: 
                max_future_q = np.max(future_qs_list[index])
                new_q = transition[2] + self.discount * max_future_q 
            else:
                new_q = transition[2] 

            current_q = current_qs_list[index]
            current_q[transition[1]] = new_q

            X.append(transition[0])
            y.append(current_q)

        #debug
        #start_fit_time = time.time()

        # Backpropagation of Neural Network
        self.local_q_network.fit(np.array(X), np.array(y), batch_size=self.sample_batch_size, verbose=0, shuffle=False)

        #debug
        #elapsed_fit_time = time.time() - start_fit_time
        #print("debug:: Model fitted in", elapsed_fit_time)

        # Added counter to update Local Q Network -> Target Q Network
        if is_done:
            self.current_update_counter += 1

        # Update Local Q Network -> Target Q Network
        if self.current_update_counter >= self.network_update_interval:
            self.update_target_q_network()

            #debug
            print("debug:: target q network updated!")


    def human_play(self):
        done = 0
        prev_observation = self.env.reset()
        action = -1
        while not done:
            self.env.render()
            time.sleep(1/10)
            
            while action < 0:
                if keyboard.is_pressed('left'):
                    action = 0
                elif keyboard.is_pressed('right'):
                    action = 1
            if keyboard.is_pressed('left'):
                action = 0
            elif keyboard.is_pressed('right'):
                action = 1

                time.sleep(0.1)

            observation, reward, done, info = self.env.step(action)

            #debug
            print("observation:", observation)
            print("reward:", reward)

            if done:
                break

        return

   