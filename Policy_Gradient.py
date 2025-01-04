# -*- coding: utf-8 -*-
"""Untitled1.ipynb
Automatically generated by Colab.
"""

import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Environment Initialization
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Creating a model
model = Sequential()
model.add(Dense(24, input_dim=n_states, activation='relu'))
model.add(Dense(n_actions, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
# Learning Options
gamma = 0.99  # Discounting factor
num_episodes = 2000  # Number of episodes
# Training
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards = 0
    states = []
    actions = []
    while not done:
        # Create a column vector of size 16 for the current state
        state_vector = np.zeros(n_states)
        state_index = state  # state must be the state index
        state_vector[state_index] = 1
        states.append(state_vector)

        probs = model.predict(state_vector.reshape(1, n_states))
        action = np.random.choice(n_actions, p=probs.flatten())
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        rewards += reward
        state = next_state

# Training
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards = 0
    states = []
    actions = []
    while not done:
        # Create a column vector of size 16 for the current state
        state_vector = np.zeros(n_states)
        state_index = state  # state must be the state index
        state_vector[state_index] = 1
        states.append(state_vector)

        probs = model.predict(state_vector.reshape(1, n_states))
        action = np.random.choice(n_actions, p=probs.flatten())
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        rewards += reward
        state = next_state

    # Train the model (Policy Gradient update)
    for s, a in zip(states, actions):
        target = model.predict(s.reshape(1, n_states))
        target[0][a] = rewards  # Use the last reward for learning
        model.fit(s.reshape(1, n_states), target, epochs=1, verbose=0)

state_vector = np.zeros(16)
state_index = state  # The state is assumed to be a state index between 0 and 15
state_vector[state_index] = 1
probs = model.predict(state_vector.reshape(1, 16))

# Testing
total_rewards = 0
state = env.reset()
done = False
while not done:
    state_vector = np.zeros(n_states)
    state_index = state
    state_vector[state_index] = 1
    probs = model.predict(state_vector.reshape(1, n_states))
    action = np.argmax(probs)  # Take the action with the highest probability
    state, reward, done, _ = env.step(action)
    total_rewards += reward

print(f'Total rewards after training: {total_rewards}')

print(f'Total rewards after training: {total_rewards}')

# Testing and visualizing the agent path
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
state = env.reset()
done = False
path = []  # Agent Path
while not done:
    state_vector = np.zeros(n_states)
    state_index = state
    state_vector[state_index] = 1
    probs = model.predict(state_vector.reshape(1, n_states))
    action = np.argmax(probs)  # Take the action with the highest probability
    state, reward, done, _ = env.step(action)
    path.append((state, action))  # Save the path

import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Visualize the agent's path in the maze
plt.figure(figsize=(8, 8))
grid = env.render(mode='ansi')  # Get the visualization of the labyrinth
for (s, a), (next_s, _) in zip(path, path[1:]):
    row, col = s // 4, s % 4
    next_row, next_col = next_s // 4, next_s % 4
    plt.arrow(col + 0.5, row + 0.5, next_col + 0.5 - col - 0.5, next_row + 0.5 - row - 0.5,
              head_width=0.3, head_length=0.3, fc='k', ec='k')
plt.title('Agent Path')
plt.show()

# Visualize the total reward per episode
plt.figure(figsize=(10, 6))
plt.plot(total_rewards)
plt.title('Total Rewards After Training')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Environment Initialization
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Creating a model
model = Sequential()
model.add(Dense(24, input_dim=n_states, activation='relu'))
model.add(Dense(n_actions, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Learning Options
gamma = 0.99  # Discount factor
num_episodes = 2000  # Number of episodes
learning_rate = 0.001  # Level of training
batch_size = 32

# Function to update the Policy Gradient model
def update_model(model, states, actions, rewards):
    targets = model.predict(states)
    for i, action in enumerate(actions):
        targets[i][action] = rewards[i]
    model.fit(states, targets, epochs=1, verbose=0)

# Training
episode_rewards = []  # List to store rewards for each episode
running_rewards = []  # List for storing moving average rewards

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards = 0
    states = []
    actions = []
    discounted_rewards = []  # List for storing discounted rewards

    while not done:
        # Create a column vector of size 16 for the current state
        state_vector = np.zeros(n_states)
        state_index = state  # state must be the state index
        state_vector[state_index] = 1
        states.append(state_vector)

        probs = model.predict(state_vector.reshape(1, n_states))
        action = np.random.choice(n_actions, p=probs.flatten())
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        rewards += reward
        state = next_state

        # Discounted Reward Calculation
        if done:
            discounted_rewards.append(rewards)
        else:
            discounted_rewards.append(rewards * gamma)

    # Updating the Policy Gradient model
    update_model(model, np.array(states), np.array(actions), np.array(discounted_rewards))

    episode_rewards.append(rewards)
    running_rewards.append(np.mean(episode_rewards[-100:]))  # Moving average of rewards for the last 100 episodes

    if episode % 100 == 0:
        print(f'Episode {episode}, Total reward: {rewards}, Running reward: {running_rewards[-1]}')

# Testing
total_rewards = 0
state = env.reset()
done = False
path = []
while not done:
    state_vector = np.zeros(n_states)
    state_index = state
    state_vector[state_index] = 1
    probs = model.predict(state_vector.reshape(1, n_states))
    action = np.argmax(probs)  # Take the action with the highest probability
    state, reward, done, _ = env.step(action)
    total_rewards += reward
    path.append((state, action))
