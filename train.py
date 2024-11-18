import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import gym
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras_rl2.agents.dqn import DQNAgent
from keras_rl2.memory import SequentialMemory
from keras_rl2.policy import EpsGreedyQPolicy

# Set up the environment
env = gym.make("Breakout-v0")
nb_actions = env.action_space.n

# Define the model for the agent
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(nb_actions, activation="linear"))

# Configure the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])

# Train the agent
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Save the trained policy
dqn.save_weights("policy.h5", overwrite=True)
env.close()
