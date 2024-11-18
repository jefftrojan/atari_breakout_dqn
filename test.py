import gym
import cv2
import numpy as np
from keras.models import load_model

def downsample(observation):
	s = cv2.cvtColor(observation[30:,:,:], cv2.COLOR_BGR2GRAY)
	s = cv2.resize(s, (80,80), interpolation = cv2.INTER_AREA) 
	s = s/255.0
	return s

def update_state(state,observation):
	ds_observation = downsample(observation)
	state.append(ds_observation)
	if len(state) > 4:
		state.pop(0)

def sample_action(model,s):
	return np.argmax(model.predict(np.array([np.stack((s[0],s[1],s[2],s[3]),axis=2)]))[0])

env = gym.make('Breakout-v0')
model = load_model('model.h5')
done = False
state = []
observation = env.reset()
update_state(state,observation)

while not done:
	env.render()
	if len(state) < 4:
		action = env.action_space.sample()
	else:
		action = sample_action(model,state)
	observation, reward, done, _ = env.step(action)
	update_state(state,observation)