import gym
import cv2
from collections import deque, namedtuple
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam

conv = namedtuple('Conv', 'filter kernel stride')

class Buffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)
        
    def add(self, s, a, r, s2, t):
        s = np.stack((s[0], s[1], s[2], s[3]), axis=2)
        s2 = np.stack((s2[0], s2[1], s2[2], s2[3]), axis=2)
        self.buffer.appendleft((s, a, r, s2, t))

    def sample(self, batch_size):
        return random.sample(list(self.buffer), batch_size)

class DQN:
    def __init__(self, buff, batch_size=32, min_buff=10000, gamma=0.99, learning_rate=2.5e-4):
        self.buffer = buff
        self.min_buffer = min_buff
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.model = create_network(learning_rate)
        self.target_model = create_network(learning_rate)
        self.copy_network()

    def train(self):
        if len(self.buffer.buffer) < self.min_buffer:
            return
        
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, terminal = map(np.array, zip(*batch))
        
        next_state_action_values = np.max(self.target_model.predict(next_states, verbose=0), axis=1)
        targets = self.model.predict(states, verbose=0)
        targets[range(self.batch_size), actions] = rewards + self.gamma * next_state_action_values * np.invert(terminal)
        self.model.train_on_batch(states, targets)

    def copy_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def predict(self, x):
        s = np.stack((x[0], x[1], x[2], x[3]), axis=2)
        return self.model.predict(np.array([s]), verbose=0)

def create_network(learning_rate, conv_info=[conv(32,8,4), conv(64,4,2), conv(64,3,1)], dense_info=[512], input_size=(84,84,4)):
    model = Sequential()
    for i, cl in enumerate(conv_info):
        if i == 0:
            model.add(Conv2D(cl.filter, cl.kernel, padding="same", strides=cl.stride, activation="relu", input_shape=input_size))
        else:
            model.add(Conv2D(cl.filter, cl.kernel, padding="same", strides=cl.stride, activation="relu"))
    model.add(Flatten())
    for dl in dense_info:
        model.add(Dense(dl, activation="relu"))
    # Breakout has 4 possible actions: NOOP, FIRE, RIGHT, LEFT
    model.add(Dense(4))
    adam = Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=adam)
    return model

class Breakout:
    def __init__(self):
        self.env = gym.make('ALE/Breakout-v5', render_mode='human')
        self.epsilon = 1.0
        self.buffer = Buffer(100000)  # Increased buffer size for Breakout
        self.dqn = DQN(self.buffer)
        self.copy_period = 10000
        self.itr = 0
        self.eps_step = 0.0000009
        self.fire_reset = True  # Flag to handle FIRE action at reset

    def sample_action(self, s):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.dqn.predict(s)[0])

    def play_one_episode(self):
        observation, _ = self.env.reset()
        done = False
        state = []
        prv_state = []
        total_reward = 0
        truncated = False
        lives = 5  # Breakout starts with 5 lives
        
        # Initial FIRE to start the game
        if self.fire_reset:
            observation, _, _, _, _ = self.env.step(1)  # FIRE action
        
        while not (done or truncated):
            if len(state) < 4:
                action = 1 if self.fire_reset else self.env.action_space.sample()  # FIRE for first action
            else:
                action = self.sample_action(state)
            
            prv_state = state.copy()
            if len(prv_state) > 4:
                prv_state = prv_state[-4:]
                
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Handle loss of life
            current_lives = info.get('lives', 5)
            if current_lives < lives:
                lives = current_lives
                if lives > 0:  # If game isn't over, fire to restart
                    observation, _, _, _, _ = self.env.step(1)
            
            done = terminated
            
            processed_obs = downsample(observation)
            state.append(processed_obs)
            if len(state) > 4:
                state.pop(0)
                
            if len(state) == 4 and len(prv_state) == 4:
                self.buffer.add(prv_state, action, reward, state, done)
                
            total_reward += reward
            
            self.itr += 1
            if self.itr % 4 == 0:
                self.dqn.train()
            self.epsilon = max(0.1, self.epsilon - self.eps_step)
            if self.itr % self.copy_period == 0:
                self.dqn.copy_network()
                
        return total_reward

def downsample(observation):
    # Convert to grayscale
    s = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84 (standard for DQN Breakout)
    s = cv2.resize(s, (84, 84), interpolation=cv2.INTER_AREA)
    # Normalize
    s = s / 255.0
    return s

def main():
    b = Breakout()
    try:
        for i in range(100000):
            total_reward = b.play_one_episode()
            print(f"Episode {i} total reward:", total_reward)
            if i % 100 == 0:
                print("Saving the model")
                b.dqn.model.save(f"breakout_model-{i}.h5")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving final model...")
        b.dqn.model.save("breakout_model_final.h5")

if __name__ == "__main__":
    main()