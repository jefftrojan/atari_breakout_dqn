import gym
import cv2
import numpy as np
from keras.models import load_model
import argparse
import time

def downsample(observation):
    s = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    s = cv2.resize(s, (84, 84), interpolation=cv2.INTER_AREA)
    s = s / 255.0
    return s

class BreakoutPlayer:
    def __init__(self, model_path, render_mode='human'):
        self.env = gym.make('ALE/Breakout-v5', render_mode=render_mode)
        self.model = load_model(model_path)
        
    def get_action(self, state):
        s = np.stack((state[0], state[1], state[2], state[3]), axis=2)
        q_values = self.model.predict(np.array([s]), verbose=0)
        return np.argmax(q_values[0])
    
    def play_episode(self, delay=0.03):
        observation, _ = self.env.reset()
        state = []
        total_reward = 0
        done = False
        truncated = False
        lives = 5
        
        # Initial FIRE to start the game
        observation, _, _, _, _ = self.env.step(1)
        
        while not (done or truncated):
            if len(state) < 4:
                action = 1  # FIRE
            else:
                action = self.get_action(state)
            
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Handle loss of life
            current_lives = info.get('lives', 5)
            if current_lives < lives:
                lives = current_lives
                if lives > 0:
                    observation, _, _, _, _ = self.env.step(1)
            
            processed_obs = downsample(observation)
            state.append(processed_obs)
            if len(state) > 4:
                state.pop(0)
                
            total_reward += reward
            done = terminated
            
            if delay > 0:
                time.sleep(delay)
        
        return total_reward

def main():
    parser = argparse.ArgumentParser(description='Play Breakout using a trained DQN model')
    parser.add_argument('model_path', type=str, help='Path to the trained model file (.h5)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to play')
    parser.add_argument('--delay', type=float, default=0.03, help='Delay between actions (seconds)')
    
    args = parser.parse_args()
    
    player = BreakoutPlayer(args.model_path)
    
    for episode in range(args.episodes):
        total_reward = player.play_episode(delay=args.delay)
        print(f"Episode {episode + 1} - Total Reward: {total_reward}")

if __name__ == "__main__":
    main()