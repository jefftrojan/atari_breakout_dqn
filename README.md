# Atari Breakout Reinforcement Learning Agent

## Overview

This project implements a reinforcement learning agent designed to play the classic arcade game **Breakout** using OpenAI's Gym environment and Deep Q-Learning (DQN) techniques. The agent learns to control a paddle to break bricks by bouncing a ball against them.

## Project Structure

### 1. `train.py`

This file is responsible for training the DQN agent to play Breakout. It includes the following key components:

- **Buffer Class**: Implements a replay buffer to store experiences (state, action, reward, next state, done).
- **DQN Class**: Defines the DQN agent, including the neural network architecture and training methods.
- **Breakout Class**: Manages the game environment, including action sampling, episode management, and experience collection.
- **Downsample Function**: Preprocesses the game frames for input into the neural network.
- **Main Function**: Initializes the Breakout game and starts the training loop.

### 2. `test.py`

This file is used to evaluate the performance of the trained DQN agent in the Breakout game. Key components include:

- **Downsample Function**: Similar to the one in `train.py`, it preprocesses the game frames.
- **Update State Function**: Updates the state with the latest observation.
- **Sample Action Function**: Chooses an action based on the model's predictions.
- **Game Loop**: Runs the game, rendering the environment and executing actions based on the agent's policy.


### 3. `play.py`

This file is used to play the game using a trained DQN model. It loads the model and plays a specified number of episodes, with a delay between actions.

To play the game, run:

```bash
python play.py --model_path <path_to_model> --episodes <number_of_episodes> --delay <delay_between_actions>
```

### 4. `breakout.py`

This file implements the Breakout game using Tkinter. It includes:

- **BreakoutGame Class**: Manages the game state, including paddle and ball movement, collision detection, and score tracking.
- **AI Paddle Movement**: Placeholder for integrating AI to control the paddle.
- **Game Logic**: Handles ball movement, collision detection, and game state updates.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries:
  - `gym`
  - `opencv-python`
  - `numpy`
  - `keras`
  - `tkinter` (for Breakout game)

You can install the required libraries using pip:

```bash
pip install gym opencv-python numpy keras
```

### Training the Agent

To train the DQN agent for Breakout, run the following command:

```bash
python train.py
```

This will start the training process, and the model will be saved periodically.

### Testing the Agent

To test the trained agent, run:

```bash
python test.py
```

This will load the pre-trained model and allow the agent to play the Breakout game.

### Playing Breakout

To run the Breakout game, execute:

```bash
python breakout.py
```

This will open a window where you can play the game. The AI paddle movement logic can be integrated into this file.

## Code Definitions

### Key Classes and Functions

- **Buffer**: 
  - `__init__(size)`: Initializes the buffer with a specified size.
  - `add(s, a, r, s2, t)`: Adds a new experience to the buffer.
  - `sample(batch_size)`: Samples a batch of experiences from the buffer.

- **DQN**:
  - `__init__(buff, batch_size, min_buff, gamma, learning_rate)`: Initializes the DQN agent with a replay buffer and neural network.
  - `train()`: Trains the model using a batch of experiences.
  - `copy_network()`: Copies weights from the model to the target model.
  - `predict(x)`: Predicts the action values for a given state.

- **Breakout**:
  - `__init__()`: Initializes the Breakout environment and DQN agent.
  - `sample_action(s)`: Samples an action based on the current state.
  - `play_one_episode()`: Plays one episode of the game, collecting experiences.

- **downsample(observation)**: Preprocesses the observation by converting it to grayscale, cropping, resizing, and normalizing.

- **main()**: The entry point for training the agent.


