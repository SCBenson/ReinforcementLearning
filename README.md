# Reinforcement Learning Agent with Deep Q-Network (DQN)

## Overview
This implementation provides a reinforcement learning agent using a Deep Q-Network (DQN) to navigate in a 2D environment. The agent learns to find the shortest path to a goal by iteratively exploring the environment and improving its policy.

## Components

### Agent Class
The main class that handles interactions with the environment. Key functionalities:
- Initialization of Q-network, replay buffer, and learning parameters
- Action selection using epsilon-greedy policy
- Conversion between discrete and continuous action spaces
- Storage of transitions in the replay buffer
- Processing rewards and updating the Q-network
- Epsilon decay over time using a cosine schedule

### ReplayBuffer Class
Stores transitions (state, action, reward, next_state) for experience replay:
- Uses a double-ended queue with a maximum size of 100,000 transitions
- Allows sampling of transitions for training

### Network Class
A neural network implementation using TensorFlow:
- Two hidden layers with 100 units each and ReLU activation
- Input dimension of 2 (for 2D state space)
- Output dimension of 4 (for the four possible discrete actions)

### DQN Class
Handles the training of the Q-network:
- Maintains both a primary Q-network and a target Q-network
- Updates the target network periodically to stabilize learning
- Implements loss calculation using the Bellman equation
- Uses Adam optimizer for gradient updates

## Action Space
The agent can take four discrete actions that are converted to continuous movements:
- 0: Move left (-0.02, 0)
- 1: Move right (0.02, 0)
- 2: Move up (0, 0.02)
- 3: Move down (0, -0.02)

## Learning Process
1. The agent explores the environment using epsilon-greedy policy
2. Experiences are stored in the replay buffer
3. After collecting sufficient data, mini-batches are sampled for training
4. The Q-network is updated to minimize the temporal difference error
5. The target network is periodically updated to match the Q-network
6. Epsilon decreases over time according to a cosine decay schedule

## Reward Structure
Rewards are based on the distance to the goal:
- Base reward: 1 - distance_to_goal
- Higher rewards for being closer to the goal
- Scaled rewards for different distance thresholds

## Hyperparameters
- Episode length: 230 steps
- Buffer size before training: 120 transitions
- Mini-batch size: 100 transitions
- Target network update frequency: Every 55 steps
- Discount factor (gamma): 0.99
- Learning rate: 0.005
- Initial epsilon: 1.0

## Usage
This agent is designed to be used in a compatible environment that provides:
- A 2D state representation
- Distance to goal measurement
- The ability to apply continuous actions

The agent can be integrated into a simulation or training loop by calling its methods in the following order:
1. `get_next_action(state)` to get the next action to take
2. Apply the action in the environment to get next_state and distance_to_goal
3. `set_next_state_and_distance(next_state, distance_to_goal)` to update the agent
4. `has_finished_episode()` to check if the episode has ended

For evaluation, the `get_greedy_action(state)` method can be used to get the best action without exploration.
