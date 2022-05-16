############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import time
import collections
import random

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 230
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Action types
        self.actions = [0, 1, 2, 3]
        # Action probabilities
        self.probabilities = [0.1, 0.3, 0.3, 0.3]
        # Initialise epsilon value
        self.epsilon = 1
        # Initialise our Q-Network
        self.dqn = DQN()
        # Initialise the Replay Buffer
        self.buff = ReplayBuffer()
        # Step counter per episode
        self.step_tick = self.num_steps_taken
        # Episode Counter
        self.episode = 0
        # Q-update
        self.q_update = 55
        # Epoch List
        self.epoch = []
        # Losses per step in an episode
        self.losses = []
        # Average Losses per episode
        self.avg_loss = []
        # Average step loss
        self.average_loss = 0
        # Buffer Size before training the Q-network
        self.buffer_size = 120
        # Counter for when the one episode doesn't fit Buffer_size
        self.tick = 0
        # Time that the program started.
        self.start_time = time.time()
        #Time that has elapsed
        self.elapsed = 0
        # Epsilon list
        self.epsilon_list = []
        # Reward_list
        self.reward_list = []
        # Avg Reward per ep
        self.avg_reward = []


    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        
        # check if it is actually a new episode
        if self.step_tick == self.episode_length:                
            
            self.elapsed = int(time.time() - self.start_time)
            
            self.epsilon = self.epsilon_cosine_decay(self.elapsed)
            
            self.epsilon_list.append(self.epsilon)

            self.epoch.append(self.episode)
            
            print(f'Episode: {self.episode} Epsilon: {self.epsilon}')
            
            # Increase episode by 1
            self.episode+=1
            
            # Reset the step counter
            self.step_tick = self.num_steps_taken
            
            buffer_list = self.buff.buffer_list
            buff_size  = self.buffer_size
            
            if len(buffer_list) > buff_size:
                self.average_loss = sum(self.losses)/len(self.losses)
                self.avg_loss.append(self.average_loss)
            else:
                self.tick+=1
                self.avg_loss.append(0)
            
            self.avg_reward.append(sum(self.reward_list)/len(self.reward_list))

            return True
        else:
            return False

# Function for the agent to choose its next action
    def _choose_next_action(self, eps):
        
        if self.epsilon > random.uniform(0,1):
            
            discrete_action = np.random.choice(self.actions, 1, self.probabilities)[0]
        else:
            s_t = torch.tensor(self.state, dtype = torch.float32)
            q_epsilon = self.dqn.q_network.forward(s_t)
            q_epsilon = q_epsilon.detach().numpy()
            discrete_action = np.argmax(q_epsilon)
        # Return a random discrete action between 0 and 3.
        return int(discrete_action)
    
    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        
        if discrete_action == 0:
            # Move 0.1 leftwards
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        
        
        elif discrete_action == 1:
            # Move 0.1 rightwards
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        
        
        elif discrete_action == 2:
            # Move 0.1 upwards
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        
        
        elif discrete_action == 3:
            # Move 0.1 downwards
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        
        return continuous_action

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        
        # Choose the next action.
        discrete_action = self._choose_next_action(self.epsilon)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = discrete_action
        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        
        # Update the Target Q-Network every n episodes
        
        buff = self.buff
        
        buff_size = self.buffer_size
        
        self.step_tick+=1 
        
        if self.step_tick % self.q_update == 0:
            self.dqn.update_target_network()
           
        # Convert the distance to a reward
        if np.all(self.state == next_state):
            
            reward = (1 - distance_to_goal)/5
        
        # If the agent is around halfway to the goal increase the reward
        elif distance_to_goal < 0.5:
            reward = 1.05*(1 - distance_to_goal)
        elif distance_to_goal < 0.49:
            reward = 1.1*(1-distance_to_goal)
        elif distance_to_goal < 0.3:
            reward = 1.5*(1-distance_to_goal) 
        elif distance_to_goal < 0.1:
            reward = 2 * (1-distance_to_goal)
        elif distance_to_goal < 0.03:
            reward = 2.5 * (1-distance_to_goal)
        else:
            
            reward = 1 - distance_to_goal
        
        self.reward_list.append(reward)
        
        
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        #self.transition = transition
        
        buff_list = buff.append_transition(transition)
        
        if len(buff_list) > buff_size:
            
            minibatch_indices = np.random.choice(range(len(buff_list)), 100)
            
            minibatch_inputs = []
            
            for index in range(len(minibatch_indices)):
                minibatch_inputs.append(buff_list[minibatch_indices[index-1]])   
            
            loss = self.dqn.train_q_network(minibatch_inputs)
            
            loss_value = torch.tensor(loss).item()
            
            self.losses.append(loss_value)
            
            
                       
        
    def epsilon_cosine_decay(self, elapse):
        lam = 0.04
        A = 0.2
        return 0.8 + (-elapse/800) + A * np.exp(-lam*elapse) * np.cos(elapse)

# Function for the agent to choose its next action
    def _choose_next_greedy_action(self, state):
        
        s_t = torch.tensor(state, dtype = torch.float32)
        q_epsilon = self.dqn.q_network.forward(s_t)
        q_epsilon = q_epsilon.detach().numpy()
        discrete_action = np.argmax(q_epsilon)
        # Return a random discrete action between 0 and 3.
        return int(discrete_action)

    def get_greedy_action(self, state):
        self.state = state
        return self._discrete_action_to_continuous(self._choose_next_action(0))
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self.dqn.q_network.eval()
        with torch.no_grad():
            actions = self.dqn.q_network.forward(state_tensor).squeeze(0)

        self.dqn.q_network.train()
        return np.argmax(actions.numpy())



    # Function to get the greedy action for a particular state
    #def get_greedy_action_(self, state):
        
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        #s_t = torch.tensor(state, dtype = torch.float32)
        #greedy = self.dqn.q_network.forward(s_t)
        #greedy = greedy.detach().numpy()
        
        #discrete_action = np.argmax(greedy)
                
        #continuous_action = self._discrete_action_to_continuous(discrete_action)
        
        #return continuous_action

class ReplayBuffer:
    
    def __init__(self):
        self.buffer_list = collections.deque(maxlen=100000)
        
    def append_transition(self, transition):
        
        self.buffer_list.append(transition)
        
        return self.buffer_list

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Create a Q-Target Network
        self.qtarget_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.005)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, minibatch_inputs):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch_inputs)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()
    def update_target_network(self):
        
        self.qtarget_network.load_state_dict(self.q_network.state_dict())
    
    def _calculate_loss(self, minibatch_inputs):
        # Function to calculate the loss for a particular transition.
        
        # Transition = (state, action, reward, next state)
        gamma = torch.tensor([0.99], dtype=torch.float32)
        states = []
        actions = []
        rewards = []
        future_state = [] # will hold the future states for each iteration

        for t in range(0,100):
            
            states.append(minibatch_inputs[t][0])
            actions.append(minibatch_inputs[t][1])
            rewards.append(minibatch_inputs[t][2])
            future_state.append(minibatch_inputs[t][3])
        
        states_tensor = torch.tensor(states, dtype=torch.float32)
        
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        future_tensor = torch.tensor(future_state, dtype=torch.float32)
    
        
        # [100,1] Action tensor
        a=np.reshape(actions_tensor.unsqueeze(0),(100,1))
        
        # Gathers the Q-values for the current state given the action
        q_values = self.q_network.forward(states_tensor).gather(dim=1, index=a)
    
        
        # this gets the max Q for a future action
        q_target_values = self.qtarget_network.forward(future_tensor).max(1)[0]
        
        # this computes the bellman equation 
        q_target = rewards_tensor.squeeze(-1) + gamma * q_target_values
        
        
        loss = torch.nn.MSELoss()(q_values.squeeze(-1), q_target)
        
        return loss

