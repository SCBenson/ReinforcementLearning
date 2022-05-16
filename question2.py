import numpy as np
import torch
import time
import random
import cv2
from q_value_visualiser import QValueVisualiser
from matplotlib import pyplot as plt
from environment import Environment
import collections
# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()
        # List of actions
        self.actions = [0, 1, 2, 3] # 0 is left, 1 is right, 2 is up and 3 is down.
        
        self.dqn = DQN()
        
        self.epsilon = 1 # Initial epsilon value
        
        self.gamma = 0.9


    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0


    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_next_action(self):
        
        if self.epsilon > random.uniform(0,1):
            discrete_action = np.random.choice(self.actions)
        else:
            s_t = torch.tensor(self.state, dtype = torch.float32)
            q_epsilon = self.dqn.q_network.forward(s_t)
            q_epsilon = q_epsilon.detach().numpy()
            #print(q_epsilon)
            discrete_action = np.argmax(q_epsilon)
            #print(discrete_action)
        # Return a random discrete action between 0 and 3.
        return int(discrete_action)

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        
        if discrete_action == 0:
            # Move 0.1 leftwards
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        
        
        elif discrete_action == 1:
            # Move 0.1 rightwards
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        
        
        elif discrete_action == 2:
            # Move 0.1 upwards
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        
        
        elif discrete_action == 3:
            # Move 0.1 downwards
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        

        else:
            print("There is a problem with the input discrete_action")
        
        return continuous_action
            

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


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
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

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
        gamma = torch.tensor([0.9], dtype=torch.float32)
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
    
class ReplayBuffer:
    
    def __init__(self):
        self.buffer_list = collections.deque(maxlen=5000)
        
    def append_transition(self, transition):
        
        self.buffer_list.append(transition)
        
        return self.buffer_list
    
    #def minibatch_sampler(self, buff_list):
        # Choose a random transition tuple to sample.
        #minibatches = []
        #for index in buff_list:
            #sample = np.random.choice(len(buff_list))
            #minibatches.append(buff_list[sample])
        
        #return minibatches
        
# Main entry point
if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=False, magnification=500)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    qnet = agent.dqn
    tick = 0
    buff = ReplayBuffer()
    visualiser = QValueVisualiser(environment=environment, magnification=500)
    losses = []
    avg_loss = []
    iterations = []
    buff_size = 100
    epoch = [] #episode list for plotting
    ep=0 #episode counter
    set_episode_size = 100

    # Create a graph which will show the loss as a function of the number of training iterations
    fig, ax = plt.subplots()
    ax.set(xlabel='Episodes', ylabel='Loss', title='Loss Curve for Q3 (b)')

    # Loop over episodes
    while ep < set_episode_size:
        # Reset the environment for the start of the episode.

        #losses = []
        epoch.append(ep)
        agent.epsilon = max(0.05, 0.99*agent.epsilon)
        if ep % 10 == 0:
            qnet.update_target_network()
        ep+=1

        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(150):
                    
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            
            # Append the transition and return the updated collection.
            buff_list = buff.append_transition(transition)
            
            if len(buff_list) > buff_size:
                
                minibatch_indices = np.random.choice(range(len(buff_list)), 100)
                minibatch_inputs = []
                #print(minibatch_indices)
                for index in range(len(minibatch_indices)):
                    
                    minibatch_inputs.append(buff_list[minibatch_indices[index-1]])
                    
                
                # Train the Q Network with the minibatch
                loss = qnet.train_q_network(minibatch_inputs)
                #loss = agent.dqn.train_q_network(minibatch_inputs)              
                
                # Get the loss as a scalar value
                loss_value = torch.tensor(loss).item()
                
                # Store this loss in the list
                losses.append(loss_value)
                #time.sleep(0.2)
                
            else:
                pass
        
        if len(buff_list) > buff_size:
            average_loss = sum(losses)/len(losses)
            avg_loss.append(average_loss)
        else:
            tick+=1
            avg_loss.append(0)
        
        print(f"{ep}/{set_episode_size}")
        #if ep == set_episode_size:
            
            #else:
                #pass
                
    # Plot the loss at each episode versus the episodes.   
    ax.plot(epoch[tick:], avg_loss[tick:], color='blue')
    plt.yscale('log')
    plt.show()
    fig.savefig("dd.png", dpi=150)

        # Visualiser
        
### Plot Q values on discretised grid ###
    environment.display = True

    # Discretise the continuous world into a grid with a linearly discrete number of states
    disc = 10
    states = np.zeros((disc**2, 2))
    for s_idx in range(disc**2):
        si = (s_idx // disc) / disc + 0.05 # TODO: add 0.05 or no?
        sj = (s_idx % disc) / disc + 0.05 # TODO: add 0.05 or no?
        states[s_idx] = [si, sj]
    
    # Convert numpy state array to torch tensor
    states_tensor = torch.tensor(states, dtype=torch.float32)
    # Feed the discretised state tensor into the Q network to find the 4 Q values for each state, and convert to numpy.
    output = agent.dqn.q_network(states_tensor).detach().numpy()
    # Split Q network output into 2D grid-like dimensions
    q_values = np.array(np.split(output, len(output)/disc))

    # Create Q value visualiser and visualise Q values
    visualiser = QValueVisualiser(environment=environment, magnification=500)
    visualiser.draw_q_values(q_values)         

    
    ## Greedy Policy Visualisation
    agent.reset()
    agent.epsilon = 0
    greedy_steps = range(0,20)
    for g in greedy_steps:
        agent.step()
        time.sleep(0.5)

  


