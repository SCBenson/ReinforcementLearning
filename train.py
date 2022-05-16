import time
import numpy as np

from enviro import Environment
from priority import Agent
from matplotlib import pyplot as plt


# Main entry point
if __name__ == "__main__":

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(1606674828)
    #1606674828
    #np.random.seed(random_seed)

    # Create a random environment
    environment = Environment(magnification=500)

    # Create an agent
    priority = Agent()
    

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600
    
    plt.ion()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
    ax1.set(xlabel="Episode", ylabel="Average Loss", title="Loss Curve")
    ax1.set_yscale('log')
    ax2.set(xlabel="Episode", ylabel="Average Reward", title="Reward Curve")
    ax3.set(xlabel="Episode", ylabel="Epsilon", title="Epsilon Decay")


    display_count = 0

    # Train the agent, until the time is up
    while time.time() < end_time:
        # If the action is to start a new episode, then reset the state
        if priority.has_finished_episode():
            state = environment.reset()
        # Get the state and action from the agent
        action = priority.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        priority.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment and curves
        if display_on and display_count % 200 == 0:
            display_count = 0
            environment.show(state)
            ax1.plot(priority.avg_loss)
            ax2.plot(priority.avg_reward)
            ax3.plot(priority.epsilon_list)
            #fig.savefig("learning.png")
        display_count += 1
    

    # Test the agent for 100 steps, using its greedy policy
    state = environment.reset()
    has_reached_goal = False
    for step_num in range(100):
        action = priority.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        #environment.show(state)
        priority.distance.append(distance_to_goal)
        #print(f'I am at {state} going to {next_state} with action {action} DtG:{distance_to_goal}')
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        state = next_state
    
    environment.show(state, True)

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')

    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))

