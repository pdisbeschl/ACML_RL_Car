import gym
import numpy as np
import seaborn as sb

""" 
I would have liked to replace the car with an Audi Q model but havent had the time
https://www.audi.nl/nl/web/nl/modellen/q8/q8.html 
"""

""" Toggle this if you want to have a heatmap. """
heatmap = True

""" Toggle this to switch between random exploration and epsilon greedy. """
useEpsilon = True

""" Toggle the repetitions of the experiment. Use 1 if you want to see the result immeadeately. Use 100 to replicate the experiment """
repetitions = 1

#Maximum steps until we count it as a loss
max_steps = 100000
#Define the discretized space of velocities
vel_space = np.around(np.arange(-0.07,0.071,0.01),3)
#Define the discretized space of positions
pos_space = np.around(np.arange(-1.2,0.61,0.1),2)

#Initialise the Q-matrix
Q = np.zeros((len(pos_space), len(vel_space), 3))
episode_counter = 0

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=max_steps,      # MountainCar-v0 uses 200
)

env = gym.make('MountainCarMyEasyVersion-v0')

"""
    Initialises the Q-values with a random uniform values within the range [0,1]
    @param Q: The Q-matrix for all states
    @return: The updated Q-matrix with random values
"""
def initialise_Q_random(Q):
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            a = np.random.uniform(low = -1, high = 1, size=3)
            Q[i][j] = a
    return Q
            
"""
    Discretizes an observations by finding the closes discretized value
    @param observation: An observations containing position and velocity
    @return: the indices for the state of the Q matrix (position, velocity)
"""
def discretize_state(observation):
    pos, vel =  observation
    pos_bin = int(np.digitize(pos, pos_space))-1
    vel_bin = int(np.digitize(vel, vel_space))-1
    return [pos_bin, vel_bin]

"""
    Find the action with the highest Q-value for a given state
    @param Q: The Q-matrix containing the state-action pairs
    @param state: the current state
    @return: The action with the highest Q-value
"""
def max_action(Q, state):
    Q_values = Q[state[0]][state[1]]
    action = np.argmax(Q_values)
    return action

"""
    Iterate over the Q-matrix and update the values.
    @param Q: The matrix with the state-action pairs (Q-values)
    @param executed_actions: A list with all the actions that we took to get to the current state.
                             If we use epsilon greedy this is just the current and the previous state
                             If we use random exploration this contains all steps since we only update at the end
    @param reward: The reward for the last action (-1 if we don't reach the goal and proportional to the amount of maximal steps if we made it to the top)
    @return: The updated Q-matrix
"""
def update_Q_values(Q, executed_actions, reward):
    #To calculate our alpha we want to initialise a matrix to count the states that we visit
    times_visited=np.zeros((len(pos_space),len(vel_space),3))

    #If we use random exploration we want to update the last value aswell    
    if not useEpsilon:
        last_state = executed_actions[-1][0]
        last_action = executed_actions[-1][1]
        Q[last_state[0]][last_state[1]][last_action] += reward
    
        times_visited[last_state[0]][last_state[1]][last_action] += 1 
    
    #Update all state-action pairs that we have traversed to get to the goal
    for i in range(len(executed_actions)-2, -1, -1):
        last_state = executed_actions[i+1][0]
        last_action = executed_actions[i+1][1]
        
        current_state = executed_actions[i][0]
        current_action = executed_actions[i][1]
        
        #If we use random exploration compute alpha 
        alpha = 1/(1+times_visited[current_state[0]][current_state[1]][current_action])
        if useEpsilon:
            #Use a learning rate of 0.2 if we use epsilon greedy
            alpha = 0.2
        #Compute our update delta for the Q-value
        delta = alpha*(-1 + 0.9*Q[last_state[0]][last_state[1]][last_action] - Q[current_state[0]][current_state[1]][current_action])
        Q[current_state[0]][current_state[1]][current_action] += delta
        
        #Add one visit to our Q-matrix
        times_visited[current_state[0]][current_state[1]][current_action] += 1        
    return Q
        
avg_seps = 0
"""
    This checks if we converged. However, our definition of converged differs from the standard definitions- We do not check if the Q-values don't change
    Instead we check if we can reach the goal in less than 200 steps. If so we assume convergance. If might occasionally occur that the agent does not reach
    the goal.
    @return: If we can reach the goal in less than 200 actions
"""
def converged():
    observation = env.reset()
    done = False
    timesteps = 0
    while not done:
        discrete_obs = discretize_state(observation)
        action = max_action(Q, discrete_obs)        
        observation, reward, done, info = env.step(action)        
        timesteps+=1
        if timesteps > 200:
            break
    if timesteps < 200:
        global avg_seps 
        avg_seps += timesteps
        return True
    else:
        return False


#training
total_steps = 0
total_episodes = 0
for j in range(0,repetitions):
    epsilon = 0.2

    Q = initialise_Q_random(Q)
    episode_counter = 0
    print("Run" + str(j))
    tempsteps = 0

    #Repeat training until we converged
    while not converged():
        episode_counter += 1
        #Initialise the first executed action with the start
        executed_actions = [[[7,7],0]]
        total_reward = 0
        observation = env.reset()
        done = False
        timesteps = 0
        #Update epsilon to slowly reduce exploration
        epsilon = epsilon * 1.1
        while not done:
            #Discretize the observation
            discrete_obs = discretize_state(observation)
            
            #Epsilon greedy -  Choose the action with the highest Q-value. With some probability execute a random action
            if useEpsilon and np.random.rand() > 1 - epsilon:
                action = max_action(Q, discrete_obs)
            #Random exploration -  Choose a random action
            else:
                action = np.random.randint(0, env.action_space.n)
            
            #Add the most recent action. If we use random exploration we only want to update the action if we change state
            if useEpsilon or len(executed_actions) == 0 or not (executed_actions[-1] == [discrete_obs, action]):
                executed_actions.append([discrete_obs, action])        
            
            #Execute step
            observation, reward, done, info = env.step(action)
            timesteps+=1

            #If we are done we change our reward if we made it to the goal
            if done and timesteps < max_steps:
                reward = (1 - timesteps/max_steps)
            
            #Epsilon approach - update the last action
            if useEpsilon:
                last_actions = executed_actions[len(executed_actions)-2:len(executed_actions)]
                Q = update_Q_values(Q, last_actions, reward)    
        #Random expl
        if not useEpsilon:
            Q = update_Q_values(Q, executed_actions, reward)    
            
    
        print("Episode " + str(episode_counter) + " finished after " + str(timesteps) + " timesteps with reward: " + str(reward))
        tempsteps += timesteps
    total_steps += tempsteps/episode_counter
    
    print("Episodes until converged: " + str(episode_counter))
    total_episodes += episode_counter    
print("Average timesteps until converged: " + str(total_steps/repetitions) + " and episode_counter on average: " + str(total_episodes/repetitions)+ " Goal reached in average steps: " + str(avg_seps/repetitions))

#Draw a heatmap 
""" You might have to enable debug mode and debug in this as the heatmap is closed after closing the program """
if heatmap:
    best_move=np.zeros((len(pos_space),len(vel_space)))
    for i in range(0,len(pos_space)):
        for j in range(0,len(vel_space)):
            actions = np.array(Q[i][j])
            action = np.argmax(actions)
            best_move[i][j] = action
    ax = sb.heatmap(best_move,linewidths=.5,xticklabels=vel_space,yticklabels=pos_space,cbar_kws={'label': 'Best Move'})
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Position')
    
#If we don't draw a heatmap we draw show 10 runs of the agent with the computed Q-matrix
else: 
    for _ in range(10):
        observation = env.reset()
        done = False
        timesteps = 0
        while not done:
            env.render()
            discrete_obs = discretize_state(observation)
            action = max_action(Q, discrete_obs)        
            observation, reward, done, info = env.step(action)        
            timesteps+=1
        print(timesteps)
    
