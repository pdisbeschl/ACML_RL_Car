import gym
import numpy as np
import seaborn as sb

heatmap = True

max_steps = 100000
gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=max_steps,      # MountainCar-v0 uses 200
)

env = gym.make('MountainCarMyEasyVersion-v0')


def initialise_Q_random(Q):
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            a = np.random.uniform(low = -1, high = 1, size=3)
            Q[i][j] = a
    return Q
            
            
def discretize_state(observation):
    pos, vel =  observation
    pos_bin = int(np.digitize(pos, pos_space))-1
    vel_bin = int(np.digitize(vel, vel_space))-1
    return [pos_bin, vel_bin]


def max_action(Q, state, actions=[0, 1, 2]):
    Q_values = Q[state[0]][state[1]]
    action = np.argmax(Q_values)
    return action

def update_Q_values(Q, executed_actions, reward):
    times_visited=np.zeros((len(pos_space),len(vel_space),3))

    last_state = executed_actions[-1][0]
    last_action = executed_actions[-1][1]
    Q[last_state[0]][last_state[1]][last_action] += reward

    times_visited[last_state[0]][last_state[1]][last_action] += 1 
    
    for i in range(len(executed_actions)-2, -1, -1):
        last_state = executed_actions[i+1][0]
        last_action = executed_actions[i+1][1]
        
        current_state = executed_actions[i][0]
        current_action = executed_actions[i][1]
        
        alpha = 1/(1+times_visited[current_state[0]][current_state[1]][current_action])
        delta = alpha*(reward + 0.9*Q[last_state[0]][last_state[1]][last_action] - Q[current_state[0]][current_state[1]][current_action])
        Q[current_state[0]][current_state[1]][current_action] += delta
        times_visited[current_state[0]][current_state[1]][current_action] += 1        
    return Q
        

vel_space = np.around(np.arange(-0.07,0.071,0.01),3)
pos_space = np.around(np.arange(-1.2,0.61,0.1),2)

Q = np.zeros((len(pos_space), len(vel_space), 3))
times_visited=np.zeros((len(pos_space),len(vel_space),3))

Q = initialise_Q_random(Q)
epsilon = 0.9
#training
for i in range(10):
    executed_actions = []
    total_reward = 0
    observation = env.reset()
    done = False
    timesteps = 0
    while not done:
        discrete_obs = discretize_state(observation)
        # Determine next action - epsilon greedy strategy
        #if np.random.random() < 1 - epsilon:
        #    action = np.argmax(Q[state_adj[0], state_adj[1]]) 
        #else:
        #    action = np.random.randint(0, env.action_space.n)

        # Determine next action - epsilon greedy strategy
        action = np.random.randint(0, env.action_space.n)

        if len(executed_actions) == 0 or not (executed_actions[-1] == [discrete_obs, action]):
            executed_actions.append([discrete_obs, action])        
        observation, reward, done, info = env.step(action)
        timesteps+=1
        if timesteps < max_steps:
            reward = 1
    Q = update_Q_values(Q, executed_actions, reward)    
    if timesteps < 2000:
        break

    print("Episode " + str(i) + " finished after " + str(timesteps) + " timesteps with reward: " + str(reward))


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

else: 
    #testing
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
    
