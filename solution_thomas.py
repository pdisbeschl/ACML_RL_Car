import gym
slow = False
import numpy as np

env = gym.make("MountainCar-v0")
max_steps = 100000
learning_rate = 0.9

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=max_steps,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')

vel_space=np.around(np.arange(-0.07,0.071,0.01),3)
pos_space=np.around(np.arange(-1.2,0.61,0.1),2)

state_space=np.zeros((len(pos_space),len(vel_space),3))

converge = False
ctr = 0
while (converge==False and ctr < 20):
    ctr+=1
    observation = env.reset()
    state_action_obs = [];
    done = False
    timesteps = 0
    final_reward = 0;
    while not done:
        if slow: env.render()
            
        pos_bin=np.digitize(observation[0],pos_space)-1
        vel_bin=np.digitize(observation[1],vel_space)-1
        
        action = env.action_space.sample() # your agent here (this takes random actions)
        
        state_action_obs.append([pos_bin,vel_bin,action])
        
        observation, final_reward, done, info = env.step(action)
        
        timesteps+=1
        if slow: print("observation")
        if slow: print("reward")
        if slow: print("done")
    if timesteps < max_steps: 
        final_reward=1.0
        
    print("Episode", ctr, "finished after", timesteps, "timesteps with reward",final_reward)  
    
    times_visited=np.zeros((len(pos_space),len(vel_space),3))
    
    for i in range(timesteps-1,-1,-1):
        
        state_action = state_action_obs[i]
        
        if i == timesteps-1:
            state_space[state_action[0]][state_action[1]][state_action[2]]+=final_reward
            times_visited[state_action[0]][state_action[1]][state_action[2]]+=1
            
        else:
            last_state_action = state_action_obs[i+1]
            alpha = 1/(1+times_visited[state_action[0]][state_action[1]][state_action[2]])
            state_space[state_action[0]][state_action[1]][state_action[2]] += alpha * ((final_reward+(learning_rate*state_space[last_state_action[0]][last_state_action[1]][last_state_action[2]]))-state_space[state_action[0]][state_action[1]][state_action[2]])
            times_visited[state_action[0]][state_action[1]][state_action[2]]+=1
            
    observation = env.reset()
    done = False
    timesteps = 0
    final_reward = 0;
    while not done:
        if slow: env.render()
        pos_bin=np.digitize(observation[0],pos_space)-1
        vel_bin=np.digitize(observation[1],vel_space)-1

        actions = np.array(state_space[pos_bin][vel_bin])
        action = np.argmin(actions)

        observation, final_reward, done, info = env.step(action)
        timesteps+=1
        if slow: print("observation")
        if slow: print("reward")
        if slow: print("done")
    print("Episode finished after ", timesteps, "timesteps.")
    
    if timesteps < max_steps: 
        converge=True
    
observation = env.reset()
done = False
timesteps = 0
final_reward = 0;
while not done:
    env.render()
    pos_bin=np.digitize(observation[0],pos_space)-1
    vel_bin=np.digitize(observation[1],vel_space)-1
    
    actions = np.array(state_space[pos_bin][vel_bin])
    action = np.argmin(actions)
    
    observation, final_reward, done, info = env.step(action)
    timesteps+=1
    if slow: print("observation")
    if slow: print("reward")
    if slow: print("done")
print("Episode finished after ", timesteps, "timesteps.")