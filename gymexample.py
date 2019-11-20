import gym
slow = True

#env = gym.make("MountainCar-v0")

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)
env = gym.make('MountainCarMyEasyVersion-v0')

for _ in range(10):
    observation = env.reset()
    done = False
    timesteps = 0
    while not done:
        if slow: env.render()
        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        timesteps+=1
        if slow: print observation
        if slow: print reward
        if slow: print done
    print "Episode finished after ", timesteps, "timesteps."
