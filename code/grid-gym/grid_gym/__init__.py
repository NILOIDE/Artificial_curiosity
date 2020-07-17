from gym.envs.registration import register

register(id='GridWorld10x10-v0',
         entry_point='grid_gym.envs:GridWorld10x10',
         max_episode_steps=10*5
         )

register(id='GridWorld25x25-v0',
         entry_point='grid_gym.envs:GridWorld25x25',
         max_episode_steps=25*5
         )

register(id='GridWorld42x42-v0',
         entry_point='grid_gym.envs:GridWorld42x42',
         max_episode_steps=42*2
         )

register(id='GridWorldRandFeatures42x42-v0',
         entry_point='grid_gym.envs:GridWorldRandFeatures42x42',
         max_episode_steps=42*2
         )

register(id='GridWorldSubspace50x50-v0',
         entry_point='grid_gym.envs:GridWorldSubspace50x50',
         max_episode_steps=50*2
         )

register(id='GridWorldBox11x11-v0',
         entry_point='grid_gym.envs:GridWorldBox11x11',
         max_episode_steps=11*10
         )

register(id='GridWorldSpiral28x28-v0',
         entry_point='grid_gym.envs:GridWorldSpiral28x28',
         max_episode_steps=28*28
         )

register(id='GridWorldSpiral52x50-v0',
         entry_point='grid_gym.envs:GridWorldSpiral52x50',
         max_episode_steps=52*50
         )

