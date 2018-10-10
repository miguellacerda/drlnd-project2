from unityagents import UnityEnvironment
from ddpg_multiple_agents import Agents
import torch
import numpy as np

### ENVIRONMENT ###
env = UnityEnvironment(file_name='Reacher.app', seed=0)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = env_info.vector_observations.shape[1]

agents = Agents(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=12)


### load weights ####
actor_dict = torch.load('checkpoint_actor_best.pth', map_location={'cuda:0': 'cpu'})
critic_dict = torch.load('checkpoint_critic_best.pth', map_location={'cuda:0': 'cpu'})

agents.actor_local.load_state_dict(actor_dict)
agents.actor_target.load_state_dict(actor_dict)

agents.critic_local.load_state_dict(critic_dict)
agents.critic_target.load_state_dict(critic_dict)


####################

states = env_info.vector_observations
agents.reset() # set the noise to zero
score = np.zeros(num_agents)
for t in range(1000):
    actions = agents.act(states, add_noise=False) 
    env_info = env.step(actions)[brain_name]
    next_states = env_info.vector_observations             # get the next states
    rewards = env_info.rewards                             # get the rewards
    dones = env_info.local_done                            # see if the episode has finished for any agent
    agents.step(states, actions, rewards, next_states, dones)
    states = next_states
    score += rewards
    if np.any(dones):
        break 

env.close()
print('score = ', np.mean(score))

