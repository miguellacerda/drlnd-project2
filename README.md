## Udacity Deep Reinforcement Learning Nanodegree 
## Project 2: Continuous Control

### Description of Environment

In this project, 20 identical double-jointed arms (agents) must be trained to keep their hands in a target location for as long as possible. Each agent receives a reward of +0.1 for each time step that its hand is in the goal location. 

The state space for each agent has 33 dimensions corresponding to the position, rotation, velocity and angular velocities of the arm. The action set contains four continuous numbers ranging between -1 and +1, which represent the torque of the two arm joints. 

An episode was set to last for 1000 time steps. At the end of each episode, the average reward across all 20 agents was computed. The environment was considered solved when the mean of the average rewards over the last 100 consecutive episodes was greater than 30.


### Installation Instructions and Dependencies

The code is written in PyTorch and Python 3.6. I trained the agents using the Udacity workspace with GPU enabled. To run the code in this repository on a personal computer, follow the instructions below:

1. Create and activate a new environment with Python 3.6
    
   ###### Linux or Mac:
   
    `conda create --name drlnd python=3.6`
    
    `source activate drlnd`

   ###### Windows:

    `conda create --name drlnd python=3.6`
    
    `activate drlnd`

1. Install of OpenAI gym in the environment

   `pip install gym`
 
1. Install the classic control and box2d environment groups

   `pip install 'gym[classic_control]'`
   
   `pip install 'gym[box2d]'`

1. Clone the following repository and install the additional dependencies

   `git clone https://github.com/udacity/deep-reinforcement-learning.git`
   
   `cd deep-reinforcement-learning/python`
   
   `pip install .`

1. Download the Unity environment (available [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip) for macOS)


### Code Description

The environment described above was solved using an actor-critic algorithm, specifically the multi-agent deep deterministic policy gradient (DDPG) method. All of the files below can be found in the code folder.

#### Module Descriptions

- `model.py` defines the actor and critic networks
- `ddpg_multiple_agents.py` defines the DDPG agents

#### Training the Agents

- `ddpg_multiple_agents_training.ipynb` is used to train the multi-agent DDPG model

#### The Trained Actor and Critic Networks

- `checkpoint_actor_best.pth` contains the weights of the best actor network (see Report.md)
- `checkpoint_critic_best.pth` contains the weights of the best critic network (see Report.md)
- `trained_agents.py` loads the optimised network weights and runs the trained agents for 1 episode

   

