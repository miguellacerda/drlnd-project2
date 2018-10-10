## Udacity Deep Reinforcement Learning Nanodegree 
## Project 2: Continuous Control

### Description of Implementation

The environment is solved using a multi-agent deep deterministic policy gradient (DDPG) algorithm. Training proceeds as follows:

1. The 20 agents each receive a different state vector (with 33 elements) from the environment
1. Each agent feeds its state vector through the local actor network to get an action vector (with 4 elements) as output. Noise based on an Ornstein-Uhlenbeck process is added to the predicted actions to encourage exploration
1. Each agent then receives a next state vector and a reward from the environment (as well as a termination signal that indicates if the episode is complete)
1. The experience tuple `(state, action, reward, next state)` of each agent is added to a common replay buffer
1. A random sample of experience tuples is drawn from the replay buffer (once it contains enough) 
1. The sample is used to update the weights of the local critic network:
    1. The next state is fed into the target actor to obtain the next action
    1. The (next state, next action) pair is fed into the target critic to obtain an action value, Q_next
    1. The action value for the current state is then computed as Q_cur = reward + gamma*Q_next
    1. The (current state, current action) pair are fed into the local critic to obtain a predicted action value, Q_pred
    1. The MSE loss is computed between Q_cur and Q_pred, and the weights of the local critic are updated accordingly
1. The sample is used to update the weights of the local actor network:
    1. The current state is fed into the local actor to obtain predicted a action
    1. Each (current state, predicted action) pair for the sample is fed into the local critic to obtain action values
    1. The negative mean of the predicted Q values is used as a loss function to update the weights of the local actor
1. The target actor and critic networks are then soft-updated by adjusting their weights slightly toward those of their local counterparts
1. The states that were obtained in step (3) then become the current states for each agent and the process repeats from step (2)

#### Learning Algorithms

This project considers a multi-agent implementation of the DDPG algorithm.

#### Agent Hyperparameters

- `GAMMA = 0.99` is the discount factor that controls how far-sighted each agent is with respect to rewards. `GAMMA = 0` implies that only the immediate reward is important and `GAMMA = 1.0` implies that all rewards are equally important, irrespective whether they are realised soon and much later
- `TAU = 0.001` controls the degree to which the target network parameters are adjusted toward those of the local network. `TAU = 0` implies no adjustment (the target network does not ever learn) and `TAU = 1` implies that the target network parameters are completely replaced with the local network parameters
- `LR_ACTOR = 0.0001` is the learning rate for the gradient descent update of the local actor weights
- `LR_CRITIC = 0.001` is the learning rate for the gradient descent update of the local critic weights
- `BUFFER_SIZE = 100000` is the number of experience tuples `(state, action, reward, next_state, done)` that are stored in the replay buffer and avaiable for learning
- `BATCH_SIZE = 128` is the number of tuples that are sampled from the replay buffer for learning
- During training, the predicted actions were corrupted with noise based on an Ornstein-Uhlenbeck process with mean `mu = 0`, mean reversion rate `theta = 0.15` and variance `sigma = 0.2` 


#### Network Architectures and Hyperparameters

The actor network takes a state vector (33 elements) as input and returns an action vector (4 elements). It was modelled with a feedforward deep neural network comprising a 33 dimensional input layer, two hidden layers with 128 neurons and ReLU activations and a 4 dimensional output layer with a tanh activation to ensure that the predicted actions are in the range -1 to +1. Batch normalisation was applied to the input and two hidden layers. 

The critic network takes the state and action vectors as input, and returns a scalar Q value as output. It was modelled with a feedforward deep neural network with a 33 dimensional input layer (for the state vector) that was fully connected to 128 neurons in the first hidden layer with ReLU activations. The outputs of the first layer were batch normalised and concatenated with the 4 dimensional action vector as input to the second hidden layer, which also comprised 128 neurons with ReLU activations. Finally, the second hidden layer mapped to an output layer with single neuron and linear activation (outputs a real number). 


### Results

The 20 agents were trained for 500 episodes. After the first 100 episodes, the average score was 30.91 and the environment was therefore solved in the least amount of time (episodes 1 to 100). The best average score of 37.11 was obtained after 160 episodes (averaged over episodes 61 to 160). The actor and critic weights corresponding to these agents is stored in the code folder. Based on the plot below, it does not appear that training the agents for any longer would improve their performance further. The dashed vertical line indicates the point at which the enviroment was considered solved.

![scores.png](scores.png)

Here are the trained agents in action:

![trained_agents.gif](trained_agents.gif)


### Future Plans for Improvement

The performance of the agent might be improved by considering the following:

- Hyperparameter optimisation 

  Many of the hyperparameters listed above were treated as fixed. These could be tuned to improve performance.

- Duelling DQN or other learning algorithms

  One could consider using a duelling DQN that predicts the state-value function *V(s)* and advantages *A(s,a)* of taking each action within a given state. One could also consider algorithms other than DQNs. 

Further work is also required to optimise the code for training. In particular, the current implementation of prioritised replay is very slow and additional work is required to make this more computationally efficient.
