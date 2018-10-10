## Udacity Deep Reinforcement Learning Nanodegree 
## Project 2: Continuous Control

### Description of Implementation

The environment is solved using a multi-agent deep deterministic policy gradient (DDPG) algorithm. Training proceeds as follows:

1. The 20 agents each receive a different state vector (with 33 elements) from the environment
1. Each agent feeds its state vector through the local actor network to get an action vector (with 4 elements) as output
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
1. The states that was obtained in step (3) then becomes the current states for each agent and the process repeats from step (2)

#### Learning Algorithms

This project considers three variations of the deep Q-learning algorithm:

1. Vanilla DQN with fixed targets and experience replay
1. *Double* DQN with fixed targets and experience replay
1. *Double* DQN with fixed targets and *prioritised* experience replay

#### Agent Hyperparameters

##### All Deep Q-Networks

- `epsilon` controls the degree of exploration vs exploitation of the agent in selecting its actions. `epsilon = 0` implies that the agent is greedy with respect to the Q-network (pure exploitation) and `epsilon = 1` implies that the agent selects actions completely randomly (pure exploration). In this project, `epsilon` was initially annealed from 1.0 to 0.1 in steps of 0.001 after each episode, and remained fixed at 0.1 therafter. This was subsequently adapted to decay exponentially by a factor of 0.995 per episode from 1.0 to 0.01
- `GAMMA = 0.99` is the discount factor that controls how far-sighted the agent is with respect to rewards. `GAMMA = 0` implies that only the immediate reward is important and `GAMMA = 1.0` implies that all rewards are equally important, irrespective whether they are realised soon and much later
- `TAU = 0.001` controls the degree to which the target Q-network parameters are adjusted toward those of the local Q-network. `TAU = 0` implies no adjustment (the target Q-network does not ever learn) and `TAU = 1` implies that the target Q-network parameters are completelty replaced with the local Q-network parameters
- `LR = 0.0005` is the learning rate for the gradient descent update of the local Q-network weights
- `UPDATE_EVERY = 4` determines the number of sampling steps between rounds of learning (Q-network parameter updates)
- `BUFFER_SIZE = 10000` is the number of experience tuples `(state, action, reward, next_state, done)` that are stored in the replay buffer and avaiable for learning
- `BATCH_SIZE = 64` is the number of tuples that are sampled from the replay buffer for learning

##### Prioritised Experience Replay

- `e_priority = 0.01` is added to the absolute value of the TD error to ensure that none of the priorities are exactly zero. This ensures that all tuples in the replay buffer have a non-zero probability of being selected for training
- `a_priority` controls the extent to which the TD error influences the probability of selecting a tuple for training. `a_priority=0` implies that all tuples in the buffer have equal probability of selection, while `a_priority=1` implies pure priority (TD error-based) sampling. We set `a_priority = 0.6` as in [this paper](https://arxiv.org/pdf/1511.05952.pdf)
- `b_priority` controls the extent to which the biased sampling from the replay buffer is corrected in the gradient descent update. `b_priority = 0` implies no correction for bias and `b_priority = 1` implies complete bias correction. In this project, `b_priority` is increased from 0.4 to 1.0 in steps of 0.0005 after each episode, and remains fixed at 1.0 therafter (as done in [this paper](https://arxiv.org/pdf/1511.05952.pdf))


#### Model Architecture and Hyperparameters

The mapping from states to actions was modelled with a feedforward deep neural network with a 37-dimensonal input layer and a 4-dimensional output layer. The number of hidden layers, the number of neurons within each hidden layer and the activation functions applied to each layer are hyperparameters that must be selected. In this project, the ReLU activation function was used for all hidden layers and a linear activation function was applied to the output layer. Various network architectures were considered:

- 2 hidden layers with 64 neurons each
- 2 hidden layers with 128 neurons in the first layer and 68 neurons in the second layer
- 4 hidden layers with 256, 128, 64 and 32 neurons each (moving from the input layer to the ouput layer)


### Results

All agents were trained for 2000 episodes. The plots below show the average score obtained during training over the last 100 episodes for the DQN, double DQN and double DQN with prioritised experience replay algorithms, and for each of the three network architectures described above. The vertical lines indicate the number of episodes required for the average score to first equal or exceed 13.0.

![all_scores.png](all_scores.png)

The number of episodes required for the average score to first equal or exceed 13.0 and the maximum average observed over the 2000 training episodes are given in the table below (each cell reports episodes/max score).


| Architecture        | Vanilla DQN         | Double DQN          | Double DQN with PER |
| ------------------- | ------------------- | ------------------- | ------------------- |
| 64 x 64             | 919 / 14.46         | 958 / 14.52         | 1083 / 14.89        |
| 128 x 64            | 917 / 14.25         | 1055 / 14.58        | 1125 / 13.93        |
| 256 x 128 x 64 x 32 | 964 / 14.02         | 1056 / 14.44        | 1258 / 14.05        |

The results suggest that a vanilla DQN with a relatively small, fully-connected Q-network (two hidden layers with 64 neurons each) is sufficient to train a successful agent. Interestingly, prioritised experience replay takes longer to solve the environment than the other two algorithms. In all cases, the mean score exhibits large fluctuations even after it has plateaued. This suggests that the agent is spending too much time exploring, rather than exploiting the learned Q-network. The results above were obtained by allowing epsilon to decrease linearly from 1.0 to 0.1 in steps of 0.001. To reduce the degree of exploration, we next considered an agent with an epsilon that decays exponentially by a factor of 0.995 from 1.0 all the way down to 0.01. The results are shown below for the smallest neural network.

![final_scores.png](final_scores.png)

When epsilon decays at a faster rate and to a lower minimum value, the agent solves the environment significantly faster and obtains a much higher maximum average score over 2000 epsiodes. As before, the agent takes longer with prioritised experience replay and does not achieve a better average score.

| DQN Variant         | Required Episodes   | Max Average Score   | 
| ------------------- | ------------------- | ------------------- |
| Vanilla DQN         | 545                 | 16.64               |
| Double DQN          | 511                 | 16.81               |
| Double DQN with PER | 666                 | 16.60               |

Based on these results, we provide the weights that achieved the maximum average score of 16.81 after 1154 episodes of training with the double DQN algorithm and a fully connected, feedforward Q-network with 2 hidden layers with 64 neurons each. These can be loaded as described in the README.

Here is the trained agent in action:

![trained_agent.gif](trained_agent.gif)


### Future Plans for Improvement

The performance of the agent might be improved by considering the following:

- Hyperparameter optimisation 

  Many of the hyperparameters listed above were treated as fixed. These could be tuned to improve performance.

- Duelling DQN or other learning algorithms

  One could consider using a duelling DQN that predicts the state-value function *V(s)* and advantages *A(s,a)* of taking each action within a given state. One could also consider algorithms other than DQNs. 

Further work is also required to optimise the code for training. In particular, the current implementation of prioritised replay is very slow and additional work is required to make this more computationally efficient.
