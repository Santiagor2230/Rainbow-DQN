# Rainbow-DQN
implementation of Rainbow DQN

# Requirements
gym[atari,accept-rom-license] == 0.23.1

pytorch-lightining == 1.6.0

stable-baseline3

torch == 2.0.1

# Collab installations
!apt-get install -y xvfb

!pip install \
  gym[atari,accept-rom-license]==0.23.1 \
  pytorch-lightning==1.6.0 \
  stable-baselines3 \
  pyvirtualdisplay


# Description
Rainbow-DQN is a combination of Double-DQN + Dueling-DQN + Noisy-DQN + N_step-DQN + Distributional-DQN + PER-DQN. This architecture implements a distribution method of calculting the actions of the agent and in this way it can approximate closely to actions that give a higher chance of reward we also take into consideration the weights being noisy for exploration purposes, taking into consideration a copy network as a target that will update its weights over certain epochs, using Prioritized Experience Replay for prioritize data, taking into consideration both the advantage and value function as well as using number of steps to get a specific return.

# Game
Qbert

# Architecture
Rainbow DQN

# optimizer
AdamW

# Loss
Categorical Cross-entropy Loss + Huber Loss

# Video Result:



https://github.com/Santiagor2230/Rainbow-DQN/assets/52907423/69d05362-8542-4d12-a146-d9924743aa22

