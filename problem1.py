#elias vitell 0102057379

# Copyright [2025] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 2 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v3')
# If you want to render the environment while training run instead:
# env = gym.make('LunarLander-v3', render_mode = "human")


env.reset()

# Parameters

# instructions suggest 100 <= N <= 1 000
N_episodes = 100                             # Number of episodes (T_E)
discount_factor = 0.95                       # Value of the discount factor (GAMMA)
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

EPSILON_MAX = .99 # 0.99 suggested in instructions
EPSILON_MIN = .05 # 0.05 suggested in instructions
Z = .95 * N_episodes

# instructions suggest batch size N between 4 and 128
BATCH_SIZE = 128 # training batch size (N)
# LR = 3e-4 # pytorch tutorial chose 3e-4
LR = .0001 # instructions suggest 10^(-4) to 10^(-3)

# period to update target (C)

# instructions suggest buffer size of 5 000-30 000
BUFFER_SIZE = 30000 # buffer size (L)

C = BUFFER_SIZE/BATCH_SIZE

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
agent = RandomAgent(n_actions)

# ReplayBuffer

class Experience:
    def __init__(self,state, action,reward,next_state,done):
        self.state = state
        try:
            assert( not  (type(self.state) is torch.Tensor))
        except AssertionError as e:
            print("type", type(self.state))
            print("state", self.state)
        assert(type(self.state) is np.ndarray)
        self.action = action 
        self.reward = reward
        self.next_state = next_state
        assert( not  (type(self.next_state) is torch.Tensor))
        assert(type(self.next_state) is np.ndarray)
        self.done = done 

class ReplayBuffer:
    def __init__(self,max_len):
        self.buffer = []
        self.max_len = max_len
    
    def append(self, elem:Experience):
        assert (type(elem) is Experience)
        self.buffer.append(elem)
        if len(self.buffer) > self.max_len:
            self.remove()
    
    def remove(self):
        self.buffer.pop(0)
    
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, count):
        # print("sampling",len(self.buffer),count,self.max_len)
        batch = np.random.choice(a=self.buffer, size=count, replace=False)
        # print("batch", batch)
        
        states_l = [x.state for x in batch]
        actions_l = [x.action for x in batch]
        rewards_l = [x.reward for x in batch]
        next_states_l = [x.next_state for x in batch]
        dones_l = [x.done for x in batch]
        # print(f"states_l ({type(states_l)}):",states_l)
        states = torch.tensor(states_l, dtype=torch.float32)
        actions = torch.tensor(actions_l, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards_l, dtype=torch.float32)
        next_states = torch.tensor(next_states_l, dtype=torch.float32)
        dones = torch.tensor(dones_l, dtype=torch.float32)

        return states, actions, rewards, next_states, dones


class Net(torch.nn.Module):
    def __init__(self, observation_count, action_count, dim2):
        super().__init__()
        # self.layer_in = torch.nn.Linear(in_features=observation_count, out_features=dim2)
        # self.layer_act = torch.nn.Linear(in_features=dim2,out_features=dim2)
        # self.layer_out = torch.nn.Linear(in_features=dim2 ,out_features=action_count)

        # https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=observation_count, out_features=dim2),
            # torch.nn.ReLU(),
            # torch.nn.Linear(in_features=dim2,out_features=dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=dim2 ,out_features=action_count)
            
        )
    
    def forward(self,x):
        # print("x type", type(x))
        # print("x.dtype", x.dtype)
        # x = torch.tensor(x,dtype=torch.float32)
        # x = x.to(torch.float32)
        # relu = torch.nn.functional.relu

        # r1 = self.layer_in(x)
        # r2 = relu(r1)
        # r3 = self.layer_act(r2)
        # r4 = relu(r3)
        # r5 = self.layer_out(r4)

        r = self.layer(x)
        
        return r #.to(torch.float32)



def select(s, epsilon):
    # print("s type", type(s))
    rnd = np.random.random()
    b = rnd < epsilon

    if b:
        ret = torch.tensor([env.action_space.sample()])[0].item() # as suggested here: https://github.com/pytorch/tutorials/issues/474
    else:

        # print(type(s))
        if type(s) is np.ndarray:
            # ts = (torch.from_numpy(s))#.to(torch.float32)
            ts = torch.tensor([s],dtype=torch.float32)
        else:
            ts = ((s))#.to(torch.float32)
        with torch.no_grad():
            # print("ts",ts)
            nts = net(ts)
            # ret = nts.argmax().item()
            ret = int(torch.max(nts,dim=1)[1].item())
    # print(f"b={b} select returning {type(ret)}")

    return ret #.to(torch.float32) 


# intialisations
net = Net(observation_count=dim_state,action_count=n_actions,dim2=256)
# target_net = Net(observation_count=env.observation_space.shape[0],action_count=n_actions,dim2=dim_state)
buffer = ReplayBuffer(BUFFER_SIZE)

# optim = torch.optim.AdamW(params=net.parameters(),lr=LR)
optim = torch.optim.Adam(params=net.parameters(),lr=LR)

# add random
def random_sample()->Experience:
    r = lambda x=None: np.random.random(x)
    z = Experience(r(dim_state), np.random.choice(a=n_actions), r(), r(dim_state), False)
    return z 

RND_COUNT = 2000*0
for _ in range(RND_COUNT):
    buffer.append(random_sample())



# t = torch.Tensor([2])
# x = net.forward(t)
# print("done")
# exit()

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done, truncated = False, False
    state = env.reset()[0]
    # 
    # 
    # state = torch.tensor(state, dtype=torch.float32).to(torch.float32)
    total_episode_reward = 0.
    t = 0
    while not (done or truncated):
        # Take a random action
        # action = agent.forward(state).
        epsilon = max(
            EPSILON_MIN,
            EPSILON_MAX * np.pow((float(EPSILON_MIN)/float(EPSILON_MAX)),(float((i+1)-1 )/float(Z-1) ))

        )
        action = select(s=state,epsilon=epsilon)
        

        # Get next state and reward
        next_state, reward, done, truncated, _ = env.step(action)

        # types [next_state, reward, done, truncated,]: [<class 'numpy.ndarray'>, <class 'numpy.float64'>, <class 'bool'>, <class 'bool'>]
        # print("types [next_state, reward, done, truncated,]:" , [type(x) for x in [next_state, reward, done, truncated,]])
        # print(f"made one step, type(next_state)={type(next_state)}") # np.ndarray
        # next_state = next_state.to(torch.float32)
        # reward = reward.to(torch.float32)
        # # done = done.to(torch.float32)
        # trunxated = truncated.to(torch.float32)

        # add z = (s_t, a_t, r_t, s_[t+1], d_t)
        z = Experience(state, action, reward, next_state,done)
        # print( "types state, action, reward, next_state,done", [type(x) for x in [state, action, reward, next_state,done]])
        buffer.append(z)
        # state = next_state
        # This part is very much like excercise session 3 solutions
        if len(buffer) >= BATCH_SIZE:
            sample_batch = buffer.sample(BATCH_SIZE)
            # for s in sample_batch:
            #     print("sb", s.dtype)
            states, actions, rewards, next_states, dones = sample_batch
            # print("shapes", states.shape, actions.shape)
            qt = net(states).gather(1, actions.to(torch.int32)).squeeze()
            with torch.no_grad():
                # print("")
                next_q = (net(next_states).max(1)[0]) #.to(torch.float32)

                # tgt: r V r + gamma * max Q(s_next, a)
                targets = (rewards + discount_factor * next_q * (1 - dones.to(int))) #.to(torch.float32)
            
            mse = torch.nn.functional.mse_loss(input = qt, target=targets)
            optim.zero_grad()
            mse.backward()
            torch.nn.utils.clip_grad_norm_(parameters= net.parameters(),max_norm=float(2))
            optim.step()


        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)


    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

# Close environment
env.close()

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.savefig("DQN_problem.png")
plt.show()

# torch.save(net.layer_out, 'neural-network-1.pth')
# torch.save(net.layer_in, 'neural-network-1.pth')
# torch.save(net.layer_out, 'neural-network-1.pth')
torch.save(net.layer, 'neural-network-1.pth')
