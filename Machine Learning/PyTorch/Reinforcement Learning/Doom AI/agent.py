import torch
from torch import nn
import copy
import numpy as np
import random

class DeepDuelingNeuralNetwork(nn.Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        activation_function = nn.ELU#nn.ReLU
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(num_features=32),
            activation_function(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=64),
            activation_function(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            activation_function(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            activation_function(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

from collections import deque
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)

        return [self.buffer[i] for i in index]


class Agent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000) # Memory(max_size=100000)
        self.batch_size = 32

        # Exploration parameters for epsilon greedy strategy
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999#0.99999975
        self.exploration_rate_min = 0.0001#0.1
        # Q learning hyperparameters
        self.gamma = 0.95

        self.curr_step = 0
        self.burnin = 1e4 #1e5  # min. experiences before training # Set to 100 for debugging
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

        self.save_every = 1e4#5e5   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        ### DNN #######################################################
        self.use_cuda = torch.cuda.is_available()
        self.net = DeepDuelingNeuralNetwork(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.
        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """

        explore_exploit_tradeoff = np.random.rand()

        # EXPLORE
        if explore_exploit_tradeoff < self.exploration_rate:
            action_index = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online') # predict
            action_index = torch.argmax(action_values, axis=1).item() # Choose item with greatest prob
            #print(action_values)
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_index

    ### Memory #####################################################
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()


    ### Learning #####################################################
    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        #print("Learning from memory")
        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)
        q = td_est.mean().item()
        return (q, loss)


    ### Save & Load ##################################################
    def save(self):
        if self.curr_step > 1e5 and self.save_every > 1e4:
            self.save_every = 5e2

        #save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        save_path = self.save_dir / f"doomguy_net.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        #if not load_path.exists():
        #    raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate

class DQNAgent(Agent):
    # Just make it such that the DQNAgent never uses the target model. The Target Q value is the actual reward?
    @torch.no_grad()
    def td_target(self, reward, next_state, done): # Target Function
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        #next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]

        # Using the Bellman Equation
        next_Q = next_state_Q[np.arange(0, self.batch_size), best_action] # Q_online(s,a)
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
