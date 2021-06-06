

import time
import retro
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math

import sys
sys.path.append('../../')
from a2c_agent import A2CAgent # algos.agents
#from algos.models import ActorCnn, CriticCnn
from actor_critic_cnn import ActorCnn, CriticCnn
from stack_frame import preprocess_frame, stack_frame #algos.preprocessing.

#env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='contest')
env =  retro.make(game='Airstriker-Genesis', state='Level1')
env.seed(0)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
env.reset()
'''
plt.figure()
plt.imshow(env.reset())
plt.title('Original Frame')
plt.show()
'''
possible_actions = {
            # No Operation
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Left
            1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            # Right
            2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            # Left, Down
            3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            # Right, Down
            4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            # Down
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # Down, B
            6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            # B
            7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }


def random_play():
    score = 0
    env.reset()
    for i in range(200):
        env.render()
        action = possible_actions[np.random.randint(len(possible_actions))]
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            print("Your Score at end of game is: ", score)
            break
    env.reset()
    env.render(close=True)
#random_play()

'''
plt.figure()
plt.imshow(preprocess_frame(env.reset(), (1, -1, -1, 1), 84), cmap="gray")
plt.title('Pre Processed image')
plt.show()
'''

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames


INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)
SEED = 0
GAMMA = 0.99           # discount factor
ALPHA= 0.0001          # Actor learning rate
BETA = 0.0005          # Critic learning rate
UPDATE_EVERY = 100     # how often to update the network

agent = A2CAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, UPDATE_EVERY, ActorCnn, CriticCnn)

'''
env.viewer = None
# watch an untrained agent
state = stack_frames(None, env.reset(), True)
for j in range(200):
    env.render(close=False)
    action, _, _ = agent.act(state)
    next_state, reward, done, _ = env.step(possible_actions[action])
    state = stack_frames(state, next_state, False)
    if done:
        env.reset()
        break
env.render(close=True)
'''

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)

SAVE_PATH = 'model.chkpt'
def save_model(agent, path=SAVE_PATH):
    actor, critic = agent.actor_net, agent.critic_net
    torch.save( dict(actor=actor.state_dict(), critic=critic.state_dict()), path)

def load_model(agent, path=SAVE_PATH):
    actor, critic = agent.actor_net, agent.critic_net
    ckp = torch.load(path)
    state_dict = ckp.get('actor')
    actor.load_state_dict(state_dict)
    state_dict = ckp.get('critic')
    critic.load_state_dict(state_dict)

load_model(agent)

def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    for i_episode in range(start_epoch + 1, n_episodes+1):

        state = stack_frames(None, env.reset(), True)
        score = 0

        # Punish the agent for not moving forward
        prev_state = {}
        steps_stuck = 0
        timestamp = 0
        while timestamp < 10000:
            env.render()
            action, log_prob, entropy = agent.act(state)
            next_state, reward, done, info = env.step(possible_actions[action])
            score += reward

            timestamp += 1
            # Punish the agent for standing still for too long.
            if (prev_state == info):
                steps_stuck += 1
            else:
                steps_stuck = 0
            prev_state = info

            if (steps_stuck > 20):
                reward -= 1

            next_state = stack_frames(state, next_state, False)
            agent.step(state, log_prob, entropy, reward, done, next_state)
            state = next_state
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        clear_output(True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        #plt.show()
        plt.savefig("training.jpg")
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        save_model(agent)

    return scores

scores = train(1000)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.viewer = None

# watch an untrained agent
state = stack_frames(None, env.reset(), True)
for j in range(10000):
    env.render(close=False)
    action, _, _ = agent.act(state)
    next_state, reward, done, _ = env.step(possible_actions[action])
    state = stack_frames(state, next_state, False)
    if done:
        env.reset()
        break
env.render(close=True)
