### Set up vizdoom environment ################################################
from vizdoom import *        # Doom Environment

DOOM_CONFIG = "doom/basic.cfg"
DOOM_SCENARIO = "doom/basic.wad"
def create_environment():
    game = DoomGame()
    game.load_config(DOOM_CONFIG)
    game.set_doom_scenario_path(DOOM_SCENARIO)
    game.init()

    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions

game, possible_actions = create_environment()

### Preprocessing Environment#############################################
import numpy as np

def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10,30:-30]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [84,84])

    return preprocessed_frame


from collections import deque# Ordered collection with ends

stack_size = 4 # We stack 4 frames
stacked_frames  =  deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        #stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        #stacked_state = np.stack(stacked_frames, axis=2)

    stacked_state = np.array(list(stacked_frames))
    return stacked_state, stacked_frames

def get_next_state(is_new_episode):
    global stacked_frames
    next_state = game.get_state().screen_buffer
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, is_new_episode)
    return next_state, stacked_frames

import agent

### Memory #####################################################################
from skimage import transform# Help us to preprocess the frames
import random

### MEMORY HYPERPARAMETERS
batch_size = 64
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time


### Prediction ###############################################################

### Training ####################################################################
### TRAINING HYPERPARAMETERS
total_episodes = 5000        # Total episodes for training
max_steps = 100              # Max possible steps in an episode

import datetime
from pathlib import Path
save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

#doomguy = agent.Agent(state_dim=(4, 84, 84), action_dim=3, save_dir=save_dir)
doomguy = agent.DQNAgent(state_dim=(4, 84, 84), action_dim=3, save_dir=save_dir)
#doomguy.load('/tmp/doomguy_net.chkpt')


from metrics import MetricLogger
logger = MetricLogger(save_dir)

game.init()
for episode in range(total_episodes):
    step = 0
    # Initialize the rewards of the episode
    episode_rewards = []

    # Make a new episode and observe the first state
    game.new_episode()
    state, stacked_frames = get_next_state(True)
    while True:
        step += 1

        # Predict the action to take and take it
        action = doomguy.act(state) # action index
        reward = game.make_action(possible_actions[action])
        #action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

        # Look if the episode is finished
        done = game.is_episode_finished()
        episode_rewards.append(reward)
        q, loss = doomguy.learn()

        logger.log_step(reward, loss, q)

        # If the game is finished
        if done:
            # the episode ends so no next state
            next_state = np.zeros((84,84), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            # Get the total reward of the episode
            total_reward = np.sum(episode_rewards)
            break
        else:
            next_state, stacked_frames = get_next_state(False)
            #memory.add((state, action, reward, next_state, done))
            doomguy.cache(state, next_state, action, reward, done)
            state = next_state

        if step > max_steps:
            total_reward = np.sum(episode_rewards)
            break

    logger.log_episode()
    if episode % 20 == 0:
        logger.record(episode=episode, epsilon=doomguy.exploration_rate, step=doomguy.curr_step)
