### Set up vizdoom environment
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

import random
import time
def test_environment():
    game, actions = create_environment()

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print ("\treward:", reward)
            time.sleep(0.02)
        print ("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()

game, possible_actions = create_environment()
test_environment()
