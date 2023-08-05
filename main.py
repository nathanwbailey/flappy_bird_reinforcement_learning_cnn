import os
import sys
sys.path.append("./PyGame-Learning-Environment")
import pygame as pg
from ple import PLE 
from ple.games.flappybird import FlappyBird
from ple import PLE
import agent
import torch

game = FlappyBird(width=256, height=256)
p = PLE(game, display_screen=False)
p.init()
actions = p.getActionSet()
#List of possible actions is go up or do nothing
action_dict = {0: actions[1], 1: actions[0]}
state = p.getScreenRGB()
len_state = torch.FloatTensor(state).size()
#print(len_state)
n_actions = len(action_dict)

agent = agent.Agent(BATCH_SIZE=32, MEMORY_SIZE=40000, GAMMA=0.99, example_image=state, output_dim=n_actions, action_dim=n_actions, action_dict=action_dict, EPS_START=1.0, EPS_END=0.00001, EPS_DECAY_VALUE=0.99999, TAU = 0.005, lr = 1e-6)

agent.train(episodes=10000000, env=p)