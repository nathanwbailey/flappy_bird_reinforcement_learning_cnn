import os
import sys
sys.path.append("./PyGame-Learning-Environment")
import pygame as pg
import cv2
import torch
import numpy as np
from ple import PLE 
from ple.games.flappybird import FlappyBird
from itertools import count

def preprocess_image(img):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img,1)
    img = img[:-52, ...]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    img = img[..., np.newaxis]
    img = torch.FloatTensor(img)
    img = img.permute(2, 0, 1)
    return img


game = FlappyBird(width=256, height=256)
p = PLE(game, display_screen=False)
p.init()
actions = p.getActionSet()
#List of possible actions is go up or do nothing
action_dict = {0: actions[1], 1: actions[0]}
for episode in range(10000):
    p.reset_game()
    state = p.getGameState()
    state = p.getScreenRGB()
    state = preprocess_image(state)
    state = torch.squeeze(state)
    state = state.numpy()
    #Inf count functiom
    for c in count():
        #Choose an action, get back the reward and the next state as a result of taking the action
        reward = p.act(action_dict[1])
        state = p.getScreenRGB()
        cv2.imwrite('ximage_orig_'+str(c)+'.png', state)
        state = preprocess_image(state)
        state = torch.squeeze(state)
        state = state.numpy()
        cv2.imwrite('ximage_'+str(c)+'.png', state)
            