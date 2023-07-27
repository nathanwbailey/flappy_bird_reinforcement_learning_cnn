import torch
import model
import CacheRecall
import numpy as np
import random
from itertools import count
import pygame as pg
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import cv2

class Agent():
    

    def __init__(self, BATCH_SIZE, MEMORY_SIZE, GAMMA, action_dim, action_dict, EPS_START, EPS_END, EPS_DECAY_VALUE, lr, TAU) -> None:
        self.BATCH_SIZE
        

    def preprocess_image(self, img):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:-52, ...]
        img = cv2.resize(img, (80, 80))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img[..., np.newaxis]
        img = torch.FloatTensor(img)
        img = img.permute(2, 0, 1)
        return img


    def plot_durations(self):
        plt.figure(1)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        #Plot the durations
        plt.plot(durations_t.numpy())
        # Take 100 episode averages of the durations and plot them too, to show a running average on the graph
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig(self.network_type+'_training.png')

