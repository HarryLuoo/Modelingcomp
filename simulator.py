import numpy as np
import random
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision 
import torchvision.transforms as transforms

PRESSURE = 1.1
RAIN = 0.9

class StairCaseSimulator():
    # k = number of stairs, w = stair width in meters, l =stair length in meters, n = number of walkers daily,
    # r = centimeters of rain yearly, h = stair height in meters, sWidth = mean walker shoulder width, gDelt = gate standard deviatior,
    # shoe = tuple for dimensions of the shoe, pRInit = probability distribution for intial right step
    # PLIit = probability distribution for initial left step
    def __init__(self, k, w, l, n, h, sWidth, gDelt, shoe, pRInit, pLInit, d = 0.01, r = 0):
        self.k = k
        self.w = w
        self.l = l
        self.n = n
        self.h = h
        self.r = r
        self.d = d
        self.sWidth = sWidth
        self.gDelt = gDelt
        self.shoe = shoe
        self.pRInit = pRInit
        self.pLInit = pLInit
        self.M = int(w/d)
        self.N = int(l/d)
        self.stairs = np.zeros((k, self.M, self.N))
        self.stairsFlow = np.zeros((self.k, self.M, self.N))
        self.flowDirection = np.zeros((self.k, self.M, self.N))


    def simulateWalker(self):
        LEFT = 1
        RIGHT = 0
        sigma_x = 0
        sigma_y =0
        parity = random.randint(0, 1)
        place = np.zeros(2)
        while True:
            if parity is LEFT:
                place[0] = random.gauss(self.pLInit[0], self.pLInit[1])
                place[1] = random.gauss(self.pLInit[2], self.pLInit[3])
            
            else:
                place[0] = random.gauss(self.pRInit[0], self.pRInit[1])
                place[1] = random.gauss(self.pRInit[2], self.pRInit[3])
            if 0 <= place[0] <= self.w and 0 <= place[1] <= self.l:
                break
        for i in range(self.k):
            while True:
                if parity is LEFT:
                    place[0] = random.gauss(place[0] + self.sWidth, sigma_x)
                    place[1] = random.gauss(place[1], sigma_y)
            
                else:
                    place[0] = random.gauss(place[0] - self.sWidth, sigma_x)
                    place[1] = random.gauss(place[1], sigma_y)
                if 0 <= place[0] <= self.w and 0 <= place[1] <= self.l:
                    break

            xLow = int((place[0] + self.shoe[0])/self.d)
            xHigh = int((place[0] - self.shoe[0])/self.d)
            yLow = int((place[1] + self.shoe[1])/self.d)
            yHigh = int((place[1] - self.shoe[1])/self.d)
            for j in range(xLow, xHigh):
                for k in range(yLow, yHigh):
                    self.stairs[i, j, k] += PRESSURE #we assume this contains how many centimeters of erosion a footstep will give
            parity != parity
    def flowTraverse(self, n, x, y, minimum):
        neighbors = []
        norm = (
            max(self.stairs[n, x, y-1] - minimum, 0)
            + max(self.stairs[n, x, y+1] - minimum, 0)
            + max(self.stairs[n, x-1, y] - minimum, 0)
            + max(self.stairs[n, x+1, y] - minimum, 0)
        )
    
        # Early exit if already processed
        if norm == 0 or self.stairsFlow[n, x, y] != 0:
            return None

        if y > 0:
            neighbors.append((self.stairs[n, x, y-1], max(self.stairs[n, x, y-1] - minimum, 0) / norm, x, y-1))
        if x > 0:
            neighbors.append((self.stairs[n, x-1, y], max(self.stairs[n, x-1, y] - minimum, 0) / norm, x-1, y))
        if y < self.N - 1:
            neighbors.append((self.stairs[n, x, y+1], max(self.stairs[n, x, y+1] - minimum, 0) / norm, x, y+1))
        if x < self.M - 1:
            neighbors.append((self.stairs[n, x+1, y], max(self.stairs[n, x+1, y] - minimum, 0) / norm, x+1, y))
        else: #water falls down to next stair
            self.flowDirection[n][x][y] = 2
            return None
        rand = random.random()
        sum = 0
        for i, elem in enumerate(neighbors):
            sum += elem[1]
            if rand < sum:
                self.flowDirection[n][x][y] = i
                return elem

    def simulateRain(self, threshold):
        minimum = np.min(self.stairs)
        indices = np.where(minimum <= self.stairs <= minimum + threshold)
        self.stairsFlow += 1
        for i in range(self.k):
            if i != 0: # water falls from previous level.
                for j in range(self.M):
                    height = int(self.h - self.stairs[i-1][j][self.N-1])
                    self.stairsFlow[i][j][height] += self.stairsFlow[i-1][j][self.N-1]
            currIndic = np.where(indices[0] == self.k)
            for idx in zip(*currIndic):
                curr = self.flowTraverse(idx[0], idx[1], idx[2], minimum + threshold)
                self.stairsFlow[i][curr[2]][curr[3]] += self.stairsFlow[i][idx[1]][idx[2]]
                while True:
                    temp = self.flowTraverse(i, curr[2], curr[3], minimum + threshold)
                    if temp is None:
                        break
                    self.stairsFlow[i][temp[2]][temp[3]] += self.stairsFlow[i][curr[3]][curr[3]]


    def calculateRainErosion(self):
        # this method assumes the stairsFlow matrix is populated with a value representing the 
        # number of centimeters rain flowing. We need to determine how many centimers of erosion this would cause. 
        # once determined this value should be added to that in the stairs matrix.
        return None
            

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
      
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3,3), stride=1, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=(2, 2,2), stride=(2, 2,2))  # Downsample (2x2 pooling)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 8 * 2, 128)  
        self.fc2 = nn.Linear(128, 4) 

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))  # (16, 16, 32)
        x = self.pool(nn.ReLU()(self.conv2(x)))  # (32, 8, 16)
        
       
        x = self.flatten(x)  
        x = nn.ReLU()(self.fc1(x))  
        x = self.fc2(x)  
        return x

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc

