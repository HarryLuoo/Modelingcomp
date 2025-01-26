import numpy as np
import random
from tqdm import tqdm

PRESSURE = 1.1
RAIN = 0.9

def poisson_binary(rate, time_steps, dt):
    """
    Simulates a Poisson process returning 1 (event) or 0 (no event) at each time step.
    
    Parameters:
        rate (float): Average rate (events per unit time)
        time_steps (int): Number of time steps to simulate
        dt (float): Time step size
    
    Returns:
        numpy.ndarray: Array of 1s and 0s representing events over time
    """
    p_event = rate * dt
    if p_event > 1:
        p_event = 1.0
    
    return np.random.choice([1, 0], size=time_steps, p=[p_event, 1 - p_event])

class StairCaseSimulator:
    def __init__(self, k, w, l, n, h, sWidth, gDelt, shoe, muS, muN, sigS, sigN, p_u, d=0.01, r=0, beta=0.01):
        """
        Initialize the staircase simulator.
        
        Parameters:
            k (int): Number of stairs
            w (float): Width of stairs
            l (float): Length of stairs
            n (int): Number of discretization points
            h (float): Height of stairs
            sWidth (float): Step width
            gDelt (float): Gait deviation
            shoe (tuple): Shoe dimensions (length, width)
            muS, muN (np.array): Mean positions for different walker types
            sigS, sigN (np.array): Variance for different walker types
            p_u (float): Probability of walker type
            d (float): Discretization size
            r (float): Rain intensity
            beta (float): Erosion coefficient
        """
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
        self.M = int(w / d)
        self.N = int(l / d)
        self.stairs = np.zeros((k, self.M, self.N))
        self.p_u = p_u
        self.muS = muS
        self.muN = muN
        self.sigS = sigS
        self.sigN = sigN
        self.beta = beta

    def simulate_walker(self):
        """Optimized walker simulation"""
        LEFT, RIGHT = 1, 0
        parity = random.randint(0, 1)
        
        place = np.zeros(2)
        
        if random.random() < self.p_u:
            mean = self.muN
            sig = self.sigN
            stepWidth = self.sWidth / 2
            stepSig = self.gDelt / 2
        else:
            mean = self.muS
            sig = self.sigS
            stepWidth = self.sWidth
            stepSig = self.gDelt

        while True:
            if parity == LEFT:
                place = np.random.normal([mean[0][0], mean[0][1]], 
                                       [np.sqrt(sig[0][0]), np.sqrt(sig[0][1])])
            else:
                place = np.random.normal([mean[1][0], mean[1][1]], 
                                       [np.sqrt(sig[1][0]), np.sqrt(sig[1][1])])
            if 0 <= place[0] <= self.w and 0 <= place[1] <= self.l:
                break

        xShoe = self.shoe[0]
        yShoe = self.shoe[1]

        for i in range(self.k):
            while True:
                if parity == LEFT:
                    place[0] = np.random.normal(place[0] + stepWidth, stepSig)
                    place[1] = np.random.normal(place[1], stepSig)
                else:
                    place[0] = np.random.normal(place[0] - stepWidth, stepSig)
                    place[1] = np.random.normal(place[1], stepSig)
                
                if 0 <= place[0] <= self.w and 0 <= place[1] <= self.l:
                    break

            xLow = max(0, int((place[0] - xShoe) / self.d))
            xHigh = min(self.M, int((place[0] + xShoe) / self.d))
            yLow = max(0, int((place[1] - yShoe) / self.d))
            yHigh = min(self.N, int((place[1] + yShoe) / self.d))

            self.stairs[i, xLow:xHigh, yLow:yHigh] += PRESSURE

            parity = 1 - parity

    def apply_erosion_filter(self):
        """Applies erosion to the stairs"""
        self.stairs = self.stairs + self.beta * (self.stairs ** 2)

    def run_simulation(self, timeMax, timeStep, rainFreq, walkFreq, threshold):
        """Run the simulation"""
        n = int(timeMax / timeStep)
        
        walker_rate = min(walkFreq * timeStep, 1.0)
        walkerEvents = poisson_binary(walker_rate, n, timeStep)
        
        store_interval = max(1, n // 100)
        num_stored_steps = n // store_interval + 1
        stairs = np.zeros((num_stored_steps, self.k, self.M, self.N))
        
        pbar = tqdm(range(n), desc='Simulating days', 
                   unit='steps', unit_scale=timeStep)
        
        j = 0
        for i in pbar:
            if walkerEvents[i]:
                self.simulate_walker()
            
            if i % 100 == 0:
                self.apply_erosion_filter()
            
            if i % store_interval == 0:
                stairs[j] = self.stairs.copy()
                j += 1
                
            if i % 1000 == 0:
                pbar.set_description(f'Day {i*timeStep:.1f}/{timeMax}')

        return stairs, np.linspace(0, timeMax, num_stored_steps)
