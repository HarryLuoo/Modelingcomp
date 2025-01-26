import matplotlib.pyplot as plt
import numpy as np
from stairSim import StairCaseSimulator
from tqdm import tqdm

def plot_stairs(stairs, k, w, l, title):
    """
    Helper function to plot the stairs heatmap and contours
    
    Parameters:
        stairs (np.ndarray): Stair deformation data
        k (int): Number of stairs
        w (float): Width of stairs
        l (float): Length of stairs
        title (str): Plot title
    
    Returns:
        tuple: (figure, axis) matplotlib objects
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        combined_view = np.zeros((stairs.shape[1], stairs.shape[2]))
        for i in range(k):
            combined_view += stairs[i]
            
        im = ax.imshow(combined_view.T, cmap='viridis', 
                      extent=[0, w, 0, l], origin='lower',
                      aspect='auto')
        contour = ax.contour(combined_view.T, cmap='gray', 
                           extent=[0, w, 0, l], origin='lower')
        ax.clabel(contour, inline=True, fontsize=8)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Deformation intensity (m)', rotation=270, labelpad=15)
        
        ax.set_title(title)
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Length (m)')
        
        return fig, ax
    
    except Exception as e:
        print(f"Error in plot_stairs: {e}")
        return None, None

def validate_parameters(k, w, l, timeMax, timeStep, walkFreq):
    """Validate simulation parameters"""
    assert k > 0, "Number of stairs must be positive"
    assert w > 0, "Width must be positive"
    assert l > 0, "Length must be positive"
    assert timeMax > 0, "Simulation time must be positive"
    assert timeStep > 0, "Time step must be positive"
    assert walkFreq >= 0, "Walker frequency cannot be negative"

def main():
    # Staircase parameters
    k = 5  # Number of stairs
    w = 2  # Width in meters
    l = 1  # Length in meters
    n = 100  
    h = 0.15  # Stair height in meters
    sWidth = 0.5  # Shoulder width
    gDelt = 0.1  # Gait deviation
    shoe = (0.3, 0.1)  # Shoe dimensions
    muS = np.array([[0.3, 0.5], [0.7, 0.5]])
    muN = np.array([[0.4, 0.5], [0.6, 0.5]])
    sigS = np.array([[0.01, 0], [0, 0.01]])
    sigN = np.array([[0.005, 0], [0, 0.005]])
    p_u = 0.2
    beta = 0.0001  # Erosion coefficient

    # Simulation parameters
    timeMax = 30  # Total simulation time in days
    timeStep = 1/12  # 2-hour time steps
    rainFreq = 0
    walkFreq = 500  # Walkers per day
    threshold = 0

    try:
        validate_parameters(k, w, l, timeMax, timeStep, walkFreq)

        print(f"Initializing simulation with {timeMax} days, {walkFreq} walkers per day")
        
        simulator = StairCaseSimulator(k, w, l, n, h, sWidth, gDelt, shoe, 
                                     muS, muN, sigS, sigN, p_u, beta=beta)

        print("Running simulation...")
        stairs_history, t = simulator.run_simulation(timeMax, timeStep, rainFreq, walkFreq, threshold)
        print("Simulation complete!")

        # Plot initial state
        plot_stairs(stairs_history[0], k, w, l, 'Initial State of Stairs')
        
        # Plot final state
        plot_stairs(stairs_history[-1], k, w, l, f'Stair Deformation After {timeMax} Days')
        
        # Plot erosion over time
        fig, ax = plt.subplots(figsize=(10, 5))
        total_erosion = [np.sum(stairs) for stairs in stairs_history]
        time_points = np.linspace(0, timeMax, len(total_erosion))
        ax.plot(time_points, total_erosion)
        ax.set_title('Total Erosion Over Time')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Total Deformation (m)')
        ax.grid(True)
        
        plt.show()

    except Exception as e:
        print(f"Error in simulation: {e}")
        raise

if __name__ == "__main__":
    main()
