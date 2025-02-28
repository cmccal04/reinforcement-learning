import numpy as np
import matplotlib.pyplot as plt

# Define the racetrack grid
racetrack = np.array([
    ['#', '#', '#', '#', '#', '#', 'F', '#', '#', '#', '#', '#'],
    ['#', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
    ['#', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
    ['S', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
    ['S', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
    ['S', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#']
])

# Print the track
for row in racetrack:
    print("".join(row))

def visualize_racetrack(racetrack):
    """Display the racetrack using Matplotlib."""
    cmap = {'#': 'black', '.': 'white', 'S': 'green', 'F': 'red'}
    
    # Convert racetrack symbols to colors
    color_grid = [[cmap[cell] for cell in row] for row in racetrack]

    fig, ax = plt.subplots(figsize=(6, 4))
    
    for y in range(len(racetrack)):
        for x in range(len(racetrack[0])):
            ax.add_patch(plt.Rectangle((x, len(racetrack) - y - 1), 1, 1, color=color_grid[y][x]))
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, len(racetrack[0]))
    ax.set_ylim(0, len(racetrack))
    plt.show()

visualize_racetrack(racetrack)
