import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np

class ScatterAnimation():
    """
    Animation of scatter plots. Scatters the 2-d array contents of a 3-d
    array input. Interpolates distances between individual points,
    with pausing on each frame chunk.
    Args:
        position (3-d array-like container, int): Contains x, y positions of each point
            per frame chunk 
        sizes (3-d array-like container, int): Contains sizes of each point per frame chunk
        frame_labels (2-d list-like): Contains labels per frame chunk - corresponds to
            length of position and size
        text_labels (2-d list-like): Contains labels for each point in a single frame -
            corresponds number of points
        animated (int): Number of moving interpolated frames between scatters; default 10
        paused (int): Number of paused frame per scatter; default 3
        threshold (int): Size threshold for text_labels; default 0
        colors (bool): Colorful scatter plots
        cmap (matplotlib cmap): Specify a colormap
        shadows (bool): Leave a shadow of every point's initial position
        lines (bool): Draw a line between each point's final and initial position
        axis (list): Axis of the figure
        title (string): Title of the figure
        xlabel (string): X axis label of the figure
        ylabel (string): Y axis label of the figure
        repeat(bool): Automatically repeat the generated animation
        fontsize (int): Font sizes for entire animation; default 16
        frame_loc (2 element tuple): Location on axis of the frame labels; default (0, 0)
        figsize (2 element tuple): Size of the figure
        debug (bool): Debug mode - show frame counter; default False
    """

    def __init__(self, position, sizes=None, frame_labels=None, text_labels=None,
                 animated=10, paused=3, threshold=0, colors=False, cmap=None, shadows=False,
                 lines=False, axis=[0, 1, 0 ,1], title=None, xlabel=None, ylabel=None, repeat=True,
                 fontsize=16, frame_loc=(0, 0), figsize=None, debug=False, helpers=None):
        # --- Parameter initialization --- #

        # Position arrays
        self.position = position
        self.sizes = sizes
        
        # Label arrays
        self.frame_labels = frame_labels
        self.frame_loc = frame_loc
        self.text_labels = text_labels
        
        # Animation and pause frames
        self.animated = animated
        self.paused = paused
        self.framedelta = paused + animated
        self.numframes = self.framedelta * len(position)
        
        # Size label threshold
        self.threshold = threshold
        
        # Color handling
        self.colors = colors
        self.cmap = cmap
        
        # Accessories
        self.shadows = shadows
        self.lines = lines
        
        # Flags
        self.debug = debug
        
        # --- Animation features --- #
        self.fontsize=fontsize
        
        self.fig = plt.figure(figsize=figsize)
        self.ax = plt.gca()
        plt.axis(axis)
        plt.title(title, fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)

        self.helpers = helpers
        
        self.anim = animation.FuncAnimation(self.fig, self.update, frames=range(self.framedelta, self.numframes),
                                            init_func=self.initScat, blit=False, repeat=repeat, interval=50)

    def initScat(self):
        if self.frame_labels is not None:    
            self.frame_text = self.ax.text(self.frame_loc[0], self.frame_loc[1], self.text_labels[0], fontsize=24)

        self.ax_texts = []
        if self.text_labels is not None:
            for i, label in enumerate(self.text_labels):
                if self.sizes[0][i] > self.threshold:
                    self.ax_texts.append(self.ax.text(self.position[0].T[0][i],
                                                      self.position[0].T[1][i],
                                                      self.text_labels[i]))
                else:
                    self.ax_texts.append(None)
        
        if self.helpers is not None:
            for func, params in self.helpers:
                func(*params)

        plt.legend(fontsize=self.fontsize)

        if self.debug:
            self.idx_counter = self.ax.text(.25, .35, "0", fontsize=24)

        self.c = None
        if self.colors:
            self.c = np.arange(len(self.text_labels))

        # Initial scatter
        self.scat = plt.scatter(self.position[0].T[0], self.position[0].T[1],
                                c=self.c, cmap=self.cmap,
                                s=self.sizes[0], animated=True, alpha=.8, edgecolors='k')

        # Leave behind shadow of animation
        if self.shadows:
            self.ax.scatter(self.position[0].T[0], self.position[0].T[1], c=self.c, cmap=self.cmap, s=self.sizes[0], alpha=0.25)

        plt.close()
        return self.scat,

    def update(self, i):
        # Indices by scatter dimension
        next_idx = i // self.framedelta
        prev_idx = next_idx - 1
        
        # Progress counter
        print(f'... Processing Frame {i} ...', end='\r')
        
        # Last frame - draw lines
        if self.lines:
            if i == self.numframes - 1:
                idx = self.sizes[0] > self.threshold
                a = self.position[0][idx]
                b = self.position[-1][idx]

                for a, b in zip(np.array([a[:,0],b[:,0]]).T,np.array([a[:,1],b[:,1]]).T):
                    self.ax.plot(a, b, color='k', alpha=0.5, linestyle='-')

        if self.debug:
            self.idx_counter.set_text(i)
        
        # Pause frame
        if i % self.framedelta < self.paused:
            return self.scat,
        
        offset_delta = (self.position[next_idx] - self.position[prev_idx])
        offset = self.position[prev_idx] + offset_delta * (i % (self.framedelta) - self.paused) / self.animated

        offset_idx = (np.absolute(offset_delta).T[0] > 0.25) | (np.absolute(offset_delta).T[1] > 0.25)
        offset[offset_idx] = self.position[next_idx][offset_idx]
        
        sizes = self.sizes[next_idx]
        self.scat.set_offsets(offset)
        
        if self.frame_labels is not None:
            self.frame_text.set_text(self.frame_labels[next_idx])
        
        if self.text_labels is not None:
            for i in range(len(self.text_labels)):
                if self.ax_texts[i] is not None:
                    self.ax_texts[i].set_position(offset[i])

        if self.sizes is not None:
            self.scat.__sizes = sizes

        return self.scat,