import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.spatial import distance

class AnimatedScatter(object):

    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, locs, spikes, orientations, stims, pca_data, n_frames):
        

        self.orientations = orientations
        self.locs = locs
        self.s = np.zeros(len(locs))+100
        self.c = spikes
        self.stims = stims
        self.pca_data =pca_data
                
        # Setup the fig/axes
        self.fig = plt.figure(num=None, figsize=(28, 8), dpi=80, facecolor='w', edgecolor='k')


        self.gs = gridspec.GridSpec(2,4)

        #Setup neuron rasters plot
        self.ax1 = plt.subplot(self.gs[0:2,0:2])
        self.ax1.set_title("Neuron Rasters", fontsize=15)
        self.ax1.patch.set_facecolor('white')
        self.ax1.set_xlim(0,250)
        self.ax1.set_ylim(0,250)
        self.ax1.set_ylabel("Distance ($\mu$m)",fontsize=15)
        self.ax1.set_xlabel("Distance ($\mu$m)",fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        self.title = self.ax1.text(0.13,0.96, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.ax1.transAxes, ha="center")
                
        #Setup drift grating visualization
        self.ax2 = plt.subplot(self.gs[0:1,2:3])
        self.ax2.set_title("Visual Stimulus", fontsize=15)
        self.ax2.patch.set_facecolor('black')
        self.ax2.set_xlim(-0.5,len(self.stims[0])-0.5)
        self.ax2.set_ylim(-0.5,len(self.stims[0])-0.5)
        self.ax2.get_xaxis().set_visible(False); self.ax2.yaxis.set_ticks([])

        #Setup dim reduction plot
        self.ax3 = plt.subplot(self.gs[0:1,3:4])
        self.ax3.set_title("Ensemble State Space (tSNE)", fontsize=15)
        self.ax3.patch.set_facecolor('white')
        self.ax3.set_xlim(min(self.pca_data.T[0]), max(self.pca_data.T[0]))
        self.ax3.set_ylim(min(self.pca_data.T[1]), max(self.pca_data.T[1]))
        self.ax3.get_xaxis().set_visible(False); self.ax3.get_yaxis().set_visible(False)
        
        #Setup ENSEMBLE SPACEs
        self.ax4 = plt.subplot(self.gs[1:2,2:3])
        self.ax4.set_title("Horizontal stim heatmap", fontsize=15)
        #self.ax4.patch.set_facecolor('white')
        self.ax4.set_xlim(min(self.pca_data.T[0]), max(self.pca_data.T[0]))
        self.ax4.set_ylim(min(self.pca_data.T[1]), max(self.pca_data.T[1]))
        self.ax4.get_xaxis().set_visible(False); self.ax4.get_yaxis().set_visible(False)
        
        self.ax5 = plt.subplot(self.gs[1:2,3:4])
        self.ax5.set_title("Vertical stim heatmap", fontsize=15)
        #self.ax4.patch.set_facecolor('white')
        self.ax5.set_xlim(min(self.pca_data.T[0]), max(self.pca_data.T[0]))
        self.ax5.set_ylim(min(self.pca_data.T[1]), max(self.pca_data.T[1]))
        self.ax5.get_xaxis().set_visible(False); self.ax5.get_yaxis().set_visible(False)

        if False:
            self.ax6 = plt.subplot(self.gs[1:2,3:4])
            self.ax6.set_title("Locomotion", fontsize=12)
            #self.ax4.patch.set_facecolor('white')
            self.ax6.set_xlim(min(self.pca_data.T[0]), max(self.pca_data.T[0]))
            self.ax6.set_ylim(min(self.pca_data.T[1]), max(self.pca_data.T[1]))
            self.ax6.get_xaxis().set_visible(False); self.ax6.get_yaxis().set_visible(False)

            self.ax7 = plt.subplot(self.gs[1:2,4:5])
            self.ax7.set_title("Correct Reward", fontsize=12)
            #self.ax4.patch.set_facecolor('white')
            self.ax7.set_xlim(min(self.pca_data.T[0]), max(self.pca_data.T[0]))
            self.ax7.set_ylim(min(self.pca_data.T[1]), max(self.pca_data.T[1]))
            self.ax7.get_xaxis().set_visible(False); self.ax7.get_yaxis().set_visible(False)
        
        self.lines = []


        self.vertical_matrix = np.zeros((100,100), dtype=np.float32)
        self.vertical = self.ax4.imshow(self.vertical_matrix, cmap='viridis')

        self.horizontal_matrix = np.zeros((100,100), dtype=np.float32)
        self.horizontal = self.ax5.imshow(self.horizontal_matrix, cmap='viridis')
        
        
        # Then setup FuncAnimation.
        self.im = []
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=range(100,n_frames,1), interval=1, 
                                           blit=True, repeat=False)
        #self.anim_running = True
        
   
        if False:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=4, metadata=dict(artist='Me'), bitrate=100)
            self.ani.save('movie_out.mp4', writer=writer, dpi=50)

        
    def update(self,i):
        #print "sleeping..."
        if i==0: time.sleep(.1)
        
        """Update the scatter plot."""
        print i, self.orientations[i][0]
        #if i >300:
        #    exit()
        
        #self.title.set_text("Frame: "+str(i) + ", "+str(round(float(i)/4.01,1))+"sec.")
        self.title.set_text("Rec time: "+str(round(float(i)/4.01,1))+"sec.")

        self.scat = self.ax1.scatter(self.locs[0], self.locs[1], c=self.c[i], s=self.s, cmap='Reds')

        #self.stim = self.ax2.scatter(self.locs[0], self.locs[1], c=self.c[self.ctr], s=self.s, animated=True)
        #self.stim = self.ax2.imshow(np.roll(np.roll(stims[orientations[i][0]],int(i/3)),int(i/3),axis=1), vmin=0, vmax=1.0, cmap='Greys')
        self.stim = self.ax2.imshow(np.roll(np.roll(self.stims[self.orientations[i][0]],int(i%4/3),axis=1),int(i%4/3),axis=0), vmin=0, vmax=1.0, cmap='Greys', interpolation='sinc')

        #self.pca = self.ax3.scatter(pca_data.T[0], pca_data.T[1], c=pca_time_array[i], edgecolor='w',s=50, animated=True, cmap='Blues', alpha=0.7)

        #self.lines = self.ax3.plot([pca_data.T[0][i-1],pca_data.T[0][i]], [pca_data.T[1][i-1], pca_data.T[1][1]], color='white', alpha=0.7)
        colors = np.zeros((30,3),dtype=np.float32)
        colors[:,1]=1 #np.arange(30)/30.
        
        #self.lines = self.ax3.plot([pca_data.T[0][i], pca_data.T[0][i+1]], [pca_data.T[1][i], pca_data.T[1][i+1]], color='blue', alpha=0.7)
        #for k in range(30):
		#	self.lines = self.ax3.plot(pca_data.T[0][i+k:i+k+1], pca_data.T[1][i+k:i+k+1], color=colors[k], alpha=0.7)
        
        self.pca = self.ax3.scatter(self.pca_data.T[0], self.pca_data.T[1], s=50 , color='grey')
        self.pca = self.ax3.scatter(self.pca_data.T[0][i], self.pca_data.T[1][i], s=75 ,cmap='Blues')
        #self.ax3.clear()
        
        if self.orientations[i][0] == 1:
			x0, y0 = self.pca_data.T[1][i], self.pca_data.T[0][i]
			sigma = 5.
			x, y = np.arange(100), np.arange(100)

			gx = np.exp(-(x-x0)**2/(2*sigma**2))
			gy = np.exp(-(y-y0)**2/(2*sigma**2))
			g = np.outer(gx, gy)
			self.horizontal_matrix = self.horizontal_matrix + g / np.sum(g)  # normalize, if you want that

			self.horizontal = self.ax4.imshow(self.horizontal_matrix, cmap='viridis')
			
        if self.orientations[i][0] == 2:
			x0, y0 = self.pca_data.T[1][i], self.pca_data.T[0][i]
			sigma = 5.

			x, y = np.arange(100), np.arange(100)

			gx = np.exp(-(x-x0)**2/(2*sigma**2))
			gy = np.exp(-(y-y0)**2/(2*sigma**2))
			g = np.outer(gx, gy)
			self.vertical_matrix = self.vertical_matrix + g / np.sum(g)  # normalize, if you want that

			self.vertical = self.ax5.imshow(self.vertical_matrix, cmap='viridis')

        # We need to return the updated artist for FuncAnimation to draw.
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.stim, self.title, self.pca, self.horizontal, self.vertical #, self.lines
        

    def show(self):
        
        plt.show()
        #self.ani.stop()
        
def ensemble_detect(pca_data):
    
    sigma = 3.
    x, y = np.arange(100), np.arange(100)
    matrix = np.zeros((100,100), dtype=np.float32)

    for k in range(len(pca_data.T[0])):
        x0, y0 = pca_data.T[0][k], pca_data.T[1][k]
        gx = np.exp(-(x-x0)**2/(2*sigma**2))
        gy = np.exp(-(y-y0)**2/(2*sigma**2))
        g = np.outer(gx, gy)
        g = g / np.sum(g)  # normalize, if you want that

        matrix+= g
    
    for i in range(0, len(matrix)):
        for j in range(i+1, len(matrix)):
            matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]

    neighborhood_size = 5
    threshold = 0.05

    data = matrix

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)

    plt.imshow(data)
    #plt.savefig('/tmp/data.png', bbox_inches = 'tight')

    #plt.autoscale(False)
    plt.plot(x,y, 'ro')
    #plt.savefig('/tmp/result.png', bbox_inches = 'tight')



    #plt.imshow(np.flipud(matrix), vmin=0, vmax=1., cmap='viridis')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.show()
    
    ensemble_list = []
    for k in range(len(x)): ensemble_list.append([])
    for k in range(len(pca_data)):
        a = [pca_data[k][0], pca_data[k][1]]
        for p in range(len(x)):
            b = [x[p], y[p]]
            if distance.euclidean(a,b)<2:
                ensemble_list[p].append(k)
                break
    
    for k in range(len(ensemble_list)): print ensemble_list[k]
    
    return ensemble_list

