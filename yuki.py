import numpy as np
import math
import scipy.special
import matplotlib.pyplot as plt
import time
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import tifffile as tiff

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import os

import scipy.io
from ensemble_utils import AnimatedScatter, ensemble_detect

#Load neuron locations/ROIs
file_name = "/home/cat/data/yuki/fov6_flash-009/fov6_flash-009_Cycle00001_VoltageRecording_001.csv"    
#locs = scipy.io.loadmat(file_name, mdict=None, appendmat=True)
#locs = [locs['Coord_active'][:,0], locs['Coord_active'][:,1]]


#********* LOAD LIGHT ARRAY *************
if os.path.exists(file_name[:-4]+"_ontimes.txt"):
    on_times = np.loadtxt(file_name[:-4]+"_ontimes.txt", dtype=np.float32)
else:
    stim_array = np.loadtxt(file_name, delimiter=',')
    print stim_array
    light = stim_array[:,1]
    on_times=[]
    for k in range(len(light)-1):
        if light[k+1]>(light[k]+3):
            on_times.append(k)
    print "on_times: ", on_times

    np.savetxt(file_name[:-4]+"_ontimes.txt", on_times)

on_times = np.float32(on_times)*1E-3
print on_times


file_name = '/home/cat/data/yuki/fov6_flash-009/fov6_flash-009.xml'
if os.path.exists(file_name[:-4]+"_frametime.txt"): 
    dt = np.loadtxt(file_name[:-4]+"_frametime.txt", dtype=np.float32)
else:
    import xmltodict
    with open(file_name) as fd:
        doc = xmltodict.parse(fd.read())
    dt = doc["PVScan"]['PVStateShard']['PVStateValue'][3]['@value']
    f = open(file_name[:-4]+"_frametime.txt", 'w')
    f.write(dt)  # python will convert \n to os.linesep
    f.close() 
print dt

#********* LOAD IMAGING ARRAY **************

#file_name = '/home/cat/data/yuki/fov6_flash-009/fov6_flash-009_ArcLight2_kalman.npy'
#file_name = '/home/cat/data/yuki/fov6_flash-009/fov6_flash-009_ArcLight2_turboreg_kalman.tif'
file_name = '/home/cat/data/alejandro/G2M5/20170511/000/G2M5_C1V1_GCaMP6s_20170511_000.tif'
images_kalman = tiff.imread(file_name)
np.save(file_name[:-4], images_kalman)
quit()
images_kalman=np.load(file_name[:-4]+".npy", mmap_mode='c')
#np.save(file_name[:-4]+"_float16.npy", np.float16(images_kalman))
#print images_kalman.shape
#plt.plot(np.mean(images_kalman[:,100:400,256], axis=1))
#plt.show()
#quit()

len_rec = 201999.

#frate = len_rec/len(images_kalman)
frate = 1./dt
print frate

img_stack = []
for k in range(len(on_times)):
    #temp_stack = images_kalman[int(on_times[k]/frate):int(on_times[k]/frate)+200]
    temp_stack = images_kalman[int(on_times[k]*frate):int(on_times[k]*frate)+200]
    print temp_stack.shape
    img_stack.append(temp_stack)

#img_stack = np.array(img_stack)
#print img_stack.shape



print "TODO: filter/remove bleaching..."
print "TODO: detect light from image signal peaks... triggers don't look right..."

if False:
    
    img_average = np.mean(img_stack, axis=0)
    print img_average.shape
    np.save(file_name[:-4]+"_mean", img_average)

    baseline = np.mean(img_average, axis=0)
    print baseline.shape
    plt.imshow(baseline)
    plt.show()
    
    img_dff = np.divide(np.subtract(img_average,baseline),baseline)
    
    #img_dff = np.
    
    f = open(file_name[:-4]+"_mean_binary.bin", 'wb')
    print type(img_average[k][0][0])
    for k in range(len(img_average)):
        img_average[k].tofile(f)    

    f = open(file_name[:-4]+"_dff_binary.bin", 'wb')
    for k in range(len(img_dff)):
        img_dff[k].tofile(f)    

else:
    img_average =np.load(file_name[:-4]+"_mean.npy")

if False:
    img_max = np.max(img_stack,axis=0)
    print img_max.shape
    np.save(file_name[:-4]+"_max", img_max)
else:
    img_max =np.load(file_name[:-4]+"_max.npy")

#for k in range(0,len(images_kalman),10):
#    print k
#    plt.imshow(images_kalman[k], cmap='Greys')
#    plt.show()


if True:
    print "saving masked flash screen files"
    for on_time in on_times:
        start_frame = int(on_time*frate)

        for k in range(0,10,1):
            #plt.imshow(images_kalman[start_frame+k])
            images_kalman[start_frame+k]=images_kalman[start_frame-1]
            plt.title(str(k))
            #plt.show()
    np.save(file_name[:-4]+"_masked", np.float16(images_kalman))
    quit()

#************* PLOT AVERAGE DATA MOVIES ************
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
f0 = 0
#s = a0*np.sin(2*np.pi*f0*t)
#l, = plt.plot(t, s, lw=2, color='red')
#img_data = img_average
img_data = images_kalman

ax=plt.subplot(1,2,1)
img = plt.imshow(img_data[0], cmap='Greys')
#plt.show()
#plt.axis([0, 1, -10, 10])
cmaps = []
axcolor = 'lightgoldenrodyellow'
axframe = plt.axes([0.25, 0.1, 0.65, 0.03])#, facecolor=axcolor)
#axamp = plt.axes([0.25, 0.15, 0.65, 0.03])#, facecolor=axcolor)

frame = Slider(axframe, 'frame', 0, len(img_data), valinit=f0)
#samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


def update(val):
    #k = frame.val
    #freq = sfreq.val
    #l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    #img.set_ydata(img_data[val])
    img.set_data(img_data[int(frame.val)])    
    img.set_cmap(radio.value_selected)
    fig.canvas.draw_idle()
    
    #print button.value_selected
    
frame.on_changed(update)
#samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    frame.reset()
    #samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15])#, facecolor=axcolor)
radio = RadioButtons(rax, ('Greys', 'viridis', 'plasma'), active=0)


def colorfunc(label):
    img.set_color(label)
    fig.canvas.draw_idle()
    
radio.on_clicked(colorfunc)

plt.show()



quit()



















quit()

import csv
with open(file_name, 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
     for row in spamreader:
         stim_array.append(np.float32(row))

print stim_array


quit()

#Load spikes
file_name = "/media/cat/500GB/in_vivo/alejandro/test_data/Spikes.mat"
rasters = scipy.io.loadmat(file_name, mdict=None, appendmat=True)
spikes = rasters['Spikes'].T
#print spikes.shape
spikes = spikes * 100 + 1
#print spikes
#quit()
























#test_array = np.random.random((3,3))
#print test_array
#print np.roll(test_array,1,axis=1)
#quit()

#Make movie stack array 
#raster_stack = np.zeros((len(x), len(spikes[0])), dtype=np.int8)


#Load high-actitivity frames
file_name = '/media/cat/500GB/in_vivo/alejandro/test_data/Pks_Frame.mat'
data = scipy.io.loadmat(file_name, mdict=None, appendmat=True)
high_activity_frames = data['Pks_Frame'][0]

#Load orientation
file_name = '/media/cat/500GB/in_vivo/alejandro/test_data/vectorOrientationT.mat'
stimuli = scipy.io.loadmat(file_name, mdict=None, appendmat=True)

orientations = stimuli['vectorOrientationT']

#Make stimulus arrays
horizontal = np.zeros((15,15), dtype=np.int8)
horizontal[:,::2]=1.
vertical = np.zeros((15,15), dtype=np.int8)
vertical[::2,:]=1.
grey = np.zeros((15,15), dtype=np.int8)+0.5

stims = [grey, horizontal, vertical]

#orients = []
#for k in range(len(orientations)):
	#orients.append(stims[orientations[k][0]])

#Compute binarized vectors
file_name = "/media/cat/500GB/in_vivo/alejandro/test_data/Spikes.mat"
rasters = scipy.io.loadmat(file_name, mdict=None, appendmat=True)
spike_array = rasters['Spikes'].T

print spike_array.shape
print spike_array

from dim_reduction.dim_reduction import dim_reduction_general 

X = spike_array
method = 3
pca_data = dim_reduction_general(X, method, file_name)

print pca_data.shape
print pca_data[0:40]

#Normalize PCA data so it goes from 0..1000 in all directions; easier to plot heat maps below
pca_data[:,0] = (pca_data[:,0]-np.min(pca_data[:,0]))/(np.max(pca_data[:,0])-np.min(pca_data[:,0]))
pca_data[:,1] = (pca_data[:,1]-np.min(pca_data[:,1]))/(np.max(pca_data[:,1])-np.min(pca_data[:,1]))
#pca_data[:,2] = (pca_data[:,2]-np.min(pca_data[:,2]))/(np.max(pca_data[:,2])-np.min(pca_data[:,2]))

pca_data = pca_data*100.


#***************** SHOW ENSEMBLE HEAT MAP ***************
#ensemble_list = ensemble_detect(pca_data)


#******************* PLOT DETECTED ENSEMBLE RASTERS ******************
if False: 
    offset=0
    colors=['blue', 'red', 'green', 'brown', 'pink', 'magenta', 'orange', 'black', 'cyan']
    ax=plt.subplot()

    #Plot ensemble rasters
    for i in range(len(ensemble_list)): #range(len(Sort_sua.units)):
        print "... unit: ", i

        x = np.float32(ensemble_list[i])/4.     #Convert to seconds

        ymin=np.zeros(len(x))
        ymax=np.zeros(len(x))
        ymin+=offset+0.8
        ymax+=offset

        plt.vlines(x, ymin, ymax, linewidth=1, color=colors[i%9], alpha=1) #colors[mod(counter,7)])
        
        offset=offset+1.0



    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("Time (sec)", fontsize=24)
    ax.set_ylabel("Ensembles", fontsize=24)
    plt.tick_params(axis='both', which='both', labelsize=24)
    plt.ylim(0, offset)
    plt.show()


#****************** Show ENSEMBLES ***************
if False:
    file_name = "/media/cat/500GB/in_vivo/alejandro/test_data/Spikes.mat"
    rasters = scipy.io.loadmat(file_name, mdict=None, appendmat=True)
    spikes = rasters['Spikes'].T
    print spikes

    #Loop over each ensemble and build a 3D plot
    xpos = locs[0]
    ypos = locs[1]

    fig = plt.figure()
    ctr=0
    for k in range(len(ensemble_list)):
        print "ensemble: ", k
        dz = np.zeros(len(locs[0]))
        for p in range(len(ensemble_list[k])):
            frame_index = ensemble_list[k][p] 
            dz = dz+spikes[frame_index]
        dz_max = np.max(dz)
        
        indexes = np.where(dz>(dz_max*.5))[0]
        if len(indexes)<=1: continue
        ax1 = fig.add_subplot(4,6,ctr+1)
        ax1.scatter(locs[0][indexes], locs[1][indexes], c=dz[indexes], vmin=0, cmap='Reds')

        ctr+=1

        if k!=0:
            ax1.set_xticks([])
            ax1.set_yticks([])
        plt.xlim(0,250)
        plt.ylim(0,250)
        plt.ylabel("#"+str(k)+",  "+str(int(np.max(dz))) + " events", fontsize=12)

    plt.show()




if False:
    num_elements = len(xpos)
    zpos = np.zeros(len(xpos), dtype=np.float32)

    dx = np.ones(len(xpos))+1
    dy = np.ones(len(xpos))+1

    fig = plt.figure()
    ctr=0
    for k in range(len(ensemble_list)):
        dz = np.zeros(len(locs[0]))
        for p in range(len(ensemble_list[k])):
            frame_index = ensemble_list[k][p] 
            dz = dz+spikes[frame_index]

        ax1 = fig.add_subplot(4,6,ctr+1, projection='3d')
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
        ctr+=1


        if k!=0:
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])

    plt.show()


#***************** RUN ANIMATION ******************
n_frames = 3000
a = AnimatedScatter(locs, spikes, orientations, stims, pca_data, n_frames)
a.show()

print "exited clean..."


#***************** SKIP ANIMATION - RUN CODE ONLY ********************

x0, y0 = pca_data.T[1][i], pca_data.T[0][i]
sigma = 5.
 
x, y = np.arange(100), np.arange(100)

gx = np.exp(-(x-x0)**2/(2*sigma**2))
gy = np.exp(-(y-y0)**2/(2*sigma**2))
g = np.outer(gx, gy)
self.vertical_matrix = self.vertical_matrix + g / np.sum(g)  # normalize, if you want that

self.vertical = self.ax5.imshow(self.vertical_matrix, cmap='viridis')


   
print a.vertical 

quit()























#***********GENERATE ANIMATIONS
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=15000)

fig = plt.figure()
im = []

#gs = gridspec.GridSpec(2,len(self.ca_stack)*2)
gs = gridspec.GridSpec(1,2)

#[Ca] stacks
title = "Neuron activity vs. Time"
#for k in range(len(self.ca_stack)):    
	#ax = plt.subplot(gs[0:2,k*2:k*2+2])
	#plt.title(titles[k], fontsize = 12)

	#v_max = np.nanmax(np.ma.abs(self.ca_stack[k])); v_min = -v_max
	#ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0
	#im.append(plt.imshow(self.ca_stack[k][0], vmin=v_min, vmax = v_max, cmap=plt.get_cmap('jet'), interpolation='none'))

#PCA stack
ax = plt.subplot(gs[0,0:1])
plt.title("Neuron Rasters", fontsize = 12)
ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0

print spikes[0]

print len(locs[0]), len(spikes[0])
colors = []
for k in range(101):
	colors.append((0,1,0))
colors=np.array(colors)
#plt.scatter(locs[0], locs[1], s=100, color=(colors.T*spikes[0]).T)
#plt.show()
all_colors = []
for k in range(3000):
	all_colors.append((colors.T*spikes[0]).T)

im.append(plt.scatter(locs[0], locs[1], s=100, color=all_colors[0]))

#Loop to combine all video insets into 1
print "...making final video..."
def updatefig(j):
	print "...frame: ", j
	#plt.suptitle(self.selected_dff_filter+'  ' +self.dff_method + "\nFrame: "+str(j)+"  " +str(format(float(j)/self.img_rate-self.parent.n_sec,'.2f'))+"sec  ", fontsize = 15)
	#plt.suptitle("Time: " +str(format(float(j)/self.img_rate-self.parent.n_sec,'.2f'))+"sec  Frame: "+str(j), fontsize = 15)

	# set the data in the axesimage object
	ctr=0
	#for k in range(len(self.ca_stack)): 
	#	im[ctr].set_array(self.ca_stack[k][j]); ctr+=1
	
	im[0].set_array(plt.scatter(locs[0], locs[1], s=100, color=all_colors[j]))
	#im[ctr].set_array(self.movie_stack[j]); ctr+=1
	#im[ctr].set_array(self.lever_stack[j]); ctr+=1
	#im[ctr].set_array(self.annotation_stacks[j]); ctr+=1

	# return the artists set
	return im
	
# kick off the animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(len(spikes)), interval=100, blit=False, repeat=True)
#ani = animation.FuncAnimation(fig, updatefig, frames=range(len(self.ca_stack[1])), interval=100, blit=False, repeat=True)

if False:
	#ani.save(self.parent.root_dir+self.parent.animal.name+"/movie_files/"+self.selected_session+'_'+str(len(self.movie_stack))+'_'+str(self.selected_trial)+'trial.mp4', writer=writer, dpi=300)
	ani.save(file_name[:-4]+'.mp4', writer=writer, dpi=600)
plt.show()
