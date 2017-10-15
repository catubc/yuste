import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import issparse, spdiags, coo_matrix, csc_matrix
from past.utils import old_div
from scipy.ndimage.measurements import center_of_mass

try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.models import CustomJS, ColumnDataSource, Range1d
except:
    print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")


def com(A, d1, d2):
    """Calculation of the center of mass for spatial components

     Inputs:
     ------
     A:   np.ndarray
          matrix of spatial components (d x K)

     d1:  int
          number of pixels in x-direction

     d2:  int
          number of pixels in y-direction

     Output:
     -------
     cm:  np.ndarray
          center of mass for spatial components (K x 2)
    """
    from past.utils import old_div

    nr = np.shape(A)[-1]
    Coor = dict()
    Coor['x'] = np.kron(np.ones((d2, 1)), np.expand_dims(list(range(d1)), axis=1))
    Coor['y'] = np.kron(np.expand_dims(list(range(d2)), axis=1), np.ones((d1, 1)))
    cm = np.zeros((nr, 2))        # vector for center of mass
    
    cm[:, 0] = old_div(np.dot(Coor['x'].T, A), A.sum(axis=0))
    cm[:, 1] = old_div(np.dot(Coor['y'].T, A), A.sum(axis=0))

    return cm
    
def plot_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Parameters:
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)

     Cn:  np.ndarray (2D)
               Background image (e.g. mean, correlation)

     thr_method: [optional] string
              Method of thresholding: 
                  'max' sets to zero pixels that have value less than a fraction of the max value
                  'nrg' keeps the pixels that contribute up to a specified fraction of the energy

     maxthr: [optional] scalar
                Threshold of max value

     nrgthr: [optional] scalar
                Threshold of energy

     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)
               Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr

     display_number:     Boolean
               Display number of ROIs if checked (default True)

     max_number:    int
               Display the number for only the first max_number components (default None, display all numbers)

     cmap:     string
               User specifies the colormap (default None, default colormap)

     Returns:
     --------
     Coor: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    print "# neurons: ", nr
    if max_number is None:
        max_number = nr

    #if thr is not None:
    #    thr_method = 'nrg'
    #    nrgthr = thr
    #    warn("The way to call utilities.plot_contours has changed. Look at the definition for more details.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    
        
    ax = plt.gca()
    if vmax is None and vmin is None:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1), vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=vmin, vmax=vmax)
    
    coordinates = []
    cm = com(A, d1, d2)
    for i in range(np.minimum(nr, max_number)):
        print i,
    
        pars = dict(kwargs)
        if thr_method == 'nrg':
            indx = np.argsort(A[:, i], axis=None)[::-1]
            cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            thr = nrgthr

        else:  # thr_method = 'max'
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = A[:, i].flatten()
            Bvec /= np.max(Bvec)
            thr = maxthr

        if swap_dim:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='C')
        else:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='F')
        cs = plt.contour(y, x, Bmat, [thr], colors=colors)
        
        # this fix is necessary for having disjoint figures and borders plotted correctly
        p = cs.collections[0].get_paths()
        v = np.atleast_2d([np.nan, np.nan])
        for pths in p:
            vtx = pths.vertices
            num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
            if num_close_coords < 2:
                if num_close_coords == 0:
                    # case angle
                    newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                    #import ipdb; ipdb.set_trace()
                    vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)

                else:
                    # case one is border
                    vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                    #import ipdb; ipdb.set_trace()

            v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

        pars['CoM'] = np.squeeze(cm[i, :])
        pars['coordinates'] = v
        pars['bbox'] = [np.floor(np.min(v[:, 1])), np.ceil(np.max(v[:, 1])),
                        np.floor(np.min(v[:, 0])), np.ceil(np.max(v[:, 0]))]
        pars['neuron_id'] = i + 1
        coordinates.append(pars)


    if display_numbers:
        for i in range(np.minimum(nr, max_number)):
            if swap_dim:
                ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors)
            else:
                ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors)

    return coordinates
    

def check_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, **kwargs):
    """Plots contour of spatial components against a background image and returns their coordinates

     Parameters:
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)

     Cn:  np.ndarray (2D)
               Background image (e.g. mean, correlation)

     thr_method: [optional] string
              Method of thresholding: 
                  'max' sets to zero pixels that have value less than a fraction of the max value
                  'nrg' keeps the pixels that contribute up to a specified fraction of the energy

     maxthr: [optional] scalar
                Threshold of max value

     nrgthr: [optional] scalar
                Threshold of energy

     thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)
               Kept for backwards compatibility. If not None then thr_method = 'nrg', and nrgthr = thr

     display_number:     Boolean
               Display number of ROIs if checked (default True)

     max_number:    int
               Display the number for only the first max_number components (default None, display all numbers)

     cmap:     string
               User specifies the colormap (default None, default colormap)

     Returns:
     --------
     Coor: list of coordinates with center of mass, contour plot coordinates and bounding box for each component
    """
    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    print "# neurons: ", nr
    if max_number is None:
        max_number = nr

    #if thr is not None:
    #    thr_method = 'nrg'
    #    nrgthr = thr
    #    warn("The way to call utilities.plot_contours has changed. Look at the definition for more details.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    
    coordinates = []
    cm = com(A, d1, d2)
    
    #plt.imshow(Cn, interpolation=None, cmap=cmap,
    #          vmin=vmin, vmax=vmax)

    print Cn.shape
    images_kalman = np.load('/home/cat/data/alejandro/G2M5/20170511/000/G2M5_C1V1_GCaMP6s_20170511_000.npy')
    
    #************* PLOT AVERAGE DATA MOVIES ************
    from matplotlib.widgets import Slider, Button, RadioButtons

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    t = np.arange(0.0, 1.0, 0.001)
    f0 = 0

    img_data = images_kalman

    ax=plt.subplot(1,1,1)
    img = plt.imshow(img_data[0], cmap='viridis')
    cmaps = []
    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03])#, facecolor=axcolor)
    axframe2 = plt.axes([0.25, 0.15, 0.65, 0.03])#, facecolor=axcolor)

    frame = Slider(axframe, 'frame', 0, len(img_data), valinit=f0)
    frame2 = Slider(axframe2, 'frame', 0, nr, valinit=0)


    #********* PRELOAD CONTOUR VALS **************
    y_array=[]
    x_array=[]
    Bmat_array=[]
    
    #for i in range(np.minimum(nr, max_number)):
    for k in neuron_ids:
        i=k-1
        print i
        
        pars = dict(kwargs)
        if thr_method == 'nrg':
            indx = np.argsort(A[:, i], axis=None)[::-1]
            cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            thr = nrgthr

        else:  # thr_method = 'max'
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = A[:, i].flatten()
            Bvec /= np.max(Bvec)
            thr = maxthr

        #if swap_dim:
        #    Bmat = np.reshape(Bvec, np.shape(Cn), order='C')
        #else:
        Bmat = np.reshape(Bvec, np.shape(Cn), order='F')
        
        y_array.append(y)
        x_array.append(x)
        Bmat_array.append(Bmat)
        cs = ax.contour(y, x, Bmat, [thr], colors=colors)
        ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors)

    cell=int(frame2.val)

            
    def update(val):
        img.set_data(img_data[int(frame.val)])    
        img.set_cmap(radio.value_selected)
        fig.canvas.draw_idle()

        cell=int(frame2.val)
        cs = ax.contour(y_array[cell], x_array[cell], Bmat_array[cell], [thr], colors=colors)
            
    frame.on_changed(update)
    frame2.on_changed(update)

    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


    def reset(event):
        frame.reset()
        frame2.reset()
        #samp.reset()
    button.on_clicked(reset)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15])#, facecolor=axcolor)
    radio = RadioButtons(rax, ('viridis', 'Greys', 'plasma'), active=0)


    def colorfunc(label):
        img.set_color(label)
        fig.canvas.draw_idle()
        
    radio.on_clicked(colorfunc)

    plt.show()



def nb_view_patches(Yr, A, C, b, f, d1, d2, YrA = None, image_neurons=None, thr=0.99, denoised_color=None,cmap='viridis'):
    """
    Interactive plotting utility for ipython notebook

    Parameters:
    -----------
    Yr: np.ndarray
        movie

    A,C,b,f: np.ndarrays
        outputs of matrix factorization algorithm

    d1,d2: floats
        dimensions of movie (x and y)

    YrA:   np.ndarray
        ROI filtered residual as it is given from update_temporal_components
        If not given, then it is computed (K x T)        

    image_neurons: np.ndarray
        image to be overlaid to neurons (for instance the average)

    thr: double
        threshold regulating the extent of the displayed patches

    denoised_color: string or None
        color name (e.g. 'red') or hex color code (e.g. '#F0027F')

    cmap: string
        name of colormap (e.g. 'viridis') used to plot image_neurons
    """
    colormap = mpl.cm.get_cmap(cmap)
    #colormap = cmap
    grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    nr, T = C.shape
    #nA2 = np.ravel(A.power(2).sum(0))
    nA2 = np.ravel((A**2).sum(0))
    b = np.squeeze(b)
    f = np.squeeze(f)
    if YrA is None:
        Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                   (A.T * np.matrix(Yr) -
                    (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                    A.T.dot(A) * np.matrix(C)) + C)
    else:
        Y_r = C + YrA
            

    x = np.arange(T)
    z = old_div(np.squeeze(np.array(Y_r[:, :].T)), 100)
    
    
    
    if image_neurons is None:
        image_neurons = A.mean(1).reshape((d1, d2), order='F')

    #coors = get_contours(A, (d1, d2), thr, )
    #REUSING get_contours function from different place
    coors = get_contours(A, Cn=image_neurons, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None)
    cc1 = [cor['coordinates'][:, 0] for cor in coors]
    cc2 = [cor['coordinates'][:, 1] for cor in coors]
    c1 = cc1[0]
    c2 = cc2[0]

    # split sources up, such that Bokeh does not warn
    # "ColumnDataSource's columns must be of the same length"
    source = ColumnDataSource(data=dict(x=x, y=z[:, 0], y2=C[0] / 100))
    source_ = ColumnDataSource(data=dict(z=z.T, z2=C / 100))
    source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
    source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

    callback = CustomJS(args=dict(source=source, source_=source_, source2=source2, source2_=source2_), code="""
            var data = source.get('data')
            var data_ = source_.get('data')
            var f = cb_obj.get('value')-1
            x = data['x']
            y = data['y']
            y2 = data['y2']

            for (i = 0; i < x.length; i++) {
                y[i] = data_['z'][i+f*x.length]
                y2[i] = data_['z2'][i+f*x.length]
            }

            var data2_ = source2_.get('data');
            var data2 = source2.get('data');
            c1 = data2['c1'];
            c2 = data2['c2'];
            cc1 = data2_['cc1'];
            cc2 = data2_['cc2'];

            for (i = 0; i < c1.length; i++) {
                   c1[i] = cc1[f][i]
                   c2[i] = cc2[f][i]
            }
            source2.trigger('change')
            source.trigger('change')
        """)
        
    print x.shape
    print z.shape
    y_array = z.T
    t = np.arange(3000)
    for k in range(len(y_array)):
        #print x[k]
        #print z[k]
        plt.plot(t,y_array[k]+1*k)
    
        #x = np.arange(T)
        #z = old_div(np.squeeze(np.array(Y_r[:, :].T)), 100)
        
    plt.show()

    plot = bpl.figure(plot_width=600, plot_height=300)
    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
    if denoised_color is not None:
        plot.line('x', 'y2', source=source, line_width=1, line_alpha=0.6, color=denoised_color)

    slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                 title="Neuron Number", callback=callback)
    xr = Range1d(start=0, end=image_neurons.shape[1])
    yr = Range1d(start=image_neurons.shape[0], end=0)
    plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)

    plot1.image(image=[image_neurons[::-1, :]], x=0,
                y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
    plot1.patch('c1', 'c2', alpha=0.6, color='purple', line_width=2, source=source2)

    bpl.show(bokeh.layouts.layout([[slider], [bokeh.layouts.row(plot1, plot)]]))

    return Y_r


def get_contours(A, Cn, thr=None, thr_method='max', maxthr=0.2, nrgthr=0.9, display_numbers=True, max_number=None,
                  cmap=None, swap_dim=False, colors='w', vmin=None, vmax=None, **kwargs):
    """Gets contour of spatial components and returns their coordinates

     Parameters:
     -----------
     A:   np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)
     
	 dims: tuple of ints
               Spatial dimensions of movie (x, y[, z])
     
	 thr: scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)

     Returns:
     --------
     Coor: list of coordinates with center of mass and
            contour plot coordinates (per layer) for each component
            
        
    #"""
    #A = csc_matrix(A)
    #d, nr = np.shape(A)
    ##if we are on a 3D video
    #if len(dims) == 3:
        #d1, d2, d3 = dims
        #x, y = np.mgrid[0:d2:1, 0:d3:1]
    #else:
        #d1, d2 = dims
        #x, y = np.mgrid[0:d1:1, 0:d2:1]

    #coordinates = []

    ##get the center of mass of neurons( patches )
    #print A.shape
    #print dims
    ##cm = np.asarray([center_of_mass(a.toarray().reshape(dims, order='F')) for a in A.T])
    #cm = []
    #for a in A.T:
        #temp = a.toarray(); print temp.shape
        #temp = temp.reshape((62500,1), order='F')
        #cm.append(center_of_mass(temp))
    #cm = np.asarray(cm)
    #print cm.shape
    
    ##for each patches
    #for i in range(nr):
        #pars = dict()
        ##we compute the cumulative sum of the energy of the Ath component that has been ordered from least to highest
        #patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        #indx = np.argsort(patch_data)[::-1]
        #cumEn = np.cumsum(patch_data[indx]**2)

        ##we work with normalized values
        #cumEn /= cumEn[-1]
        #Bvec = np.ones(d)

        ##we put it in a similar matrix
        #Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]][indx]] = cumEn
        #Bmat = np.reshape(Bvec, dims, order='F')
        #pars['coordinates'] = []
        ## for each dimensions we draw the contour
        #for B in (Bmat if len(dims) == 3 else [Bmat]):
            ##plotting the contour usgin matplotlib undocumented function around the thr threshold
            #nlist = mpl._cntr.Cntr(y, x, B).trace(thr)

            ##vertices will be the first half of the list
            #vertices = nlist[:len(nlist) // 2]
            ## this fix is necessary for having disjoint figures and borders plotted correctly
            #v = np.atleast_2d([np.nan, np.nan])
            #for k, vtx in enumerate(vertices):
                #num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                #if num_close_coords < 2:
                    #if num_close_coords == 0:
                        ## case angle
                        #newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                        #vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)

                    #else:
                        ## case one is border
                        #vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                #v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

            #pars['coordinates'] = v if len(dims) == 2 else (pars['coordinates'] + [v])
        #pars['CoM'] = np.squeeze(cm[i, :])
        #pars['neuron_id'] = i + 1
        #coordinates.append(pars)
    #return coordinates

    #****************************

    if issparse(A):
        A = np.array(A.todense())
    else:
        A = np.array(A)

    if swap_dim:
        Cn = Cn.T
        print('Swapping dim')

    d1, d2 = np.shape(Cn)
    d, nr = np.shape(A)
    print "# neurons: ", nr
    if max_number is None:
        max_number = nr

    #if thr is not None:
    #    thr_method = 'nrg'
    #    nrgthr = thr
    #    warn("The way to call utilities.plot_contours has changed. Look at the definition for more details.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

        
    ax = plt.gca()
    if vmax is None and vmin is None:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=np.percentile(Cn[~np.isnan(Cn)], 1), vmax=np.percentile(Cn[~np.isnan(Cn)], 99))
    else:
        plt.imshow(Cn, interpolation=None, cmap=cmap,
                  vmin=vmin, vmax=vmax)
    
    coordinates = []
    cm = com(A, d1, d2)
    for i in range(np.minimum(nr, max_number)):
        print i
    
        pars = dict(kwargs)
        if thr_method == 'nrg':
            indx = np.argsort(A[:, i], axis=None)[::-1]
            cumEn = np.cumsum(A[:, i].flatten()[indx]**2)
            cumEn /= cumEn[-1]
            Bvec = np.zeros(d)
            Bvec[indx] = cumEn
            thr = nrgthr

        else:  # thr_method = 'max'
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = A[:, i].flatten()
            Bvec /= np.max(Bvec)
            thr = maxthr

        if swap_dim:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='C')
        else:
            Bmat = np.reshape(Bvec, np.shape(Cn), order='F')
        cs = plt.contour(y, x, Bmat, [thr], colors=colors)
        
        # this fix is necessary for having disjoint figures and borders plotted correctly
        p = cs.collections[0].get_paths()
        v = np.atleast_2d([np.nan, np.nan])
        for pths in p:
            vtx = pths.vertices
            num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
            if num_close_coords < 2:
                if num_close_coords == 0:
                    # case angle
                    newpt = np.round(old_div(vtx[-1, :], [d2, d1])) * [d2, d1]
                    #import ipdb; ipdb.set_trace()
                    vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)

                else:
                    # case one is border
                    vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                    #import ipdb; ipdb.set_trace()

            v = np.concatenate((v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)
        pars['CoM'] = np.squeeze(cm[i, :])
        pars['coordinates'] = v
        pars['bbox'] = [np.floor(np.min(v[:, 1])), np.ceil(np.max(v[:, 1])),
                        np.floor(np.min(v[:, 0])), np.ceil(np.max(v[:, 0]))]
        pars['neuron_id'] = i + 1
        coordinates.append(pars)

    plt.close()

    #if display_numbers:
    #    for i in range(np.minimum(nr, max_number)):
    #        if swap_dim:
    #            ax.text(cm[i, 0], cm[i, 1], str(i + 1), color=colors)
    #        else:
    #            ax.text(cm[i, 1], cm[i, 0], str(i + 1), color=colors)

    return coordinates
    
    
