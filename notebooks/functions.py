import cv2
import torch
import numpy as np
from spikeA.Neuron import Simulated_place_cell, Simulated_grid_cell
from scipy.stats import wilcoxon, pearsonr
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Location of animal position files
firstRF = "../data/jp451_lstm_firstRF.npy"
lastRF = "../data/jp451_lstm_lastRF.npy"

# GLOBAL VARIABLES
GLOBALFONTSIZE = 12
plt.rcParams['figure.dpi'] = 100

def plotMap(ax,myMap,title="",titleY=0.95,titleFontSize=10,transpose=True,cmap="jet",vmin=0,alpha=1):
    """
    Plot one 2D map
    """
    if transpose:
        ax.imshow(myMap.T,origin="lower",cmap=cmap,interpolation=None, vmin=vmin,alpha=alpha)
    else:
        ax.imshow(myMap,origin="lower",cmap=cmap,interpolation=None, vmin=vmin,alpha=alpha)
    ax.set_title(title,y=titleY,fontsize=titleFontSize)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


class RigidGridCellModel(torch.nn.Module):
    """
    Model with one spacing, one orientation and one peak rate
    """
    def __init__(self,period,orientation,peak_rate,offset):
        super().__init__()

        period = torch.tensor([period],requires_grad =True,dtype=torch.float32) # model parameters
        offset = torch.tensor(offset,requires_grad =True,dtype=torch.float32) # model parameters
        peak_rate = torch.tensor([peak_rate],requires_grad =True,dtype=torch.float32) # model parameters
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        

        self.period = torch.nn.Parameter(period)
        self.offset = torch.nn.Parameter(offset)
        self.peak_rate = torch.nn.Parameter(peak_rate)
        
        # orientation
        self.ori_scalling= 0.01 # to make the gradient for orientation similar to other parameters
        ori = torch.tensor([orientation/self.ori_scalling], requires_grad=True,dtype=torch.float32) # start with 60 degree orientation
        self.ori = torch.nn.Parameter(ori)
       
        
        ## matrix to get the cos and sin component for our 2,1 projection matrix
        self.myMatCos = torch.tensor([[1],[0]],dtype=torch.float32)
        self.myMatSin = torch.tensor([[0],[1]],dtype=torch.float32)
       
        
    def length_to_angle(self,x,period):
        xr = x/period*np.pi*2
        return (torch.atan2(torch.sin(xr), torch.cos(xr)))

    def forward(self, X):
       
        # matrix to project onto the x axis and keep only the x coordinate
        self.sori = self.ori * self.ori_scalling
       
        self.Rx0 = self.myMatCos @ torch.cos(-self.sori[0].reshape(1,1))+ self.myMatSin @ -torch.sin(-self.sori[0].reshape(1,1)) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0
        self.Rx1 = self.myMatCos @ torch.cos(- (self.sori[0].reshape(1,1)+self.pi/3)) + self.myMatSin @ -torch.sin(-(self.sori[0].reshape(1,1)+self.pi/3))
        self.Rx2 = self.myMatCos @ torch.cos(- (self.sori[0].reshape(1,1)+self.pi/3*2)) + self.myMatSin @ -torch.sin(-(self.sori[0].reshape(1,1)+self.pi/3*2))
         
        
        # distance in cm along each axis
        d0 = X @ self.Rx0
        d1 = X @ self.Rx1
        d2 = X @ self.Rx2
        
        c0 = self.length_to_angle(d0, self.period[0])
        c1 = self.length_to_angle(d1, self.period[0]) 
        c2 = self.length_to_angle(d2, self.period[0])

        # deal with the offset, project on each vector 
        d0 = self.offset @ self.Rx0
        d1 = self.offset @ self.Rx1
        d2 = self.offset @ self.Rx2

        # offset as angle
        a0 = self.length_to_angle(d0, self.period[0])
        a1 = self.length_to_angle(d1, self.period[0])
        a2 = self.length_to_angle(d2, self.period[0])
      
        rateC0 = torch.cos(c0-a0)
        rateC1 = torch.cos(c1-a1)
        rateC2 = torch.cos(c2-a2)

        rate = (rateC0+rateC1+rateC2+1.5)/4.5*self.peak_rate
        return rate

    def modelParamToGridParam(self):
        """
        Return the grid cell parameters in a dictionary
        """
        myIter = iter(self.parameters())
        pred_grid_param = {}
        period = next(myIter).detach().numpy()
        pred_grid_param["period"] = np.array([period[0],period[0],period[0]])
        pred_grid_param["offset"] = next(myIter).detach().numpy()
        pred_grid_param["peak_rate"] = next(myIter).detach().numpy()
        ori = next(myIter).detach().numpy() * self.ori_scalling
        pred_grid_param["orientation"] = np.array([ori[0],ori[0]+np.pi/3,ori[0]+np.pi/3*2])
        return pred_grid_param


def training_loop_grid_parameters(n_epochs, model, optimizer, loss_fn, X,y,verbose=True):
    
    for epoch in range (n_epochs):

        #for X,y in data_loader:
        optimizer.zero_grad()
        yhat = model(X)
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()

        if epoch % 1000 ==0:
            if verbose:
                print("Epoch: {}, Loss: {}".format(epoch,loss))
                print("Parameters")
                pred_grid_param = model.modelParamToGridParam()
                print(pred_grid_param)
                for param in model.parameters():
                    print(param)
                print("Gradients")
                for param in model.parameters():
                    print(param.grad)
                print("")

        if loss < 0.0001:
            if verbose:
                print("Final loss:", loss)
            return loss
    return loss


class GridCellModel(torch.nn.Module):
    """
    Model with 3 period, 3 orientation and 1 peak rate.
    """
    def __init__(self,period,orientation,peak_rate,offset):
        super().__init__()

        period = torch.tensor(period,requires_grad =True,dtype=torch.float32) # model parameters
        offset = torch.tensor(offset,requires_grad =True,dtype=torch.float32) # model parameters
        peak_rate = torch.tensor(peak_rate,requires_grad =True,dtype=torch.float32) # model parameters
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        

        self.period = torch.nn.Parameter(period)
        self.offset = torch.nn.Parameter(offset)
        self.peak_rate = torch.nn.Parameter(peak_rate)
        
        # orientation
        self.ori_scalling= 0.01 # to make the gradient for orientation similar to other parameters
        ori = torch.tensor(orientation/self.ori_scalling, requires_grad=True,dtype=torch.float32) # start with 60 degree orientation
        self.ori = torch.nn.Parameter(ori)
       
        
        ## matrix to get the cos and sin component for our 2,1 projection matrix
        self.myMatCos = torch.tensor([[1],[0]],dtype=torch.float32)
        self.myMatSin = torch.tensor([[0],[1]],dtype=torch.float32)
       
        
    def length_to_angle(self,x,period):
        xr = x/period*np.pi*2
        return (torch.atan2(torch.sin(xr), torch.cos(xr)))

    def forward(self, X):
       
        # matrix to project onto the x axis and keep only the x coordinate
        self.sori = self.ori * self.ori_scalling
       
        self.Rx0 = self.myMatCos @ torch.cos(-self.sori[0].reshape(1,1))+ self.myMatSin @ -torch.sin(-self.sori[0].reshape(1,1)) # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0
        self.Rx1 = self.myMatCos @ torch.cos(-self.sori[1].reshape(1,1)) + self.myMatSin @ -torch.sin(-self.sori[1].reshape(1,1))
        self.Rx2 = self.myMatCos @ torch.cos(-self.sori[2].reshape(1,1)) + self.myMatSin @ -torch.sin(-self.sori[2].reshape(1,1))
         
        
        # distance in cm along each axis
        d0 = X @ self.Rx0
        d1 = X @ self.Rx1
        d2 = X @ self.Rx2
        
        c0 = self.length_to_angle(d0, self.period[0])
        c1 = self.length_to_angle(d1, self.period[1]) 
        c2 = self.length_to_angle(d2, self.period[2])

        # deal with the offset, project on each vector 
        d0 = self.offset @ self.Rx0
        d1 = self.offset @ self.Rx1
        d2 = self.offset @ self.Rx2

        # offset as angle
        a0 = self.length_to_angle(d0, self.period[0])
        a1 = self.length_to_angle(d1, self.period[1])
        a2 = self.length_to_angle(d2, self.period[2])
      
        rateC0 = torch.cos(c0-a0)
        rateC1 = torch.cos(c1-a1)
        rateC2 = torch.cos(c2-a2)

        rate = (rateC0+rateC1+rateC2+1.5)/4.5*self.peak_rate
        return rate

    def modelParamToGridParam(self):
        """
        Return the model parameters as a dictionary
        """
        myIter = iter(self.parameters())
        pred_grid_param = {}
        pred_grid_param["period"] = next(myIter).detach().numpy()
        pred_grid_param["offset"] = next(myIter).detach().numpy()
        pred_grid_param["peak_rate"] = next(myIter).detach().numpy()

        ori = next(myIter).detach().numpy() * self.ori_scalling
        orientation = np.array([ori[0],ori[1],ori[2]])

        pred_grid_param["orientation"] = orientation
        return pred_grid_param


def figurePanelDefaultSize():
    """
    Use to keep the size of panels similar across figures
    """
    return (1.8,1.8)



def invalidate_surrounding(myMap,cm_per_bin=3, valid_radius_cm=50):
    xs,ys = np.meshgrid(np.arange(0,myMap.shape[0]),np.arange(0,myMap.shape[1]))
    midPoint=(myMap.shape[0]/2,myMap.shape[1]/2)
    distance = np.sqrt((xs.T-midPoint[0])**2 + (ys.T-midPoint[1])**2) * cm_per_bin
    myMap[distance>valid_radius_cm]=np.nan


def fit_grid_parameter_from_grid_cell_activity(n,ap,apSim,cm_per_bin = 3,xy_range=np.array([[-50,-90],[50,60]]),n_epochs=5000):
    """
    Function that finds the best grid cell model parameters (period, orientation, peak rate, offset) that predict the firing rate of the neuron
    
    The model takes x,y position as input and try to predict the firing rate of the neuron.
    
    We start with an estimate from firing rate map (offset) and spatial autocorrelation (orientation, spacing), then fit some grid cell models with gradient descent.
    
    
    """
    interval = n.ap.intervals.inter

    rowSize,colSize= 2,2
    
    ap.set_intervals(interval)
    n.spike_train.set_intervals(interval)
    n.spatial_properties.set_intervals(interval)
    n.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    n.spatial_properties.spatial_autocorrelation_map_2d(min_n_for_correlation=50,invalid_to_nan=True)
    n.spatial_properties.grid_score()
    n.spike_train.instantaneous_firing_rate(bin_size_sec = 0.02,sigma=1,outside_interval_solution="remove")

    gcSpacing = n.spatial_properties.grid_info()[0]
    gcPeakLocation = n.spatial_properties.firing_rate_map_peak_location()

    # get a template spatial autocorrelation with orientation set at 0
    grid_param = {}
    period = gcSpacing * np.cos(np.pi/6)
    grid_param["period"] = np.array([period,period,period])
    grid_param["offset"] = gcPeakLocation[1]
    grid_param["peak_rate"] = 25
    grid_param["orientation"] = np.array([0,np.pi/3,np.pi/3*2]) # 30 degree orientation means that the field to the right will be at 1,0
    apSim.set_intervals(interval)

    print(grid_param)
    sgc = Simulated_grid_cell(name="pc1",
                              offset=grid_param["offset"],
                              orientation=grid_param["orientation"],
                              period=grid_param["period"],
                              peak_rate=grid_param["peak_rate"],
                              ap=apSim)

    apSim.set_intervals(interval)
    sgc.spike_train.set_intervals(interval)
    sgc.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    sgc.spatial_properties.spatial_autocorrelation_map_2d(min_n_for_correlation=50,invalid_to_nan=True)

    refAuto = n.spatial_properties.spatial_autocorrelation_map.copy()
    invalidate_surrounding(myMap=refAuto,cm_per_bin=cm_per_bin,valid_radius_cm=gcSpacing+gcSpacing*0.4)
    refAutoSgc = sgc.spatial_properties.spatial_autocorrelation_map.copy()
    invalidate_surrounding(myMap=refAutoSgc,cm_per_bin=cm_per_bin,valid_radius_cm=gcSpacing+gcSpacing*0.4)
    autoStackReal = np.expand_dims(refAuto,0)
    autoStackSim = np.expand_dims(refAutoSgc,0)
    
    rot,cor = rotation_correlations(autoStackReal,autoStackSim,minRotation=0,maxRotation=np.pi+0.174,nRotations=180+10)
    peak_indices,_ = find_peaks(cor,height=0.3,distance=10)
    deltas = rot[peak_indices]
    
    print("Results after rotating the spatial autocorrelation")
    print("First axis at ",deltas[0])
    ncols=1
    nrows=1
    fig = plt.figure(figsize=(ncols*4, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    ax = fig.add_subplot(mainSpec[0])
    ax.plot(rot,cor)
    ax.scatter(rot[peak_indices],cor[peak_indices],color="red",s=10)
    ax.set_xlabel("Rotation")
    ax.set_ylabel("r value")
    plt.show()
    
    
    if len(deltas) > 4:
        raise ValueError("Expect less than 5 peaks while rotating spatial autocorrelation but got {}".format(len(deltas)))
    
    if len(deltas) < 3:
        raise ValueError("Expect at least 5 peaks while rotating spatial autocorrelation but got {}".format(len(deltas)))
    grid_param["orientation"] = np.array(deltas[0:3]) # take the first 3
    
   
    # get a simulated grid cells with our initial grid parameters
    sgc = Simulated_grid_cell(name="pc1",
                              offset=grid_param["offset"],
                              orientation=grid_param["orientation"],
                              period=grid_param["period"],
                              peak_rate=grid_param["peak_rate"],
                              ap=apSim)

    apSim.set_intervals(interval)
    sgc.spike_train.set_intervals(interval)
    sgc.spatial_properties.firing_rate_map_2d(cm_per_bin =cm_per_bin, smoothing_sigma_cm = 5, smoothing=True,xy_range=xy_range)
    
    print("Visualize real and simulated grid pattern before fitting")
    ncols=2
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    ax = fig.add_subplot(mainSpec[0])
    plotMap(ax,n.spatial_properties.firing_rate_map,
            title="Data - {:.2f} Hz".format(np.nanmax(n.spatial_properties.firing_rate_map)),
            titleY=0.95,titleFontSize=9,transpose=True,cmap="jet",vmin=0)
    ax = fig.add_subplot(mainSpec[1])
    plotMap(ax,sgc.spatial_properties.firing_rate_map,
            title="Sim - {:.2f} Hz".format(np.nanmax(sgc.spatial_properties.firing_rate_map)),
            titleY=0.95,titleFontSize=9,transpose=True,cmap="jet",vmin=0)
    plt.show()
    
    # get the data that will be used for modelling
    ap.set_intervals(interval)
    
    # trick to get aligned ifr and pose data
    modInterval = interval.copy()
    modInterval[0,0] = ap.pose[0,0]
    modInterval[0,1] = ap.pose[-1,0]+0.00000001
    
    n.spike_train.set_intervals(modInterval)
    n.spike_train.instantaneous_firing_rate(bin_size_sec = 0.02,sigma=1,shift_start_time = -0.0099999999,
                                            outside_interval_solution="remove")

    poseLong = ap.pose[:,1:3]
    rateLong = n.spike_train.ifr[0].copy()
        
    # remove np.nan
    keepIndices = ~np.any(np.isnan(poseLong),1)
    rate = rateLong[keepIndices]
    pose = poseLong[keepIndices]
    #print(rate.shape, pose.shape)
    
    # transform to tensors
    tpose = torch.tensor(pose,dtype=torch.float32)
    trate = torch.tensor(np.expand_dims(rate,1),dtype=torch.float32)
    print("Shape of tensors used for training:",tpose.shape,trate.shape)
    
    
    # get a rigid grid model
    rgcModel = RigidGridCellModel(period=grid_param["period"][0], 
                              peak_rate=grid_param["peak_rate"],
                              orientation=grid_param["orientation"][0],
                              offset=grid_param["offset"])
    
    grid_param_model_start = rgcModel.modelParamToGridParam()
    grid_param_model_start
    print("Fitting rigid grid cell model")
    loss_rigid = training_loop_grid_parameters(n_epochs = n_epochs,
                          model=rgcModel,
                          optimizer=torch.optim.Adam(rgcModel.parameters(),lr=0.01),
                          loss_fn = torch.nn.MSELoss(),
                          X = tpose,
                          y = trate,
                         verbose=False)
    loss_rigid = loss_rigid.detach().numpy()
    print("Loss after rigid model fitting:",loss_rigid)
    
    grid_param_model_rigid = rgcModel.modelParamToGridParam()
    sgcRigid = Simulated_grid_cell(name="pc1",
                          offset=grid_param_model_rigid["offset"],
                          orientation=grid_param_model_rigid["orientation"],
                          period=grid_param_model_rigid["period"],
                          peak_rate=grid_param_model_rigid["peak_rate"],
                          ap=apSim)
    
    sgcRigid.spatial_properties.firing_rate_map_2d(cm_per_bin=cm_per_bin,smoothing_sigma_cm=3, xy_range=xy_range)
    rStart = sgcRigid.spatial_properties.map_pearson_correlation(map1=n.spatial_properties.firing_rate_map,
                                            map2=sgc.spatial_properties.firing_rate_map)
    rRigid = sgcRigid.spatial_properties.map_pearson_correlation(map1=n.spatial_properties.firing_rate_map,
                                            map2=sgcRigid.spatial_properties.firing_rate_map)
    print("Improvement of firing rate maps correlation after rigid fitting: from {:.3f} to {:.3f}".format(rStart, rRigid))
    
    gcModel = GridCellModel(period=grid_param_model_rigid["period"], 
                        peak_rate=grid_param_model_rigid["peak_rate"], 
                        orientation=grid_param_model_rigid["orientation"],
                        offset=grid_param_model_rigid["offset"])
    print("Fitting more flexible grid cell model")
    loss_flexible = training_loop_grid_parameters(n_epochs = n_epochs,
                          model=gcModel,
                          optimizer=torch.optim.Adam(gcModel.parameters(),lr=0.01),
                          loss_fn = torch.nn.MSELoss(),
                          X = tpose,
                          y = trate,
                         verbose=False)
    loss_flexible = loss_flexible.detach().numpy()
    print("Loss after flexible model fitting:",loss_flexible)
    grid_param_model_flexible = gcModel.modelParamToGridParam()
    
    sgcFlexible = Simulated_grid_cell(name="pc1",
                          offset=grid_param_model_flexible["offset"],
                          orientation=grid_param_model_flexible["orientation"],
                          period=grid_param_model_flexible["period"],
                          peak_rate=grid_param_model_flexible["peak_rate"],
                          ap=apSim)
    sgcFlexible.spatial_properties.firing_rate_map_2d(cm_per_bin=cm_per_bin,smoothing_sigma_cm=3, xy_range=xy_range)
    rFlexible = sgc.spatial_properties.map_pearson_correlation(map1=n.spatial_properties.firing_rate_map,
                                                map2=sgcFlexible.spatial_properties.firing_rate_map)
    print("Improvement of firing rate maps correlation after flexible fitting: from {:.3f} to {:.3f}".format(rRigid,rFlexible))
    
    print("Comparison of firing rate maps after fitting different models")
    
    ncols=4
    nrows=1
    fig = plt.figure(figsize=(ncols*colSize, nrows*rowSize), constrained_layout=True) # create a figure
    mainSpec = fig.add_gridspec(ncols=ncols, nrows=1)

    ax = fig.add_subplot(mainSpec[0])
    plotMap(ax,n.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Original",fontsize=9)

    ax = fig.add_subplot(mainSpec[1])
    plotMap(ax,sgc.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated not fitted\n {:.3f}".format(rStart),fontsize=9)

    ax = fig.add_subplot(mainSpec[2])
    plotMap(ax,sgcRigid.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated rigid fit\n {:.3f}".format(rRigid),fontsize=9)

    ax = fig.add_subplot(mainSpec[3])
    plotMap(ax,sgcFlexible.spatial_properties.firing_rate_map)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simulated flexible fit\n {:.3f}".format(rFlexible),fontsize=9)

    plt.show()
    
    res = {"name":n.name,
           "grid_param_initial":grid_param,
           "grid_param_model_rigid": grid_param_model_rigid,
           "loss_rigid": loss_rigid,
           "r_rigid": rRigid,
           "grid_param_model_flexible": grid_param_model_flexible,
           "loss_flexible": loss_flexible,
           "r_flexible": rFlexible}

    n.spike_train.unset_intervals()
    n.spike_train.set_intervals(interval)
    
    #n.spatial_properties.firing_rate_map_2d()
    
    return res


def map_cor(a,b):
    """
    Correlation coefficient between two firing rate maps
    
    Arguments:
    a: 2D np.array (map1)
    b: 2D np.array (map2)
    
    Returns:
    Pearson correlation coefficient between a and b
    """
    a = a.flatten()
    b = b.flatten()
    indices1 = np.logical_and(~np.isnan(a), ~np.isnan(b))
    indices2 = np.logical_and(~np.isinf(a), ~np.isinf(b))
    indices = np.logical_and(indices1,indices2)
    if np.sum(indices)<2:
        return np.nan
    r,p = pearsonr(a[indices],b[indices])
    return r


def rotate_map(a,rotation_radian=np.pi/6):
    """
    Rotate the values in a map around the center of the map
    
    Arguments:
    a: 2D Numpy array
    rotation_radian: angle of the rotation in radian, positive is anti-clockwise
    
    Return:
    2D Numpy array with values rotated
    """

    (h, w) = a.shape
    (cX, cY) = (w // 2, h // 2) # center of the rotation
    degree = rotation_radian/(2*np.pi)*360

    # rotate by degree°, same scale
    M = cv2.getRotationMatrix2D(center = (cX, cY), angle = degree, scale = 1.0)
    a_rotated = cv2.warpAffine(a, M, (w, h), borderValue = np.nan)
    return a_rotated

def rotation_correlations(ccStack1,ccStack2,minRotation=-np.pi,maxRotation=np.pi,nRotations=72):
    """
    We rotate the maps of ccStack2 and perform a correlation coefficient between individual cc.
    We create a 2D array of correlation coefficients (rotation x cell_pairs)
    Then we get the mean coefficient at each rotation
    """
    rotations = np.linspace(minRotation,maxRotation,nRotations)
    rotatedStack = np.empty_like(ccStack2)
    corrValues = np.empty((rotations.shape[0],ccStack2.shape[0]))

    for i,r in enumerate(rotations):
        for j in range(ccStack2.shape[0]): # rotate individual maps
            rotatedStack[j,:,:] = rotate_map(ccStack2[j,:,:],rotation_radian=r)
            corrValues[i,j] = map_cor(ccStack1[j,:,:],rotatedStack[j,:,:])

    peaks = np.mean(corrValues,axis=1)
    return rotations, peaks