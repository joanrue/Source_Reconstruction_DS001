import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pdg

import scipy.io as sio
import mne

import pycartool
import pickle

import nibabel


def project_dipoles_inv3D(EEGfile,roi_file,IS,spi_file,time,verbose=True):
    """ 
    This function gathers the functions solveinv3d.m and project_dipoles.m
    
    Last modification 13/08/2020 (Joan Rué)"""
    
    eegdata = EEGfile.get_data()
    conds = EEGfile.events[:,2]   
    nconds = len(np.unique(conds))
    N = len(roi_file.names) #% number of ROIs
    E, _, T = eegdata.shape # Epochs,  _ , Time
    TCs_1D = np.zeros((N,T,E)) # full time courses for each epoch projected on main dir
    nSPI = np.zeros((N),dtype = int);
    maindir = np.zeros((N,3))
    
    vertices_left = [idx for idx,coords in enumerate(spi_file.coordinates) if coords[0]<0]
    vertices_right = [idx for idx,coords in enumerate(spi_file.coordinates) if coords[0]>=0]
    vertices = [vertices_left,vertices_right]
    SCS = []
    for cond in range(nconds):        
        mne.SourceEstimate(np.zeros((IS.shape[1],T)),vertices,tmin=EEGfile.tmin,tstep=EEGfile.times[1]-EEGfile.times[0],verbose=verbose)
        SCS.append(mne.SourceEstimate(np.zeros((IS.shape[1],T)),vertices,tmin=EEGfile.tmin,tstep=EEGfile.times[1]-EEGfile.times[0]))

    for n in range(N):        
        if verbose is not None:
            print('ROI {} of {}\n'.format(n+1,N))
        #spis_this_roi = list(map(lambda x: int(x)-1,roi_file.groups_of_indexes[n]))
        spis_this_roi = roi_file.groups_of_indexes[n]
        nSPI[n] = int(len(spis_this_roi))
        source_data_roi = np.zeros((3,nSPI[n],T,E))        
        
        for e in range(E):       
            source_data_roi[0,:,:,e] = IS[0,spis_this_roi] @ eegdata[e,:,:]
            source_data_roi[1,:,:,e] = IS[1,spis_this_roi] @ eegdata[e,:,:]
            source_data_roi[2,:,:,e] = IS[2,spis_this_roi] @ eegdata[e,:,:]            

        # SVD applied to average over epochs in the time of interest (toi)
        # consider main direction during time when biggest effect of stimulus
        # is expected

        toi_begin = 120     # ms after stim onset
        toi_end = 500       # ms after stim onset
        tstep = int((time[1]-time[0])*1000)   # time step in ms

        stim_onset = np.where(time==0)[0][0]
        ID_begin = stim_onset+int(toi_begin/tstep)
        ID_end = stim_onset+int(toi_end/tstep)
        
        mean_source_data_toi_roi  = np.mean(source_data_roi[:,:,ID_begin:ID_end,:],axis=3)
        u1,_,_ = np.linalg.svd(mean_source_data_toi_roi.reshape(3,-1))

        TCs_loc = np.zeros((T,nSPI[n],E))  
        
        for k in range(nSPI[n]):
            for e in range(E):
                TCs_loc[:,k,e] = u1[:,0].reshape(1,3) @ source_data_roi[:,k,:,e]            
        #_, forflip = group_signflip(TCs_loc.mean(2))
        #TCs_loc *= forflip.reshape(1,-1,1)
        TCs_1D[n,:,:] = np.mean(TCs_loc,axis=1)
        maindir[n,:]  = u1[:,0]
        for i in range(nconds):
            SCS[i].data[spis_this_roi,:] = np.mean(TCs_loc[:,:,conds==i],axis=2).T
    return TCs_1D,SCS,nSPI,maindir

def group_signflip(avg,tol = 0.01):
    # average of multiple subjects (subject, ROI, time)
    # tol (optional) minimum proportion allowed of negative corr
    # output:
    # signflip: ROI x Subj, individual vectors of signs to flip

    # =========================================================================
    # sign flipping across subjects    
    # =========================================================================
    if len(avg.shape)==3:
        flipped   = np.copy(avg)
        while True:
            updatevec = np.zeros((avg.shape[1],avg.shape[0]))
            for k in range(avg.shape[1]):
                vec = np.sign(np.mean(np.corrcoef(flipped[:,k,:]),axis=1))
                updatevec[k,:] = vec
                flipped[:,k,:] = vec[:,None]*flipped[:,k,:];

            if np.mean(updatevec<0)<=tol:
                break

        signflip = np.zeros((avg.shape[1],avg.shape[0]))
        for k in range(avg.shape[1]):
            for ij in range(avg.shape[0]):
                signflip[k,ij] = np.sign(np.corrcoef(avg[ij,k,:],flipped[ij,k,:])[0,1])

        forflip   = 0
    # returns the signs for flipping
    # =========================================================================
    # sign flipping inside roi
    # =========================================================================    
    else:
        flipped   = np.copy(avg)
    
        while True:
            if flipped.shape[1]==1:
                vec = np.expand_dims(np.corrcoef(np.random.randn(625,1).T),0)
            else:
                vec            = np.sign(np.mean(np.corrcoef(flipped.T),axis=1))
            updatevec      = vec
            flipped        = flipped*vec
            if np.mean(updatevec<0)<=tol:
                break

        signflip  = flipped  # returns the flipped signals
        forflip = np.zeros((avg.shape[1]))
        for k in range(avg.shape[1]):            
            forflip[k] = np.sign(np.corrcoef(avg[:,k],flipped[:,k])[0,1])

    return signflip, forflip
    
    
subjects = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
cartool_dir = '/home/localadmin/Documents/research/0_data/DS001_BIDS/derivatives/Cartool-v3.80'
cmp3_dir = '/home/localadmin/Documents/research/0_data/DS001_BIDS/derivatives/cmp-v3.0.0-beta-RC1/'
for scale in [1,2,3]:

def create_roi_files(subects, scale, cartool_dir, cmp3_dir):
    for sub in subjects:
        print('Subject ',sub)
        
        spipath = os.path.join(cartool_dir,'sub-{}/sub-{}.spi'.format(str(sub).zfill(2),str(sub).zfill(2)))

        source = pycartool.source_space.read_spi(spipath)

        brain_cartool = os.path.join(cartool_dir,'sub-{}/sub-{}.Brain.nii'.format(str(sub).zfill(2),str(sub).zfill(2)))
        brain_cartool = nibabel.load(brain_cartool)
        bc = brain_cartool.get_fdata()[:,:,:,0]


           
        impath = os.path.join(cartool_dir,'sub-{}/anat/sub-{}_label-L2008_desc-scale{}_atlas.nii.gz'.format(str(sub).zfill(2),str(sub).zfill(2),scale))
        im = nibabel.load(impath)
        imdata = im.get_fdata()

        x,y,z = np.where(imdata)
        center_brain = [np.mean(x),np.mean(y),np.mean(z)]
        source.coordinates[:,0] = - source.coordinates[:,0]
        source.coordinates = source.coordinates -source.coordinates.mean(0) + center_brain

        xyz = source.get_coordinates()        
        xyz = np.round(xyz).astype(int)
        num_spi = len(xyz)

        # label positions
        rois_file = np.zeros(num_spi)
        x_roi,y_roi,z_roi = np.where((imdata>0)&(imdata<np.unique(imdata)[-1]))

        # For each coordinate
        for spi_id,spi in enumerate(xyz):
            distances = ((spi.reshape(-1,1)-[x_roi,y_roi,z_roi])**2).sum(0)
            roi_id = np.argmin(distances)            
            rois_file[spi_id] = imdata[x_roi[roi_id],y_roi[roi_id],z_roi[roi_id]]

        groups_of_indexes = [np.where(rois_file==roi)[0].tolist() for roi in np.unique(rois_file)]
        names = [str(int(i)) for i in np.unique(rois_file) if i!=0] 

        #print([elem for elem in np.arange(len(np.unique(imdata))-2) if str(elem+1) not in names])

        rois_file_new = pycartool.regions_of_interest.RegionsOfInterest(names = names,
                                                                        groups_of_indexes=groups_of_indexes,
                                                                        source_space=source)        

        filename = os.path.join(cartool_dir,'sub-{}/More/scale{}.pickle.rois'.format(str(sub).zfill(2),scale))
        filehandler = open(filename, 'wb') 
        pickle.dump(rois_file_new, filehandler)
        filehandler.close()
    print('Done')