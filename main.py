
import os
import numpy as np
import pandas as pd

import scipy.io as sio
import mne

import pycartool
import pickle

from utils import create_roi_files, project_dipoles_inv3D

## Create .rois files

subjects = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
bids_dir = "/home/localadmin/Documents/research/0_data/DS001_BIDS/"
cartool_dir = os.path.join(bids_dir,'derivatives/Cartool-v3.80')
cmp3_dir = os.path.join('derivatives/cmp-v3.0.0-beta-RC1/')
eeglab_dir = os.path.join('derivatives/eeglab')

for scale in [1,2,3]:
    create_roi_files(subects, scale, cartool_dir, cmp3_dir)


## Perform source reconstruction

# time vector
tvec = np.arange(-1500,1000,4)
# parcellation metadata
roifname = '/home/localadmin/Documents/research/0_data/Lausanne2008_Yeo7RSNs.xlsx' 
# time for flipping svd solution sign
deflect_time = np.where((tvec>200)&(tvec<300))[0]

# for different parcellation scales.
for scale in [3,2,1]:
    roidata = pd.read_excel(roifname,sheet_name='SCALE {}'.format(scale))
    
    # Loop across subjects
    for s, sub in enumerate(subjects):    
        subject = str(sub).zfill(2)
        print('Subject {}/{}'.format(s+1,len(subjects)))
        
        # get ROI parcellations index
        roi_filename = os.path.join(cartool_dir','sub-'+subject,'More','scale{}.pickle.rois'.format(scale))
        #roi_file = pycartool.regions_of_interest.read_roi(roi_filename)
        pickle_in = open(roi_filename,"rb")
        roi_file = pickle.load(pickle_in)
        pickle_in.close()
                
        # Select only cortical areas
        cort = np.where(roidata['Structure']=='cort')[0]
        cort_missing = np.array([elem for elem in cort if str(elem+1) not in roi_file.names])
        cort = np.array([roi_file.names.index(str(elem+1)) for elem in cort if str(elem+1) in roi_file.names])
        
        print('Missing cortical areas:', cort_missing)
        
        # get IS (LAURA inverse solution)
        is_filename = os.path.join(cartool_dir,'sub-'+subject,'sub-'+subject+'.LAURA.is')
        is_file = pycartool.io.inverse_solution.read_is(is_filename)

        # load EEG data (all trials)

        spi_fname = os.path.join(cartool_dir,'sub-'+subject,'sub-'+subject+'.spi')
        spi = pycartool.source_space.read_spi(spi_fname)

        eegpath = os.path.join(eeglab_dir,'sub-'+subject)

        # Path to EEGlab files
        set_file = [elem.split('.')[0] for elem in os.listdir(eegpath) if (elem.endswith('.set') and ('FACES' in elem))]

        EEG_path = os.path.join(eegpath,set_file[0]+'.set') 
        
        # Load epoch metadata into a table with header [COND, ACC, RT, nameCOND, outliers, bat_trials]
        Behav_path = os.path.join(eegpath,'sub-'+subject+'_FACES_250HZ_behav.txt') 
        Behavfile = pd.read_csv(Behav_path, sep=",")
        Behavfile = Behavfile[Behavfile.bad_trials == 0]

        epochs = mne.read_epochs_eeglab(EEG_path, events=None, event_id=None, eog=(), verbose=None, uint16_codec=None)
        epochs.events[:,2] = list(Behavfile.COND)
        epochs.event_id = {"Scrambled":0, "Faces":1}
                # compute inverse solutions for each ROI
        r = 6

        IS = is_file['regularisation_solutions'][r]
        
        TCs_1D,SCS,nSPI,maindir = project_dipoles_inv3D(epochs,roi_file,IS,spi,epochs.times,verbose=None)
        X = np.array(TCs_1D)[cort]
        
        # Flip ROIs 
        for n in range(X.shape[0]): 
            direction = np.sign(X[n][deflect_time].mean())
            X[n] *= direction
            
        
        outdir = os.path.join(bids_dir,'sourcedata/Faces/scale{}'.format(scale)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
         
        outfile = os.path.join(bids_dir,'sourcedata/Faces/scale{}/sub-{}.npy'.format(scale,str(subjects[s]).zfill(2))
        np.save(outfile,X)