import numpy as np
import nibabel as nib
import nilearn
import nilearn.connectome
#import nipy.modalities.fmri.hrf
import sys
import argparse
from scipy.io import loadmat,savemat
import scipy.signal, scipy.interpolate
import sklearn
import os.path
import re
from utils import *

def argument_parse(argv):
    parser=argparse.ArgumentParser(description='ALFF and fALFF *after* denoising')

    parser.add_argument('--input',action='append',dest='inputvol',nargs='*')
    parser.add_argument('--confoundfile',action='append',dest='confoundfile',nargs='*')
    parser.add_argument('--outbase',action='append',dest='outbase',nargs='*')
    parser.add_argument('--skipvols',action='store',dest='skipvols',type=int,default=5)
    parser.add_argument('--lffrange',action='store',dest='lffrange',type=float,nargs=2) #,default=0.008)
    parser.add_argument('--totalfreqrange',action='store',dest='totalfreqrange',type=float,nargs=2) #,default=0.008)
    parser.add_argument('--repetitiontime','-tr',action='store',dest='tr',help='TR in seconds',type=float)
    parser.add_argument('--outlierfile',action='append',dest='outlierfile',nargs='*')
    parser.add_argument('--outputvolumeformat',action='store',dest='outputvolumeformat',choices=['same','auto','nii','nii.gz'],default='same')
    parser.add_argument('--verbose',action='store_true',dest='verbose')

    return parser.parse_args(argv)

def fmri_alff(argv):
    args=argument_parse(argv)
    inputvol_list=flatarglist(args.inputvol)
    outbase_list=flatarglist(args.outbase)
    skipvols=args.skipvols
    outlierfile_list=flatarglist(args.outlierfile)
    confoundfile_list=flatarglist(args.confoundfile)
    verbose=args.verbose
    tr=args.tr
    outputvolumeformat=args.outputvolumeformat
    lffrange=np.sort(args.lffrange)

    bpf=[-np.inf,np.inf]
    if args.totalfreqrange:
        bpf=np.sort(args.totalfreqrange)

    do_filter_rolloff=True

    is_pattern=False
    input_list=inputvol_list
    num_inputs=len(input_list)


    print("Input time series: %s" % (inputvol_list))
    print("Low-Frequency Fluctuation range Hz: [%s,%s]" % (lffrange[0],lffrange[1]))
    print("Total frequency range Hz: [%s,%s]" % (bpf[0],bpf[1]))
    print("Ignore first N volumes: %s" % (skipvols))
    print("Confound file: %s" % (confoundfile_list))
    print("Outlier timepoint file: %s" % (outlierfile_list))
    print("Output basename: %s" % (outbase_list))

    # read in confounds (from a confoundfile and/or specified motionparam and outlier arguments)
    confounds_list=[{"gmreg":None,"wmreg":None,"csfreg":None,"mp":None,"resteffect":None,"outliermat":None} for i in range(num_inputs)]

    #read in --confoundfile inputs for each input time series (if provided)
    if len(confoundfile_list)==num_inputs:
        for inputidx,confoundfile in enumerate(confoundfile_list):
            if confoundfile.lower().endswith(".mat"):
                M=loadmat(confoundfile)
                confoundmat=M['confounds']
                confoundnames=M['confoundnames']
            else:
                confoundmat=np.loadtxt(confoundfile)
                fid = open(confoundfile, 'r') 
                line=fid.readline()
                if not line or not line.startswith("#"):
                    print("Confound file does not contain confound names: %s" % (confoundfile)  )
                    sys.exit(1)
                confoundnames=line.strip().split("#")[-1].split()
                fid.close()
        
            gmidx=[i for i,x in enumerate(confoundnames) if x.startswith("GM.")]
            wmidx=[i for i,x in enumerate(confoundnames) if x.startswith("WM.")]
            csfidx=[i for i,x in enumerate(confoundnames) if x.startswith("CSF.")]
            mpidx=[i for i,x in enumerate(confoundnames) if x.startswith("motion.")]
            restidx=[i for i,x in enumerate(confoundnames) if x.startswith("rest")]
            outlieridx=[i for i,x in enumerate(confoundnames) if x.startswith("outlier.")]
        
            #outliermat=np.zeros((numvols,1))
            #resteffect=np.zeros((numvols,0))
            #gmreg=np.zeros((numvols,0))
            #wmreg=np.zeros((numvols,0))
            #csfreg=np.zeros((numvols,0))
            #mp=np.zeros((numvols,0))
        
            if len(gmidx)>0:
                confounds_list[inputidx]["gmreg"]=confoundmat[:,gmidx]
            if len(wmidx)>0:
                confounds_list[inputidx]["wmreg"]=confoundmat[:,wmidx]
            if len(csfidx)>0:
                confounds_list[inputidx]["csfreg"]=confoundmat[:,csfidx]
            if len(mpidx)>0:
                mp=confoundmat[:,mpidx]
                if mp.shape[1]>6:
                    mp=mp[:,:6]
                confounds_list[inputidx]["mp"]=mp
            if len(restidx)>0:
                confounds_list[inputidx]["resteffect"]=confoundmat[:,restidx[0]][:,None]
            if len(outlieridx)>0:
                confounds_list[inputidx]["outliermat"]=confoundmat[:,outlieridx]
            
    #read in --outlierfile inputs if provided, overwriting values from --confoundfile
    if len(outlierfile_list)==num_inputs:
        for inputidx,outlierfile in enumerate(outlierfile_list):
            outliermat=np.loadtxt(outlierfile)>0
            confounds_list[inputidx]["outliermat"]

    for inputidx,inputfile in enumerate(input_list):
        confounds_dict=confounds_list[inputidx]

        Dt,roivals,roisizes,tr_input,vol_info,input_extension = load_input(inputfile)
        if vol_info is not None and not outputvolumeformat in ["same","auto"]:
            vol_info["extension"]=outputvolumeformat
    
        print("Loaded input file: %s (%dx%d)" % (inputfile,Dt.shape[0],Dt.shape[1]))
        if tr_input:
            tr=tr_input

        numvols=Dt.shape[0]
    
        outliermat=np.zeros((numvols,1))
        if confounds_dict["outliermat"] is not None:
            outliermat=confounds_dict["outliermat"]
        
        outliermat=np.sum(vec2columns(outliermat)!=0,axis=1)[:,None]
        outliermat[:skipvols,:]=True
        outliermat=vec2columns(outliermat)
    
        numvols_not_outliers=np.sum(np.sum(np.abs(outliermat),axis=1)==0,axis=0)
        print("Non-outlier volumes: ", numvols_not_outliers)
    
        if do_filter_rolloff:
            filter_edge_rolloff_size=int(36/tr/2)*2+1 #51 for tr=0.72
            filter_edge_rolloff_std=3.6/tr #5 for tr=0.72
            filter_edge_rolloff=scipy.signal.gaussian(filter_edge_rolloff_size,filter_edge_rolloff_std)
        else:
            filter_edge_rolloff=None
        
        F, freq = nanfft(Dt,tr,outliermat=outliermat,inverse=False)
        F=2*np.abs(F)/numvols_not_outliers
    
        #note: falff should be sum(lff)/sum(total) to be fractional
        Nlff=sum((freq>=lffrange[0]) & (freq<=lffrange[1]))
        ts_alff=np.mean(F[(freq>=lffrange[0]) & (freq<=lffrange[1]),:],axis=0)
        ts_falff =ts_alff*Nlff / np.sum(F[(freq>=bpf[0]) & (freq<=bpf[1]),:],axis=0)
    
        savedfilename, shapestring = save_timeseries(outbase_list[inputidx]+"_alff", input_extension, {"ts":ts_alff,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr}, vol_info)
        print("Saved %s (%s)" % (savedfilename,shapestring))
    
        savedfilename, shapestring = save_timeseries(outbase_list[inputidx]+"_falff", input_extension, {"ts":ts_falff,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr}, vol_info)
        print("Saved %s (%s)" % (savedfilename,shapestring))
        
    
if __name__ == "__main__":
    fmri_alff(sys.argv[1:])