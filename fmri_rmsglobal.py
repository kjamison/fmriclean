import numpy as np
import nibabel as nib
import sys
import argparse
from scipy.io import loadmat,savemat
import os.path
import re
from utils import *

def argument_parse(argv):
    parser=argparse.ArgumentParser(description='Compute RMS of global signal fit to each region')

    parser.add_argument('--inputnogsr',action='append',dest='inputnogsr',nargs='*')
    parser.add_argument('--inputgsr',action='append',dest='inputgsr',nargs='*')
    parser.add_argument('--confoundfile',action='append',dest='confoundfile',nargs='*')
    parser.add_argument('--outbase',action='append',dest='outbase',nargs='*')
    parser.add_argument('--skipvols',action='store',dest='skipvols',type=int,default=5)
    parser.add_argument('--outlierfile',action='append',dest='outlierfile',nargs='*')
    parser.add_argument('--mask',action='store',dest='maskfile',help='mask over which to compute connectivity strength (eg: gray matter voxels)',nargs='*')
    parser.add_argument('--outputvolumeformat',action='store',dest='outputvolumeformat',choices=['same','auto','nii','nii.gz'],default='same')
    parser.add_argument('--concat',action='store_true',dest='concat')
    parser.add_argument('--verbose',action='store_true',dest='verbose')
    return parser.parse_args(argv)
    
def run_rmsglobal(argv):
    args=argument_parse(argv)
    inputnogsr_list=flatarglist(args.inputnogsr)
    inputgsr_list=flatarglist(args.inputgsr)
    outbase_list=flatarglist(args.outbase)
    skipvols=args.skipvols
    outlierfile_list=flatarglist(args.outlierfile)
    confoundfile_list=flatarglist(args.confoundfile)
    maskfile_list=flatarglist(args.maskfile)
    verbose=args.verbose
    outputvolumeformat=args.outputvolumeformat
    do_concat=args.concat

    is_pattern=False
    input_list=inputnogsr_list
    num_inputs=len(input_list)


    print("Input NON-GSR time series: %s" % (inputnogsr_list))
    print("Input GSR time series: %s" % (inputgsr_list))
    print("Ignore first N volumes: %s" % (skipvols))
    print("Confound file: %s" % (confoundfile_list))
    print("Outlier timepoint file: %s" % (outlierfile_list))
    print("Output basename: %s" % (outbase_list))
    print("Mask file: %s" % (maskfile_list))

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
            
            outlieridx=[i for i,x in enumerate(confoundnames) if x.startswith("outlier.")]
            
            if len(outlieridx)>0:
                confounds_list[inputidx]["outliermat"]=confoundmat[:,outlieridx]
            
    #read in --outlierfile inputs if provided, overwriting values from --confoundfile
    if len(outlierfile_list)==num_inputs:
        for inputidx,outlierfile in enumerate(outlierfile_list):
            outliermat=np.loadtxt(outlierfile)>0
            confounds_list[inputidx]["outliermat"]

    Dt_concat=[]
    for inputidx,inputfile in enumerate(inputnogsr_list):
        confounds_dict=confounds_list[inputidx]

        Dt,roivals,roisizes,tr_input,vol_info,input_extension = load_input(inputfile)
        if vol_info is not None and not outputvolumeformat in ["same","auto"]:
            vol_info["extension"]=outputvolumeformat
        
        Dt_gsr,roivals,roisizes,tr_input,vol_info,input_extension = load_input(inputgsr_list[inputidx])
        
        Dt=Dt-Dt_gsr
        
        print("Loaded input file: %s (%dx%d)" % (inputfile,Dt.shape[0],Dt.shape[1]))
        if tr_input:
            tr=tr_input

        mask=None
        if len(maskfile_list)==num_inputs:
            maskfile=maskfile_list[inputidx]
            mask,_,_,_,mask_vol_info,_ = load_input(maskfile)
            masksize=list(mask.shape[:2])+[1]
            print("Loaded mask file: %s (%s)" % (maskfile,"x".join([str(x) for x in masksize[:2]])))
            
            if mask_vol_info is not None and vol_info is not None:
                #map mask to full voxel space (and intersectc with input data mask)
                mask_full=np.zeros(mask_vol_info['mask'].shape)
                mask_full[mask_vol_info['mask']]=mask
                mask_full=(mask_full*vol_info['mask'])>0
                
                #then map mask from full voxel space to masked data space
                mask=mask_full[vol_info['mask']>0]
                
                vol_info['mask']=mask_full
                Dt=Dt[:,mask]
            else:
                mask=None
        
        numvols=Dt.shape[0]
        
        outliermat=np.zeros((numvols,1))
        if confounds_dict["outliermat"] is not None:
            outliermat=confounds_dict["outliermat"]
        
        outlierflat=np.sum(vec2columns(outliermat)!=0,axis=1)[:,None]
        outlierflat[:skipvols,:]=True
        outlierflat=outlierflat[:,0]
        
        numvols_not_outliers=np.sum(np.abs(outlierflat)==0,axis=0)
        print("Non-outlier volumes: ", numvols_not_outliers)
        
        print("Masked data size after outlier exclusion: (%dx%d)" % (Dt.shape[0],Dt.shape[1]))
        
        Dt_rms=np.sqrt(np.mean(Dt[outlierflat==0,:]**2,axis=0))
        
        if do_concat:
            Dt_concat+=[Dt_rms]
        
        if len(outbase_list)==num_inputs:
            savedfilename, shapestring = save_timeseries(outbase_list[inputidx]+"", input_extension, {"ts":Dt_rms,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr}, vol_info)
            print("Saved %s (%s)" % (savedfilename,shapestring))
    
    if do_concat and len(Dt_concat)>1:
        Dt_concat=np.mean(np.vstack(Dt_concat),axis=0)
        savedfilename, shapestring = save_timeseries(outbase_list[0]+"", input_extension, {"ts":Dt_concat,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr}, vol_info)
        print("Saved %s (%s)" % (savedfilename,shapestring))
if __name__ == "__main__":
    run_rmsglobal(sys.argv[1:])
