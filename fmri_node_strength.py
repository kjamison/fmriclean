import numpy as np
import nibabel as nib
import sys
import argparse
from scipy.io import loadmat,savemat
import os.path
import re
from utils import *
from tqdm import tqdm

def argument_parse(argv):
    parser=argparse.ArgumentParser(description='Voxelwise connectivity strength *after* denoising')

    parser.add_argument('--input',action='append',dest='inputvol',nargs='*')
    parser.add_argument('--confoundfile',action='append',dest='confoundfile',nargs='*')
    parser.add_argument('--outbase',action='append',dest='outbase',nargs='*')
    parser.add_argument('--skipvols',action='store',dest='skipvols',type=int,default=5)
    parser.add_argument('--outlierfile',action='append',dest='outlierfile',nargs='*')
    parser.add_argument('--mask',action='append',dest='maskfile',help='mask over which to compute connectivity strength (eg: gray matter voxels)',nargs='*')
    parser.add_argument('--cctransform',action='store',dest='cctransform',choices=['none','abs','pos','neg'],default='none')
    parser.add_argument('--outputvolumeformat',action='store',dest='outputvolumeformat',choices=['same','auto','nii','nii.gz'],default='same')
    parser.add_argument('--blocksize',action='store',dest='blocksize',type=int,default=200,help='Block size to limit memory usage (default: 200)')
    parser.add_argument('--verbose',action='store_true',dest='verbose')
    
    parser.add_argument('--version', action='version',version=package_version_dict(as_string=True))
    
    return parser.parse_args(argv)

def fmri_node_strength(argv):
    args=argument_parse(argv)
    inputvol_list=flatarglist(args.inputvol)
    outbase_list=flatarglist(args.outbase)
    skipvols=args.skipvols
    outlierfile_list=flatarglist(args.outlierfile)
    confoundfile_list=flatarglist(args.confoundfile)
    maskfile_list=flatarglist(args.maskfile)
    verbose=args.verbose
    outputvolumeformat=args.outputvolumeformat
    cctransform_type=args.cctransform
    
    
    #200 was fastest for 91k cifti
    #but might want to adjust this for larger voxelwise data?
    blocksize=args.blocksize
    
    is_pattern=False
    input_list=inputvol_list
    num_inputs=len(input_list)


    print("Input time series: %s" % (inputvol_list))
    print("Ignore first N volumes: %s" % (skipvols))
    print("Confound file: %s" % (confoundfile_list))
    print("Outlier timepoint file: %s" % (outlierfile_list))
    print("Output basename: %s" % (outbase_list))
    print("CC transformation: %s" % (cctransform_type))
    print("Mask file: %s" % (maskfile_list))
    print("Block size: %s" % (blocksize))

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

    for inputidx,inputfile in enumerate(input_list):
        confounds_dict=confounds_list[inputidx]

        Dt,roivals,roisizes,tr_input,vol_info,input_extension = load_input(inputfile)
        if vol_info is not None and not outputvolumeformat in ["same","auto"]:
            vol_info["extension"]=outputvolumeformat
    
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
    
        #zscore the time series, excluding outliers
        #note: use 'sum' for pearson-friendly output
        Dt=normalize(Dt[outlierflat==0,:],axis=0,denomfun='sum')
        
        print("Masked data size after outlier exclusion: (%dx%d)" % (Dt.shape[0],Dt.shape[1]))
        

        imax=Dt.shape[1]
        #imax=10000
        if cctransform_type == 'none':
            cc_filt_fun=lambda x: x
        elif cctransform_type == 'abs':
            cc_filt_fun=np.abs
        elif cctransform_type == 'pos':
            cc_filt_fun=lambda x: np.maximum(x,0)
        elif cctransform_type == 'neg':
            cc_filt_fun=lambda x: np.maximum(-x,0)
        
        Dcc=np.hstack([np.mean(cc_filt_fun(Dt[:,i:min(i+blocksize,imax)].T@Dt),axis=1) for i in tqdm(range(0,imax,blocksize))])
        Dcc-=1/Dt.shape[1] #subtract the 1/N for each voxel's entry on the diagonal
        Dcc_new=np.zeros(Dt.shape[1])
        Dcc_new[:Dcc.size]=Dcc
        Dcc=Dcc_new
        
        savedfilename, shapestring = save_timeseries(outbase_list[inputidx]+"", input_extension, {"ts":Dcc,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr}, vol_info)
        print("Saved %s (%s)" % (savedfilename,shapestring))
    
if __name__ == "__main__":
    fmri_node_strength(sys.argv[1:])
