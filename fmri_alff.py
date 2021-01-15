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

def flatlist(l):
    if l is None:
        return []
    return [x for y in l for x in y]
    
def flatarglist(l):
    if l is None:
        return []
    return flatlist([x.split(",") for x in flatlist(l)])

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

args=parser.parse_args()
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

    
def vec2columns(v):
    if(v.ndim==1):
        v=np.atleast_2d(v).T
    if v.shape[1]==1:
        if np.sum(v,axis=0)>0:
            v_new=np.zeros((v.shape[0],int(np.sum(v!=0))))
            for i,t in enumerate(np.nonzero(v>0)[0]):
                v_new[t,i]=1
        else:
            v_new=np.zeros((v.shape[0],0))
    else:
        v_new=np.hstack([vec2columns(v[:,i]) for i in range(v.shape[1])])
    return v_new
    
def nanfft(S,tr,outliermat=None,inverse=False):
    #perform fft (or ifft) ignoring NaN or outlier timepoints
    N=S.shape[0]
    
    if inverse:
        expsign=1
        denom=N
    else:
        expsign=-1
        denom=1
        
    #build FFT basis set
    n=np.atleast_2d(np.arange(N))
    X=np.zeros((N,N),dtype=np.complex)
    for k in range(N):
        X[k,:]=np.exp(expsign*1j*2*np.pi*k*n/N)
        
    notnan=~np.any(np.isnan(S),axis=1)
    if outliermat is not None:
        if outliermat.ndim > 1:
            notnan[np.sum(np.abs(outliermat),axis=1)>0]=False
        else:
            notnan[outliermat!=0]=False

    fy=X[:,notnan] @ S[notnan,:] / denom
    
    f=np.arange(N)
    f=np.minimum(f,N-f)/(tr*N)
    
    return fy, f

def loadinput(filename):
    tr=None
    volinfo=None
    roivals=None
    roisizes=None
    extension=None
    if filename.lower().endswith(".mat"):
        M=loadmat(filename)
        Dt=M['ts']
        if 'repetition_time' in M:
            tr=M['repetition_time'][0]
        roivals=M['roi_labels'][0]
        roisizes=M['roi_sizes'][0]
        extension="mat"
    elif filename.lower().endswith(".nii.gz") or filename.lower().endswith(".nii"):
        if filename.lower().endswith(".gz"):
            volext=".".join(filename.lower().split(".")[-2:])
        else:
            volext=filename.lower().split(".")[-1]
            
        Vimg=nib.load(filename)
        V=Vimg.get_fdata()
        M=np.any(V!=0,axis=3)
        Dt=V[M>0].T
        tr=Vimg.header['pixdim'][4]
        volinfo={'image':Vimg, 'shape':Vimg.shape, 'mask':M, "extension":volext}
        extension=volext
    elif filename.lower().endswith(".txt"):
        Dt=np.loadtxt(filename)
    
        fid = open(filename, 'r') 
        roivals=np.arange(1,Dt.shape[1]+1)
        roisizes=np.ones(len(roivals))
        while True: 
            line=fid.readline()
            if not line or not line.startswith("#"):
                break
            if line.find("ROI_Labels:")>0:
                roivals=fid.readline().strip().split("#")[-1].split()
                continue
            if line.find("ROI_Sizes")>0:
                roisizes=fid.readline().strip().split("#")[-1].split()
                continue
            if line.find("Repetition_time(sec)")>0:
                tr=float(line.strip().split("#")[-1])
                continue
        fid.close()
        roivals=np.array([float(x) for x in roivals])
        roisizes=np.array([float(x) for x in roisizes])
        extension="txt"
    else:
        raise Exception("Unknown input data file type: %s" % (filename))
    return Dt,roivals,roisizes,tr,volinfo,extension
    
def save_timeseries(filename_noext,outputformat,output_dict, output_volume_info=None):
    outfilename=""
    shapestring=""
    if output_volume_info is not None:
        Vimg_orig=output_volume_info['image']
        outshape=list(Vimg_orig.shape[:3])
        if output_dict["ts"].ndim > 1:
            outshape+=[output_dict["ts"].shape[0]]
        #output_dtype=Vimg_orig.get_data_dtype()
        output_dtype=np.float32
        Vnew=np.zeros(outshape,dtype=output_dtype)
        Vnew[output_volume_info['mask']]=output_dict["ts"].T
        Vimg=nib.Nifti1Image(Vnew.astype(output_dtype),affine=Vimg_orig.affine,header=Vimg_orig.header)
        outfilename=filename_noext+"."+output_volume_info["extension"]
        shapestring="x".join([str(x) for x in Vimg.shape])
        nib.save(Vimg, outfilename)
    else:
        output_dict["ts"]=np.atleast_2d(output_dict["ts"])
        shapestring="%dx%d" % (output_dict["ts"].shape[0],output_dict["ts"].shape[1])
        if outputformat == "mat":
            outfilename=filename_noext+"."+outputformat
            savemat(outfilename,output_dict,format='5',do_compression=True)
        else:
            headertxt="ROI_Labels:\n"
            headertxt+=" ".join(["%d" % (x) for x in output_dict["roi_labels"]])
            headertxt+="\nROI_Sizes(voxels):\n"
            headertxt+=" ".join(["%d" % (x) for x in output_dict["roi_sizes"]])
            headertxt+="\nRepetition_time(sec): %g" % (output_dict["repetition_time"])
            outfilename=filename_noext+"."+outputformat
            np.savetxt(outfilename,output_dict["ts"],fmt="%.18f",header=headertxt,comments="# ")
    return outfilename, shapestring
    

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

    Dt,roivals,roisizes,tr_input,vol_info,input_extension = loadinput(inputfile)
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
    
    ts_alff=np.mean(F[(freq>=lffrange[0]) & (freq<=lffrange[1]),:],axis=0)
    ts_falff =ts_alff / np.mean(F[(freq>=bpf[0]) & (freq<=bpf[1]),:],axis=0)
    
    savedfilename, shapestring = save_timeseries(outbase_list[inputidx]+"_alff", input_extension, {"ts":ts_alff,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr}, vol_info)
    print("Saved %s (%s)" % (savedfilename,shapestring))
    
    savedfilename, shapestring = save_timeseries(outbase_list[inputidx]+"_falff", input_extension, {"ts":ts_falff,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr}, vol_info)
    print("Saved %s (%s)" % (savedfilename,shapestring))