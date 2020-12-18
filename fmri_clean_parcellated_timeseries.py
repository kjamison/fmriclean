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

parser=argparse.ArgumentParser(description='fMRI Denoising after parcellation')

parser.add_argument('--input',action='append',dest='inputvol',nargs='*')
parser.add_argument('--confoundfile',action='append',dest='confoundfile',nargs='*')
parser.add_argument('--outbase',action='append',dest='outbase',nargs='*')
parser.add_argument('--inputpattern',action='append',dest='inputpattern',nargs='*',help='Pattern with "%%s" to be replaced with names from --roilist')
parser.add_argument('--roilist',action='append',dest='roilist',nargs='*')
parser.add_argument('--savets',action='store_true',dest='savets')
parser.add_argument('--skipvols',action='store',dest='skipvols',type=int,default=5)
parser.add_argument('--lowfreq',action='store',dest='lowfreq',type=float) #,default=0.008)
parser.add_argument('--highfreq',action='store',dest='highfreq',type=float)# ,default=0.09)
parser.add_argument('--repetitiontime','-tr',action='store',dest='tr',help='TR in seconds',type=float)
parser.add_argument('--filterstrategy',action='store',dest='filterstrategy',choices=['connregbp','orth','parallel','none'],default='connregbp')
parser.add_argument('--connmeasure',action='append',dest='connmeasure',choices=['none','correlation','partialcorrelation','precision','covariance'],nargs='*')
parser.add_argument('--outputformat',action='store',dest='outputformat',choices=['mat','txt'],default='mat')
parser.add_argument('--gsr',action='store_true',dest='gsr')
parser.add_argument('--nocompcor',action='store_true',dest='nocompcor')
parser.add_argument('--nomotion',action='store_true',dest='nomotion')
parser.add_argument('--hrffile',action='store',dest='hrffile')
parser.add_argument('--motionparamtype',action='store',dest='mptype',choices=['spm','hcp','fsl'],default='fsl')
parser.add_argument('--motionparam',action='append',dest='mpfile',nargs='*')
parser.add_argument('--outlierfile',action='append',dest='outlierfile',nargs='*')
parser.add_argument('--shrinkage',action='store',dest='shrinkage',default='lw')
parser.add_argument('--sequentialroi',action='store_true',dest='sequentialroi',help='Output columns for ALL sequential ROI values from 1:max (otherwise exactly the same columns as input)')
parser.add_argument('--sequentialroierrorsize',action='store',dest='sequentialroierrorsize',type=int,default=1000,help='Throw error if using --sequential and largest ROI label is larger than this')
parser.add_argument('--concat',action='store_true',dest='concat',help='Concatenate time series when multiple --input or --inputpattern are given (need multiple --confound in this case as well)')
parser.add_argument('--verbose',action='store_true',dest='verbose')

args=parser.parse_args()
inputvol_list=flatarglist(args.inputvol)
movfile_list=flatarglist(args.mpfile)
movfile_type=args.mptype.lower()
outbase_list=flatarglist(args.outbase)
outlierfile_list=flatarglist(args.outlierfile)
confoundfile_list=flatarglist(args.confoundfile)
roilist=flatarglist(args.roilist)
inputpattern_list=flatarglist(args.inputpattern)
skipvols=args.skipvols
bpfmode=args.filterstrategy
connmeasure=flatarglist(args.connmeasure)
outputformat=args.outputformat
verbose=args.verbose
do_gsr=args.gsr
do_nocompcor=args.nocompcor
do_nomotion=args.nomotion
tr=args.tr
do_savets=args.savets
do_seqroi=args.sequentialroi
sequential_roi_error_size=args.sequentialroierrorsize
do_concat=args.concat
input_shrinkage=args.shrinkage

if not connmeasure:
    connmeasure=['correlation']

connmeasure=["partial correlation" if x=="partialcorrelation" else x for x in connmeasure]
connmeasure=["partial correlation" if x=="partial" else x for x in connmeasure]
connmeasure=list(set(connmeasure))
connmeasure.sort() #note: list(set(...)) scrambles the order

if 'none' in connmeasure:
    connmeasure=['none']

connmeasure_shortname={"correlation":"corr", "partial correlation":"pcorr", "precision": "prec", "tangent":"tan", "covariance":"cov"}

########
is_pattern = len(inputpattern_list)>0 and len(roilist)>0

if is_pattern:
    input_list=inputpattern_list
    roiname_list=[]
    for roi in roilist:
        if not roi:
            continue
        roiname=roi.split("=")[0]
        roiname_list+=[roiname]
else:
    if len(roilist)>1:
        print("Multiple ROI names can only be entered when using --inputpattern")
        sys.exit(1)
    
    input_list=inputvol_list
    if roilist:
        roiname=roilist[0].split("=")[0]
    else:
        roiname=""
    roiname_list=[roiname]

num_inputs=len(input_list)

###########

if do_concat and len(outbase_list)!=1:
    print("Only 1 outputbase should be provided when using --concat")
    sys.exit(1)
elif not do_concat and len(outbase_list)!=len(input_list):
    print("Must have 1 outputbase entry for each input entry")
    sys.exit(1)

if input_shrinkage.lower() == "lw":
    pass
elif input_shrinkage.isnumeric():
    pass
else:
    print("Unknown value for shrinkage: %s" % (input_shrinkage))
    sys.exit(1)

###########

hrf_orig=None
hrf_orig_tr=None
if args.hrffile:
    #since nipy isn't working with numpy 1.18
    hrf_orig=np.loadtxt(hrffile)[:,None]
else:
    #nipy doesn't work with certain numpy versions, so let's just save it out and interpolate 
    #import nipy.modalities.fmri.hrf
    #hrf=nipy.modalities.fmri.hrf.spmt(np.arange(numvols)*tr)[:,None]
    #np.savetxt("hrf_%d.txt" % (numvols),hrf,fmt="%.18f");

    #this was generated from tr=0.8sec
    hrf_orig = np.array([0,0.00147351,0.0211715,0.0722364,0.136776,0.18755,0.209678,0.20356,0.178095,0.143632,0.10812,0.0761595,0.04961,
        0.0286445,0.0126525,0.000811689,-0.00764106,-0.0133351,-0.0167838,-0.0184269,-0.0186623,-0.0178584,-0.0163506,-0.0144316,
        -0.0123414,-0.0102627,-0.00832181,-0.00659499,-0.00511756,-0.00389459,-0.0029108,-0.00213918,-0.00154751,-0.00110304,
        -0.000775325,-0.000537817,-0.000368396,-0.000249311,-0.000166749,-0.000110239,-7.2023e-05,-4.64699e-05,-2.95653e-05,
        -1.84943e-05,-1.13126e-05,-6.69581e-06,-3.75333e-06,-1.89321e-06,-7.26446e-07,0])
    hrf_orig_tr=0.8
        
#tr=0.8
#bpf=[0.008, 0.09]
#bpf=[0.008, None]
#bpf=[None,None]
bpf=[-np.inf,np.inf]
if args.lowfreq:
    bpf[0]=args.lowfreq
if args.highfreq:
    bpf[1]=args.highfreq

if bpf[0]<=0 and not np.isfinite(bpf[1]):
    bpfmode='none'
if bpfmode=='none':
    bpf=[-np.inf,np.inf]

do_filter_rolloff=True

print("Input time series: %s" % (inputvol_list))
print("Input file pattern: %s" % (inputpattern_list))
print("ROI list: %s" % (roilist))
print("Confound file: %s" % (confoundfile_list))
print("Motion parameter file (%s-style): %s" % (movfile_type,movfile_list))
print("Outlier timepoint file: %s" % (outlierfile_list))
print("Ignore first N volumes: %s" % (skipvols))
print("Filter strategy: %s" % (bpfmode))
print("Filter band-pass Hz: [%s,%s]" % (bpf[0],bpf[1]))
print("Output basename: %s" % (outbase_list))
print("Skip compcor (WM+CSF): %s" % (do_nocompcor))
print("Skip motion regressors: %s" % (do_nomotion))
print("Global signal regression: %s" % (do_gsr))
print("Save denoised time series: %s" % (do_savets))
print("Connectivity measures: ", connmeasure)
print("Sequential ROI indexing: %s" % (do_seqroi))
print("Concatenate time series: %s" % (do_concat))
print("Covariance shrinkage: %s" % (input_shrinkage))

def addderiv(x):
    #xd=np.vstack([np.zeros([1,x.shape[1]]),np.diff(x,axis=0)])
    #add zeros AFTER deriv for consistency with CONN
    xd=np.vstack([np.diff(x,axis=0),np.zeros([1,x.shape[1]])])
    return np.hstack([x,xd])

def addsquare(x):
    return np.hstack([x,np.square(x)])
    
def addderiv_txt(s):
    if s is None:
        return None
    if isinstance(s,str):
        s=[s]
    return s+["deriv."+x for x in s]

def addsquare_txt(s):
    if s is None:
        return None
    if isinstance(s,str):
        s=[s]
    return s+["sqr."+x for x in s]
    
def normalize(x):
    xc=x-np.mean(x,axis=0)
    xdenom=np.sqrt(np.mean(xc**2,axis=0))
    xdenom[xdenom==0]=1
    return xc/xdenom
    
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
        
def convshift(g,x):
    #for convolving BPF rectangle with a smoothing function (eg: gaussian)
    return np.convolve(g/np.sum(g),np.hstack([x[::-1],x,x[::-1]]),'full')[x.shape[0]+int(g.shape[0]/2):][:x.shape[0]]
    
def naninterp(x,outliermat=None):
    #linearly interpolate segments of data with nans (to allow fftfilt)
    notnan=~np.any(np.isnan(x),axis=1)
    if outliermat is not None:
        notnan[np.sum(np.abs(outliermat),axis=1)>0,:]=False
    notnanidx=np.where(notnan)[0]
    return scipy.interpolate.interp1d(notnanidx,x[notnanidx,:],axis=0)(np.arange(x.shape[0]))
    
def fftfilt(x,tr,filt,filter_edge_rolloff=None):
    fy=np.fft.fft(np.vstack([x,x[::-1,:]]),axis=0)
    f=np.arange(fy.shape[0])
    f=np.minimum(f,fy.shape[0]-f)/(tr*fy.shape[0])
    stopmask=(f<filt[0])|(f>=filt[1])
    if filter_edge_rolloff is None:
        fy[stopmask,:]=0
    else:
        passwin=convshift(filter_edge_rolloff,~stopmask)
        fy *= np.atleast_2d(passwin).T
    y=np.real(np.fft.ifft(fy,axis=0)[:x.shape[0],:])
    return y

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
    
def dctfilt(S,tr,filt,filter_edge_rolloff=None,outliermat=None):
    N=S.shape[0]
    f=np.arange(N)/(2*tr*N)
    passmask=(f>=filt[0])&(f<filt[1])

    if filter_edge_rolloff is not None:
        passmask=convshift(filter_edge_rolloff,passmask)

    #build DCT-II basis set
    n=np.atleast_2d(np.arange(N))
    X=np.zeros((N,N))
    for k in range(N):
        X[k,:]=np.cos((np.pi/N)*(n+.5)*k)
    X[0,:]/=np.sqrt(2)
    X*=np.sqrt(2/N)

    #Xpass[~passmask]=0
    Xpass=X*np.atleast_2d(passmask).T
    
    notnan=~np.any(np.isnan(S),axis=1)
    if outliermat is not None:
        if outliermat.ndim > 1:
            notnan[np.sum(np.abs(outliermat),axis=1)>0]=False
        else:
            notnan[outliermat!=0]=False
    print("filter outliers: %d" % (np.sum(~notnan)))

    #this way will leave nans where they were and only replace the non-nans
    #Sfilt=np.nan*np.ones(S.shape)
    #Sfilt=np.zeros(S.shape)
    #Sfilt[notnan,:] = Xpass[:,notnan].T @ (X[:,notnan] @ S[notnan,:])
    
    #this option automatically interpolates nan values AFTER DCT of non-nans
    Sfilt = Xpass.T @ (X[:,notnan] @ S[notnan,:])

    return Sfilt


def loadinput(filename):
    tr=None
    volinfo=None
    roivals=None
    roisizes=None
    if filename.lower().endswith(".mat"):
        M=loadmat(filename)
        Dt=M['ts']
        if 'repetition_time' in M:
            tr=M['repetition_time'][0]
        roivals=M['roi_labels'][0]
        roisizes=M['roi_sizes'][0]
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
    else:
        raise Exception("Unknown input data file type: %s" % (filename))
    return Dt,roivals,roisizes,tr,volinfo
    
def save_timeseries(filename_noext,outputformat,output_dict, output_volume_info=None):
    outfilename=""
    shapestring=""
    if output_volume_info is not None:
        Vimg_orig=output_volume_info['image']
        outshape=list(Vimg_orig.shape[:3])
        if output_dict["ts"].ndim > 1:
            outshape+=[output_dict["ts"].shape[0]]
        Vnew=np.zeros(outshape,dtype=Vimg_orig.get_data_dtype())
        Vnew[output_volume_info['mask']]=output_dict["ts"].T
        Vimg=nib.Nifti1Image(Vnew.astype(Vimg_orig.get_data_dtype()),affine=Vimg_orig.affine,header=Vimg_orig.header)
        outfilename=filename_noext+"."+output_volume_info["extension"]
        shapestring="x".join([str(x) for x in Vimg.shape])
        nib.save(Vimg, outfilename)
    else:
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

def save_connmatrix(filename_noext,outputformat,output_dict):
    outfilename=""
    shapestring="%dx%d" % (output_dict["C"].shape[0],output_dict["C"].shape[1])
    if outputformat == "mat":
        outfilename=filename_noext+"."+outputformat
        savemat(outfilename,output_dict,format='5',do_compression=True)
    else:
        headertxt="ROI_Labels:\n"
        headertxt+=" ".join(["%d" % (x) for x in output_dict["roi_labels"]])
        headertxt+="\nROI_Sizes(voxels):\n"
        headertxt+=" ".join(["%d" % (x) for x in output_dict["roi_sizes"]])
        headertxt+="\nCovariance_estimator: %s" % (output_dict["cov_estimator"])
        headertxt+="\nCovariance_shrinkage: %s" % (output_dict["cov_shrinkage"])
        outfilename=filename_noext+"."+outputformat
        np.savetxt(outfilename,output_dict["C"],fmt="%.18f",header=headertxt,comments="# ")
    return outfilename, shapestring
            
def compute_connmatrix(ts,conntype,input_shrinkage="lw"):
    if input_shrinkage.lower() == "lw":
        covest=sklearn.covariance.LedoitWolf()
    elif input_shrinkage.isnumeric():
        input_shrinkage=float(input_shrinkage)
        if input_shrinkage == 0:
            covest=sklearn.covariance.EmpiricalCovariance()
        else:
            covest=sklearn.covariance.ShrunkCovariance(shrinkage=input_shrinkage)
        
    #Note on ConnectivityMeasure:
    # * for "correlation", nilearn fits cov_estimator(standardize(timeseries)), so shrinkage is applied AFTER normalization
    # * denoised data are already zscored going in, so correlation and covariance should be nearly identical
    #covest=sklearn.covariance.LedoitWolf()
    #covest=sklearn.covariance.EmpiricalCovariance()
    #covest=sklearn.covariance.ShrunkCovariance(shrinkage=)    
    E=nilearn.connectome.ConnectivityMeasure(kind=conntype, vectorize=False, discard_diagonal=False, cov_estimator=covest)
    #C=E.fit_transform([Dt_clean[skipvols:,:]])[0]
    C=E.fit_transform([ts])[0]

    shrinkage=np.nan
    covest_class=E.cov_estimator.__class__.__name__
    if covest_class == "LedoitWolf":
        shrinkage=E.cov_estimator_.shrinkage_
    elif covest_class == "ShrunkCovariance":
        shrinkage=E.cov_estimator_.shrinkage
    elif covest_class == "EmpiricalCovariance":
        shrinkage=0

    return C, shrinkage, covest_class
    
#########################################################

#############
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
        
#read in --motionparam inputs if provided, overwriting values from --confoundfile
if len(movfile_list)==num_inputs:
    for inputidx,movfile in enumerate(movfile_list):
        mp=np.loadtxt(movfile)

        if movfile_type=="spm":
            print("Motion file %s is (%d,%d), SPM-style=(xmm,ymm,zmm,radx,rady,radz)" % (movfile,mp.shape[0],mp.shape[1]))
            #already xmm,ymm,zmm,radx,rady,radz
            mp=mp[:,:6]

        elif movfile_type=="hcp":
            print("Motion file %s is (%d,%d), HCP-style=(xmm,ymm,zmm,degx,degy,degz)" % (movfile,mp.shape[0],mp.shape[1]))
            #convert from xmm,ymm,zmm,degx,degy,degz to rad
            mp=mp[:,:6]
            mp[:,3:6]=mp[:,3:6]*np.pi/180
        elif movfile_type=="fsl":
            print("Motion file %s is (%d,%d), FSL-style=(radx,rady,radz,xmm,ymm,zmm)" % (movfile,mp.shape[0],mp.shape[1]))
            #swap mm and rad columns
            mp=mp[:,[3,4,5,0,1,2]]
        else:
            raise Exception("Unknown motion parameter file type: %s" % (movfile_type))
        confounds_list[inputidx]["mp"]=mp
    
#read in --outlierfile inputs if provided, overwriting values from --confoundfile
if len(outlierfile_list)==num_inputs:
    for inputidx,outlierfile in enumerate(outlierfile_list):
        outliermat=np.loadtxt(outlierfile)>0
        confounds_list[inputidx]["outliermat"]

##############
# main loop
for roiname in roiname_list:
    outlier_free_data_list=[]
    for inputidx,inputitem in enumerate(input_list):
        confounds_dict=confounds_list[inputidx]
        
        if is_pattern:
            inputfile=inputitem % (roiname)
        else:
            inputfile=inputitem
        Dt,roivals,roisizes,tr_input,vol_info = loadinput(inputfile)
        print("Loaded input file: %s (%dx%d)" % (inputfile,Dt.shape[0],Dt.shape[1]))
        if tr_input:
            tr=tr_input

        numvols=Dt.shape[0]
    
        did_print_nuisance_size=False

        outliermat=np.zeros((numvols,1))
        resteffect=np.zeros((numvols,0))
        gmreg=np.zeros((numvols,0))
        wmreg=np.zeros((numvols,0))
        csfreg=np.zeros((numvols,0))
        mp=np.zeros((numvols,0))
        
        if confounds_dict["gmreg"] is not None:
            gmreg=confounds_dict["gmreg"]
        if confounds_dict["wmreg"] is not None:
            wmreg=confounds_dict["wmreg"]
        if confounds_dict["csfreg"] is not None:
            csfreg=confounds_dict["csfreg"]
        if confounds_dict["mp"] is not None:
            mp=confounds_dict["mp"]
        if confounds_dict["outliermat"] is not None:
            outliermat=confounds_dict["outliermat"]
        
        if resteffect.shape[-1]==0:
            if hrf_orig_tr:
                hrf_interp=scipy.interpolate.interp1d(hrf_orig_tr*np.arange(len(hrf_orig)),hrf_orig,axis=0,kind="cubic",fill_value=0,bounds_error=False)
                hrf=hrf_interp(np.arange(numvols)*tr)[:,None]
            if hrf.shape[0] < numvols:
                hrf=np.vstack(hrf,np.zeros((numvols-hrf.shape[0],1)))
            elif hrf.shape[0] > numvols:
                hrf=hrf[:numvols,:]
            resteffect=np.convolve(np.ones(numvols),hrf[:,0])[:numvols,None]

        #might be 1d format, so expand it then collapse, mark skipvols, then re-expand
        outliermat=np.sum(vec2columns(outliermat)!=0,axis=1)[:,None]
        outliermat[:skipvols,:]=True
        outliermat=vec2columns(outliermat)
    
        outlierflat=np.sum(outliermat!=0,axis=1)
    
        onesmat=np.ones(mp.shape[0])[:,None]
        detrendmat=np.arange(mp.shape[0])[:,None]

        ########################################
        if not do_gsr:
            gmreg=np.zeros((numvols,0))
    
        if do_nocompcor:
            wmreg=np.zeros((numvols,0))
            csfreg=np.zeros((numvols,0))
        
        if do_nomotion:
            mp=np.zeros((numvols,0))
        
        confounds=np.hstack([onesmat,addderiv(gmreg),wmreg,csfreg,addsquare(addderiv(mp)),addderiv(resteffect),outliermat,detrendmat])
        confounds_to_filter=np.hstack([addderiv(gmreg),wmreg,csfreg,addsquare(addderiv(mp)),addderiv(resteffect)])
        
        confounds_orig=confounds;
        
        if do_filter_rolloff:
            filter_edge_rolloff_size=int(36/tr/2)*2+1 #51 for tr=0.72
            filter_edge_rolloff_std=3.6/tr #5 for tr=0.72
            filter_edge_rolloff=scipy.signal.gaussian(filter_edge_rolloff_size,filter_edge_rolloff_std)
        else:
            filter_edge_rolloff=None

        
        if not did_print_nuisance_size:
            print("Total nuisance regressors: %d" %  (confounds.shape[1]))
            did_print_nuisance_size=True
            
        
        if bpfmode=="parallel":
            #nilearn filters confounds, filters signals, and then denoises filtered(signals) with filtered(confounds)
            #but how does this handle outlier regressors? filtering is going to blur those out in weird ways
            #raise Exception("seq filtering hasn't been tested AT ALL yet.")
            Dt=dctfilt(Dt,tr,bpf,filter_edge_rolloff,outliermat=outlierflat)
            confounds=dctfilt(confounds_to_filter,tr,bpf,filter_edge_rolloff,outliermat=outlierflat)
        
            #remove confound time series that are all zeros after filtering
            confounds=confounds[:,np.max(abs(confounds),axis=0)>(2*np.finfo(confounds.dtype).eps)]
            confounds=np.hstack([onesmat,confounds,outliermat,detrendmat])
            #savemat(outbase_list[inputidx]+"_confounds_filtered.mat",{"confounds_filtered":confounds,"confounds_orig":confounds_orig})
            print("Total nuisance regressors after %s filter: %d" % (bpfmode,confounds.shape[1]))
        
        if bpfmode=="orth":
            Dt_clean=nilearn.signal.clean(Dt, confounds=confounds, standardize=False, t_r=tr, detrend=False,low_pass=bpf[1], high_pass=bpf[0])
        else:
            Dt_clean=nilearn.signal.clean(Dt, confounds=confounds, standardize=False, t_r=tr, detrend=False)

        if bpfmode=="connregbp":
            #Dt=nilearn.signal.clean(Dt.copy(), detrend=False, standardize=False, low_pass=bpf[1], high_pass=bpf[0], t_r=tr)
            #what should we do about outliers when filtering?
            #the outlier regressors just set those timepoints to 0
            #option 1: just filter it as-is with outliers set to 0
            #option 2: interpolate outlier segments
            #option 3: use dctfilt with projection that ignores outlier segments
            #
            #Important: We do get some ringing at edges of outlier segments
            #   * option 3 is always better than option 2 for ringing
            #   * If we do this, should we do some kind of post-filtering global signal regression to minimize global ringing?

            #confounds_clean=dctfilt(confounds,tr,bpf)
            #savemat(outbase+"_testconfounds_clean.mat",{"confounds":confounds,"confounds_clean":confounds_clean})
        
            Dt_clean=dctfilt(Dt_clean, tr, bpf,filter_edge_rolloff,outliermat=outlierflat) #, scipy.signal.gaussian(21,5))
            #Dt_clean=fftfilt(naninterp(Dt_clean,outliermat=outlierflat), tr, bpf, scipy.signal.gaussian(21,5))
            #Dt_clean=fftfilt(Dt_clean, tr, bpf)

        if do_seqroi:
            #make full 
            maxroi=np.max(roivals).astype(np.int)
            if len(roivals) < sequential_roi_error_size and maxroi > sequential_roi_error_size:
                raise Exception("Maximum ROI label (%d) exceeded allowable size (%d), suggesting a mistake. If this was intentional, set --sequentialerrorsize" % (maxroi,sequential_roi_error_size))
            roivals_seq=np.arange(1,maxroi+1)
            roisizes_seq=np.zeros(maxroi)
            roisizes_seq[roivals.astype(np.int)-1]=roisizes
        
            Dt_clean_seq=np.zeros((Dt_clean.shape[0],maxroi),dtype=Dt_clean.dtype)
            Dt_clean_seq[:,roivals.astype(np.int)-1]=Dt_clean
        
            roivals=roivals_seq
            roisizes=roisizes_seq
            Dt_clean=Dt_clean_seq.copy()
        else:
            pass

        if do_gsr:
            gsrsuffix="_gsr"
        else:
            gsrsuffix=""

        if roiname:
            roisuffix="_"+roiname
        else:
            roisuffix=""
    
        if do_savets and len(outbase_list)==num_inputs:
            savedfilename, shapestring = save_timeseries(outbase_list[inputidx]+roisuffix+gsrsuffix+"_tsclean", outputformat, {"ts":Dt_clean,"roi_labels":roivals, "roi_sizes":roisizes,"repetition_time":tr}, vol_info)
            print("Saved %s (%s)" % (savedfilename,shapestring))
    
        #note: skipvols is already included in outlierflat
        Dt_clean_outlierfree=normalize(Dt_clean[outlierflat==0,:])
        
        if do_concat:
            outlier_free_data_list+=[Dt_clean_outlierfree]
        else:
            for cm in connmeasure:
                if cm == 'none':
                    continue
                if vol_info is not None:
                    print("Connectivity matrices for voxelwise input is currently disabled!")
                    continue
                C,shrinkage,covest_class = compute_connmatrix(Dt_clean_outlierfree, cm, input_shrinkage)
        
                Cdict={"C":C,"roi_labels":roivals,"roi_sizes":roisizes,"shrinkage":shrinkage,'cov_estimator':covest_class}
                savedfilename, shapestring = save_connmatrix(outbase_list[inputidx]+roisuffix+gsrsuffix+"_FC%s" % (connmeasure_shortname[cm]),outputformat,Cdict)
                print("Saved %s (%s)" % (savedfilename,shapestring))

    if not do_concat:
        continue
    
    if do_gsr:
        gsrsuffix="_gsr"
    else:
        gsrsuffix=""

    if roiname:
        roisuffix="_"+roiname
    else:
        roisuffix=""
        
    #concatenate multiple scans 
    Dt_clean_outlierfree=np.vstack(outlier_free_data_list)
    
    for cm in connmeasure:
        if cm == 'none':
            continue
        if vol_info is not None:
            print("Connectivity matrices for voxelwise input is currently disabled!")
            continue
        C,shrinkage,covest_class = compute_connmatrix(Dt_clean_outlierfree, cm, input_shrinkage)
        
        Cdict={"C":C,"roi_labels":roivals,"roi_sizes":roisizes,"shrinkage":shrinkage,'cov_estimator':covest_class}
        savedfilename, shapestring = save_connmatrix(outbase_list[0]+roisuffix+gsrsuffix+"_FC%s" % (connmeasure_shortname[cm]),outputformat,Cdict)
        print("Saved %s (%s)" % (savedfilename,shapestring))
######################################
######################################
######################################
