import numpy as np
import nibabel as nib
from scipy.io import loadmat,savemat
import scipy.signal, scipy.interpolate
import os.path

def flatlist(l):
    if l is None:
        return []
    return [x for y in l for x in y]
    
def flatarglist(l):
    if l is None:
        return []
    return flatlist([x.split(",") for x in flatlist(l)])
    

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

def filename_split_extension(filepath):
    filedir,filename=os.path.split(filepath)
    if "." in filename:
        if filename.lower().endswith(".gz"):
            extension=".".join(filename.split(".")[-2:])
        else:
            extension=filename.split(".")[-1]
        filebase=os.path.join(filedir,filename[:-len(extension)-1])
    else:
        extension=""
        filebase=filepath
    return filebase,extension
    
def load_input(filename):
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
        eps=2*np.finfo(V.dtype).eps #mask by eps instead of 0
        M=np.any(np.abs(V)>eps,axis=3)
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
    outputformat_split=None
    if outputformat is None:
        filename_noext,outputformat=filename_split_extension(filename_noext)
        outputformat_split=outputformat
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
        if outputformat_split:
            ext=outputformat_split
        else:
            ext=output_volume_info["extension"]
        outfilename=filename_noext+"."+ext
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
            if "is_outlier" in output_dict:
                headertxt+="\nOutlier_volumes:\n"
                headertxt+=" ".join(["%d" % (x) for x in output_dict["is_outlier"]])
            outfilename=filename_noext+"."+outputformat
            np.savetxt(outfilename,output_dict["ts"],fmt="%.18f",header=headertxt,comments="# ")
    return outfilename, shapestring
    
    

def prepadZero(x,n):
    return np.vstack([np.zeros([n,x.shape[1]]),x])
    
############################
def params2matrix(P):
    #adapted from spm's spm_matrix()
    T  =   np.matrix([
            [1,   0,   0,   P[0]],
            [0,   1,   0,   P[1]],
            [0,   0,   1,   P[2]],
            [0,   0,   0,   1]])

    R1  =  np.matrix([
            [1,   0,              0,              0],
            [0,   np.cos(P[3]),   np.sin(P[3]),   0],
            [0,  -np.sin(P[3]),   np.cos(P[3]),   0],
            [0,   0,              0,              1]])

    R2  =  np.matrix([
            [np.cos(P[4]),   0,   np.sin(P[4]),   0],
            [0,              1,   0,              0],
            [-np.sin(P[4]),  0,   np.cos(P[4]),   0],
            [0,              0,   0,              1]])

    R3  =  np.matrix([
            [np.cos(P[5]),   np.sin(P[5]),   0,   0],
            [-np.sin(P[5]),  np.cos(P[5]),   0,   0],
            [0,              0,              1,   0],
            [0,              0,              0,   1]])
            
    R   = R1*R2*R3;

    return T * R