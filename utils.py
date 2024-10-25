import numpy as np
import nibabel as nib
from scipy.io import loadmat,savemat
import scipy.signal, scipy.interpolate
import sys
import os.path
import pandas as pd

from nilearn import __version__ as nilearn_version
from scipy import __version__ as scipy_version
from numpy import __version__ as numpy_version
from nibabel import __version__ as nibabel_version

from _version import __version__, __version_date__

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

def normalize(x,axis=0,denomfun='mean'):
    xc=x-np.mean(x,axis=axis)
    if denomfun=='mean':
        xdenom=np.sqrt(np.mean(xc**2,axis=axis))
    elif denomfun=='sum':
        xdenom=np.sqrt(np.sum(xc**2,axis=axis))
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
        notnan[np.sum(np.abs(outliermat),axis=1)>0]=False
    notnanidx=np.where(notnan)[0]
    return scipy.interpolate.interp1d(notnanidx,x[notnanidx,:],axis=0,bounds_error=False,fill_value=0)(np.arange(x.shape[0]))
    
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
    X=np.zeros((N,N),dtype=complex)
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

def filename_split_extension(filepath, is_cifti=False):
    filedir,filename=os.path.split(filepath)
    if "." in filename:
        if is_cifti:
            dotparts=filename.split(".")
            dotparts_nii=[i for i,x in enumerate(dotparts) if x.lower()=="nii"]
            if len(dotparts_nii)==0:
                extension=""
            else:
                extension=".".join(dotparts[dotparts_nii[-1]-1:])
        else:
            if filename.lower().endswith(".gz"):
                extension=".".join(filename.split(".")[-2:])
            else:
                extension=filename.split(".")[-1]
        if extension:
            filebase=os.path.join(filedir,filename[:-len(extension)-1])
        else:
            filebase=filepath
    else:
        extension=""
        filebase=filepath
    return filebase,extension

def get_first_volume_3D(voldata):
    if voldata.ndim == 4:
        return voldata[:,:,:,0]
    else:
        return voldata

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
        
        Vimg=nib.load(filename)
        is_cifti=type(Vimg).__name__.lower().find("cifti")>=0
        _, volext = filename_split_extension(filename,is_cifti=is_cifti)
        
        V=Vimg.get_fdata()
        eps=2*np.finfo(V.dtype).eps #mask by eps instead of 0
        
        if is_cifti:
            ax_idx=Vimg.header.mapped_indices
            ax_names=[type(Vimg.header.get_axis(ax)).__name__ for ax in ax_idx]
            time_axis=[ax for i,ax in enumerate(ax_idx) if ax_names[i].lower().find("seriesaxis")>=0]
            non_brain_axis=[ax for i,ax in enumerate(ax_idx) if ax_names[i].lower().find("brainmodelaxis")<0 and ax_names[i].lower().find("parcelsaxis")<0]
            if len(time_axis)>0:
                time_axis=time_axis[-1]
            elif len(non_brain_axis)>0:
                time_axis=non_brain_axis[-1]
            else:
                Exception("No time axis found")
            
            if time_axis < 0 or time_axis > 1:
                raise Exception("Unknown time axis: %d. Should be 0 or 1" % (time_axis))
            
            M=np.any(np.abs(V)>eps,axis=time_axis)
            if time_axis == 0:
                Dt=V[:,M>0]
            elif time_axis == 1:
                Dt=V[M>0,:].T
            
            try:
                tr=Vimg.header.get_axis(time_axis).step
            except:
                #for non SeriesAxis (eg: stacked dlabel axes), TR does not apply
                tr=0
        else:
            #read normal nifti
            if Vimg.ndim > 3:
                M=np.any(np.abs(V)>eps,axis=3)
                Dt=V[M>0].T
                tr=Vimg.header['pixdim'][4]
                time_axis=3
            else:
                M=np.abs(V)>eps
                Dt=V[M>0].T
                tr=Vimg.header['pixdim'][4]
                time_axis=3
        
        volinfo={'image':Vimg, 'shape':Vimg.shape, 'mask':M, "extension":volext, "is_cifti":is_cifti,"time_axis":time_axis}
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
    filename_noext_input=filename_noext
    outputformat_split=None
    if outputformat is None:
        filename_noext,outputformat=filename_split_extension(filename_noext)
        outputformat_split=outputformat
        
    outfilename=""
    shapestring=""
    if output_volume_info is not None:
        if output_volume_info['is_cifti']:
            Vimg_orig=output_volume_info['image']
            outshape=list(Vimg_orig.shape)
            if output_dict["ts"].ndim > 1:
                outshape[output_volume_info['time_axis']]=output_dict["ts"].shape[0]
            else:
                output_dict["ts"]=np.atleast_2d(output_dict["ts"])
                outshape[output_volume_info['time_axis']]=output_dict["ts"].shape[0]
            #output_dtype=Vimg_orig.get_data_dtype()
            output_dtype=np.float32
            Vnew=np.zeros(outshape,dtype=output_dtype)
            if output_volume_info['time_axis']==0:
                Vnew[:,output_volume_info['mask']]=output_dict["ts"]
            else:
                Vnew[output_volume_info['mask'],:]=output_dict["ts"].T
            new_header=Vimg_orig.header
            
            time_axis_type=type(new_header.get_axis(output_volume_info['time_axis'])).__name__
            if time_axis_type.lower().find("seriesaxis")>=0:
                axlist=[output_volume_info['image'].header.get_axis(i) for i in output_volume_info['image'].header.mapped_indices]
                axlist[output_volume_info['time_axis']].size=output_dict["ts"].shape[0]
                new_header=nib.cifti2.cifti2.Cifti2Header.from_axes(axlist)
                
            elif time_axis_type.lower().find("scalaraxis")>=0:
                axlist=[output_volume_info['image'].header.get_axis(i) for i in output_volume_info['image'].header.mapped_indices]
                if output_dict["ts"].shape[0] != output_volume_info['shape'][output_volume_info['time_axis']]:
                    namelist=["map%04d" % (x) for x in range(output_dict["ts"].shape[0])]
                    axlist[output_volume_info['time_axis']]=nib.cifti2.cifti2_axes.ScalarAxis(name=namelist)
                    new_header=nib.cifti2.cifti2.Cifti2Header.from_axes(axlist)
                
            Vimg=nib.cifti2.cifti2.Cifti2Image(Vnew.astype(output_dtype),header=new_header)
        else:
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
            #redo filename split now that we know. whether "is_cifti" is available
            filename_noext,outputformat=filename_split_extension(filename_noext_input,is_cifti=output_volume_info['is_cifti'])
            outputformat_split=outputformat
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

def read_motion_params(movfile, movfile_type):
    #read in motion parameters (HCP saved mmx,mmy,mmz, degx,degy,degz)
    if not movfile:
        raise Exception("Motion parameter file not specified")
    
    if movfile_type=="spm":
        mp=np.loadtxt(movfile)
        print("Motion file %s is (%d,%d), SPM-style=(xmm,ymm,zmm,radx,rady,radz)" % (movfile,mp.shape[0],mp.shape[1]))
        #already xmm,ymm,zmm,radx,rady,radz
        mp=mp[:,:6]

    elif movfile_type=="hcp":
        mp=np.loadtxt(movfile)
        print("Motion file %s is (%d,%d), HCP-style=(xmm,ymm,zmm,degx,degy,degz)" % (movfile,mp.shape[0],mp.shape[1]))
        #convert from xmm,ymm,zmm,degx,degy,degz to rad
        mp=mp[:,:6]
        mp[:,3:6]=mp[:,3:6]*np.pi/180
    elif movfile_type=="fsl":
        mp=np.loadtxt(movfile)
        print("Motion file %s is (%d,%d), FSL-style=(radx,rady,radz,xmm,ymm,zmm)" % (movfile,mp.shape[0],mp.shape[1]))
        #swap mm and rad columns
        mp=mp[:,[3,4,5,0,1,2]]
    elif movfile_type=="fmriprep":
        mp_dataframe=pd.read_table(movfile)
        print("Motion file %s is (%d,%d), fMRIPrep-style=(trans_[xyz],rot_[xyz])=(xmm,ymm,zmm,radx,rady,radz)" % (movfile,mp_dataframe.shape[0],mp_dataframe.shape[1]))
        mp=np.stack([
            mp_dataframe['trans_x'],
            mp_dataframe['trans_y'],
            mp_dataframe['trans_z'], 
            mp_dataframe['rot_x'], 
            mp_dataframe['rot_y'],
            mp_dataframe['rot_z']
            ],axis=-1)
    else:
        raise Exception("Unknown motion parameter file type: %s" % (movfile_type))
    
    #already xmm,ymm,zmm,radx,rady,radz
    mp_names=["motion.xmm","motion.ymm","motion.zmm","motion.xrad","motion.yrad","motion.zrad"]
    
    return mp, mp_names



def save_connmatrix(filename_noext,outputformat,output_dict):
    outfilename=""
    shapestring="%dx%d" % (output_dict["C"].shape[0],output_dict["C"].shape[1])
    if outputformat.startswith("."):
        outputformat=outputformat[1:]
    if outputformat.lower().endswith("mat"):
        outfilename=filename_noext+"."+outputformat
        savemat(outfilename,output_dict,format='5',do_compression=True)
    else:
        headertxt="ROI_Labels:\n"
        headertxt+=" ".join(["%d" % (x) for x in output_dict["roi_labels"]])
        headertxt+="\nROI_Sizes(voxels):\n"
        headertxt+=" ".join(["%d" % (x) for x in output_dict["roi_sizes"]])
        headertxt+="\nCovariance_estimator: %s" % (output_dict["cov_estimator"])
        headertxt+="\nCovariance_shrinkage: %s" % (output_dict["shrinkage"])
        outfilename=filename_noext+"."+outputformat
        np.savetxt(outfilename,output_dict["C"],fmt="%.18f",header=headertxt,comments="# ")
    return outfilename, shapestring
            
def load_connmatrix(filename):
    conn_dict={}
    
    if filename.lower().endswith(".mat"):
        M=loadmat(filename,simplify_cells=True)
        mfield_data_search=['C','FC','data']
        mfield_data_search=[x.upper() for x in mfield_data_search]
        for m in M:
            if m.upper() in mfield_data_search:
                conn_dict['C']=M[m]
                break
        mfield_search=['roi_labels','roi_sizes','cov_estimator','shrinkage']
        for m in mfield_search:
            if m in M:
                conn_dict[m]=M[m]
    else:
        sep=None
        if filename.lower().endswith(".csv"):
            sep=","
        with open(filename,'r') as fid:
            commentlines=[s.strip() for s in fid.readlines() if s.startswith("#")]
        for i,l in enumerate(commentlines):
            if l.upper().startswith("# ROI_LABELS"):
                labelstr=commentlines[i+1].replace("#","").split(sep)
                conn_dict['roi_labels']=[int(x) for x in labelstr]
            if l.upper().startswith("# ROI_SIZES"):
                sizestr=commentlines[i+1].replace("#","").split(sep)
                conn_dict['roi_sizes']=[float(x) for x in sizestr]
            if l.upper().startswith("# COVARIANCE_ESTIMATOR"):
                conn_dict['cov_estimator']=l.split(":")[-1].strip()
            if l.upper().startswith("# COVARIANCE_SHRINKAGE"):
                conn_dict['shrinkage']=float(l.split(":")[-1].strip())
            
        conn_dict['C']=np.loadtxt(filename,comments='#')
    
    if 'roi_labels' in conn_dict and len(conn_dict['roi_labels']) == conn_dict['C'].shape[0]:
        roiidx=np.array(conn_dict['roi_labels'],dtype=int)-1
        Cnew=np.zeros([max(roiidx)+1,max(roiidx)+1])
        d=np.median(np.diag(conn_dict['C']))
        np.fill_diagonal(Cnew,d)
        Cnew[roiidx[:,None],roiidx]=conn_dict['C']
        conn_dict['C']=Cnew
        conn_dict['roi_labels']=np.arange(1,max(roiidx)+1)
    
    return conn_dict

def get_version(include_date=False):
    """Return the version of this package"""
    if include_date:
        return __version__+" ("+__version_date__+")"
    else:
        return __version__
        
def package_version_dict(as_string=False):
    """Return a dictionary of package versions used by this package"""
    versions={
        'fmriclean': get_version(include_date=True),
        'nilearn': nilearn_version, 
        'numpy': numpy_version, 
        'scipy': scipy_version, 
        'nibabel': nibabel_version
    }
    if as_string:
        return "; ".join(["%s: %s" % (k,v) for k,v in versions.items()])
    else:
        return versions