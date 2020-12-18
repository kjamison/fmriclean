import nibabel as nib
import numpy as np
import nilearn.signal
#import nipy.modalities.fmri.hrf
import sys
import argparse
from scipy.ndimage import binary_erosion
from scipy.io import savemat
import scipy.interpolate

import os.path

parser=argparse.ArgumentParser(description='Create file with confound regressors for denoising')

parser.add_argument('--hcpmnidir',action='store',dest='hcpmnidir',help='Optional: HCP-style MNINonLinear directory to find input files')
parser.add_argument('--hcpscanname',action='store',dest='hcpscanname',help='Optional: Scan name within MNINonLinear/Results')
parser.add_argument('--input',action='store',dest='inputvol')
#parser.add_argument('--mask',action='store',dest='maskfile')
parser.add_argument('--output',action='store',dest='outputfile')
parser.add_argument('--motionparam',action='store',dest='mpfile')
parser.add_argument('--motionparamtype',action='store',dest='mptype',choices=['spm','hcp','fsl'],default='fsl')
parser.add_argument('--gmmask',action='store',dest='gmfile',help='gray-matter mask (for global signal regression)')
parser.add_argument('--wmmask',action='store',dest='wmfile',help='white-matter mask')
parser.add_argument('--csfmask',action='store',dest='csffile',help='CSF mask')
parser.add_argument('--ewmmask',action='store',dest='ewmfile',help='white-matter mask ALREADY eroded')
parser.add_argument('--ecsfmask',action='store',dest='ecsffile',help='CSF mask ALREADY eroded')
parser.add_argument('--erosionvoxels',action='store',dest='erosionvoxels',type=int,default=1,help='Number of voxels to erode for WM/CSF')
parser.add_argument('--hrffile',action='store',dest='hrffile')
parser.add_argument('--outlierfile',action='store',dest='outlierfile')
parser.add_argument('--skipvols',action='store',dest='skipvols',type=int,default=5)
parser.add_argument('--verbose',action='store_true',dest='verbose')

args=parser.parse_args()
hcpmnidir=args.hcpmnidir
hcpscanname=args.hcpscanname
inputvol=args.inputvol
#maskfile=args.maskfile
movfile=args.mpfile
movfile_type=args.mptype.lower()
outlierfile=args.outlierfile
skipvols=args.skipvols
outputfile=args.outputfile
erosionvoxels=args.erosionvoxels
verbose=args.verbose

wmfile=None
csffile=None
gmfile=None
do_erode_wm=True
do_erode_csf=True

if hcpmnidir and hcpscanname:
    if inputvol is None:
        inputvol="%s/Results/%s/%s.nii.gz" % (hcpmnidir,hcpscanname,hcpscanname)
    movfile="%s/Results/%s/Movement_Regressors.txt" % (hcpmnidir,hcpscanname)
    movfile_type='hcp'
    #maskfile="%s/Results/%s/RibbonVolumeToSurfaceMapping/goodvoxels.nii.gz" % (hcpmnidir,hcpscanname)
    gmfile="%s/ROIs/GMReg.2.nii.gz" % (hcpmnidir)
    wmfile="%s/ROIs/WMReg.2.nii.gz" % (hcpmnidir)
    csffile="%s/ROIs/CSFReg.2.nii.gz" % (hcpmnidir)
    do_erode_wm=True
    do_erode_csf=True
    print("HCP input shortcut: %s/Results/%s" % (hcpmnidir,hcpscanname))

if args.gmfile:
    gmfile=args.gmfile
    
if args.ewmfile:
    wmfile=args.ewmfile
    do_erode_wm=False
elif args.wmfile:
    wmfile=args.wmfile

if args.ecsffile:
    csffile=args.ecsffile
    do_erode_csf=False
elif args.csffile:
    csffile=args.csffile
    
hrffile=None
if args.hrffile:
    #since nipy isn't working with numpy 1.18
    hrffile=args.hrffile


if csffile.upper() == 'NONE':
    csffile=None
if wmfile.upper() == 'NONE':
    wmfile=None
if gmfile.upper() == 'NONE':
    gmfile=None

print("Input time series: %s" % (inputvol))
print("Motion parameter file (%s-style): %s" % (movfile_type,movfile))
print("Gray-matter volume mask: %s" % (gmfile))
if do_erode_wm:
    print("White-matter volume mask: %s" % (wmfile))
else:
    print("White-matter volume mask (eroded): %s" % (wmfile))
if do_erode_csf:
    print("CSF volume mask: %s" % (csffile))
else:
    print("CSF volume mask (eroded): %s" % (csffile))
print("Erosion voxels: %d (%dx%dx%d box)",erosionvoxels,erosionvoxels*2+1,erosionvoxels*2+1,erosionvoxels*2+1)
print("Outlier timepoint file: %s" % (outlierfile))
print("Consider first N volumes to be outliers: %s" % (skipvols))
print("Output filename: %s" % (outputfile))

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
    return xc/np.sqrt(np.sum(xc**2,axis=0))
    
def getmaskcomps(dataimg,maskimg,maskconfounds,ncomp,blocksize=200):
    #maskdata=dataimg.get_fdata(dtype=np.float32,caching="unchanged")[maskimg.get_fdata()>0].T
    mask=maskimg.get_fdata()>0
    if ncomp==1:
        #when only returning the mean, we can compute this faster and with less memory by just computing mask mean as we loop
        maskmean=np.zeros((dataimg.shape[3],1),dtype=np.float32)
        for i in range(0,dataimg.shape[3],blocksize):
            blockstop=min(i+blocksize,dataimg.shape[3])
            maskmean[i:blockstop,:]=np.mean(dataimg.slicer[...,i:blockstop].get_fdata(dtype=np.float32,caching="unchanged")[mask],axis=0)[:,None]
        return maskmean
        
    else:
        maskdata=np.zeros((dataimg.shape[3],np.sum(mask)),dtype=np.float32)
        for i in range(0,dataimg.shape[3],blocksize):
            blockstop=min(i+blocksize,dataimg.shape[3])
            maskdata[i:blockstop,:]=dataimg.slicer[...,i:blockstop].get_fdata(dtype=np.float32,caching="unchanged")[mask].T    
        maskmean=np.mean(maskdata,axis=1)[:,None]
    #if ncomp==1:
    #    return maskmean
    newmaskconfounds=np.hstack([maskmean,maskconfounds])
    newmaskconfounds=newmaskconfounds-np.mean(newmaskconfounds,axis=0)
    maskdata_clean=nilearn.signal.clean(maskdata, detrend=False, standardize=False, confounds=newmaskconfounds)
    comps=nilearn.signal.high_variance_confounds(maskdata_clean,n_confounds=ncomp,percentile=100,detrend=False)
    comps=np.hstack([maskmean,comps[:,1:]])
    #comps=nilearn.signal.high_variance_confounds(maskdata,n_confounds=ncomp,percentile=100,detrend=False)
    return comps

def fftfilt(x,tr,filt):
    fy=np.fft.fft(np.vstack([x,x[::-1,:]]),axis=0)
    f=np.arange(fy.shape[0])
    f=np.minimum(f,fy.shape[0]-f)/(tr*fy.shape[0])
    stopmask=(f<filt[0])|(f>=filt[1])
    fy[stopmask,:]=0
    y=np.real(np.fft.ifft(fy,axis=0)[:x.shape[0],:])
    return y
    
def dctbasis(x,tr,filt):
    B=np.real(np.fft.ifft(np.eye(2*x.shape[0],x.shape[0]),axis=0)[:x.shape[0],:])
    f=np.arange(x.shape[0])/(2*tr*x.shape[0])
    stopmask=(f<filt[0])|(f>=filt[1])
    return B[:,stopmask]

D=nib.load(inputvol)
tr=D.header['pixdim'][4]
numvols=D.shape[-1]

#read in motion parameters (HCP saved mmx,mmy,mmz, degx,degy,degz)
mp_names=[]
if movfile:
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
        error("Unknown motion parameter file type: %s" % (movfile_type))
        
    #already xmm,ymm,zmm,radx,rady,radz
    mp_names=["motion.xmm","motion.ymm","motion.zmm","motion.xrad","motion.yrad","motion.zrad"]
else:
    mp=np.zeros((numvols,0))


if hrffile is None:
    #nipy doesn't work with certain numpy versions, so let's just save it out and interpolate 
    #import nipy.modalities.fmri.hrf
    #hrf=nipy.modalities.fmri.hrf.spmt(np.arange(numvols)*tr)[:,None]
    #np.savetxt("hrf_%d.txt" % (numvols),hrf,fmt="%.18f");
    
    #this was generated from tr=0.8sec, which we then resample to the TR of the input data
    hrf800 = np.array([0,0.00147351,0.0211715,0.0722364,0.136776,0.18755,0.209678,0.20356,0.178095,0.143632,0.10812,0.0761595,0.04961,
        0.0286445,0.0126525,0.000811689,-0.00764106,-0.0133351,-0.0167838,-0.0184269,-0.0186623,-0.0178584,-0.0163506,-0.0144316,
        -0.0123414,-0.0102627,-0.00832181,-0.00659499,-0.00511756,-0.00389459,-0.0029108,-0.00213918,-0.00154751,-0.00110304,
        -0.000775325,-0.000537817,-0.000368396,-0.000249311,-0.000166749,-0.000110239,-7.2023e-05,-4.64699e-05,-2.95653e-05,
        -1.84943e-05,-1.13126e-05,-6.69581e-06,-3.75333e-06,-1.89321e-06,-7.26446e-07,0])
    hrf=scipy.interpolate.interp1d(0.8*np.arange(len(hrf800)),hrf800,axis=0,kind="cubic",fill_value=0,bounds_error=False)(np.arange(numvols)*tr)[:,None]
else:
    hrf=np.loadtxt(hrffile)[:,None]
    if hrf.shape[0] < D.shape[-1]:
        hrf=np.vstack(hrf,np.zeros((numvols-hrf.shape[0],1)))
    elif hrf.shape[0] > numvols:
        hrf=hrf[:numvols,:]
        
resteffect=np.convolve(np.ones(numvols),hrf[:,0])[:numvols,None]

gmimg=None
ecsfimg=None
ewmimg=None

if gmfile:
    gmimg=nib.load(gmfile)
    
erosionbox=np.ones((erosionvoxels*2+1,erosionvoxels*2+1,erosionvoxels*2+1))>0

if csffile:
    if do_erode_csf:
        csfimg=nib.load(csffile)
        csfnew=binary_erosion(csfimg.get_fdata()>0,structure=erosionbox)
        ecsfimg=nib.Nifti1Image(csfnew,affine=csfimg.affine,header=csfimg.header)
    else:
        ecsfimg=nib.load(csffile)

if wmfile:
    if do_erode_wm:
        wmimg=nib.load(wmfile)
        wmnew=binary_erosion(wmimg.get_fdata()>0,structure=erosionbox)
        ewmimg=nib.Nifti1Image(wmnew,affine=csfimg.affine,header=csfimg.header)
    else:
        ewmimg=nib.load(wmfile)


if outlierfile:
    outliervec=np.loadtxt(outlierfile)>0
else:
    outliervec=np.zeros((numvols,1))>0
outliervec[:skipvols]=True

outliermat_names=[]
if np.any(outliervec):
    outliermat=np.zeros((outliervec.shape[0],np.sum(outliervec)))
    for i,t in enumerate(np.nonzero(outliervec>0)[0]):
        outliermat[t,i]=1
    outliermat_names=["outlier.%d" % (x) for x in range(outliermat.shape[1])]
else:
    outliermat=np.zeros((numvols,0))

onesmat=np.ones(mp.shape[0])[:,None]
detrendmat=np.arange(mp.shape[0])[:,None]

#minmal set of confounds used for extracting compcor
confounds1=np.hstack([detrendmat,mp,outliermat,resteffect])
#confounds1=np.hstack([onesmat,addsquare(addderiv(mp)),addderiv(resteffect),outliermat,detrendmat])

gmreg=np.zeros((numvols,0))
wmreg=np.zeros((numvols,0))
csfreg=np.zeros((numvols,0))

gmreg_names=[]
wmreg_names=[]
csfreg_names=[]

if gmimg:
    if verbose:
        print("Compute GM mean")
    gmreg=getmaskcomps(D,gmimg,confounds1,1)
    gmreg_names=["GM.%d" % (x) for x in range(gmreg.shape[1])]
if ewmimg:
    if verbose:
        print("Compute eroded WM regressors")
    wmreg=getmaskcomps(D,ewmimg,confounds1,5)
    wmreg_names=["WM.%d" % (x) for x in range(wmreg.shape[1])]
if ecsfimg:
    if verbose:
        print("Compute eroded CSF regressors")
    csfreg=getmaskcomps(D,ecsfimg,confounds1,5)
    csfreg_names=["CSF.%d" % (x) for x in range(csfreg.shape[1])]

    
#mp: deriv then square, final = 24 params
#larger set of confounds with derivatives etc for full denoising
#confounds=np.hstack([onesmat,wmreg,csfreg,addsquare(addderiv(mp)),addderiv(resteffect),outliermat,detrendmat])

#gmreg=np.mean(D.get_fdata()[gmimg.get_fdata()>0],axis=0)[:,None]
#gmreg=addderiv(gmreg)

#confoundnames=["ones"]+gmreg_names+wmreg_names+csfreg_names+mp_names+["rest"]+outliermat_names+["linear"]
#confounds=np.hstack([onesmat,gmreg,wmreg,csfreg,mp,resteffect,outliermat,detrendmat])

confoundnames=["ones"] + addderiv_txt(gmreg_names) + wmreg_names + csfreg_names + addsquare_txt(addderiv_txt(mp_names)) + addderiv_txt(["rest"]) + outliermat_names + ["linear"]
confounds=np.hstack([onesmat,addderiv(gmreg),wmreg,csfreg,addsquare(addderiv(mp)),addderiv(resteffect),outliermat,detrendmat])

if outputfile.lower().endswith(".mat"):
    savemat(outputfile,{"confounds":confounds,"confoundnames":confoundnames},format='5',do_compression=True)
else:
    confoundheadertxt=" ".join(confoundnames)
    np.savetxt(outputfile,confounds,fmt="%.18f",header=confoundheadertxt,comments="# ")
    
print("Saved %s" % (outputfile))
