import numpy as np
import nilearn
import nilearn.connectome
#import nipy.modalities.fmri.hrf
import sys
import argparse
from scipy.io import loadmat,savemat
import os.path
import re

#need some way to specify which denoising I want to do:
#enable/disable:
#filtering (already set)
#gsr (single flag right now)
#compcorr
parser=argparse.ArgumentParser(description='fMRI Denoising after parcellation')

parser.add_argument('--input',action='store',dest='inputvol')
parser.add_argument('--confoundfile',action='store',dest='confoundfile')
parser.add_argument('--outbase',action='store',dest='outbase')
parser.add_argument('--inputpattern',action='store',dest='inputpattern')
parser.add_argument('--roilist',action='store',dest='roilist')
parser.add_argument('--savets',action='store_true',dest='savets')
parser.add_argument('--skipvols',action='store',dest='skipvols',type=int,default=5)
parser.add_argument('--lowfreq',action='store',dest='lowfreq',type=float) #,default=0.008)
parser.add_argument('--highfreq',action='store',dest='highfreq',type=float)# ,default=0.09)
parser.add_argument('--repetitiontime','-tr',action='store',dest='tr',help='TR in seconds',type=float)
parser.add_argument('--filterstrategy',action='store',dest='filterstrategy',choices=['connregbp','orth','none'],default='connregbp')
parser.add_argument('--connmeasure',action='append',dest='connmeasure',choices=['none','correlation','partialcorrelation','precision','covariance'],nargs='*')
parser.add_argument('--outputformat',action='store',dest='outputformat',choices=['mat','txt'],default='txt')
parser.add_argument('--gsr',action='store_true',dest='gsr')
parser.add_argument('--nocompcorr',action='store_true',dest='nocompcorr')
parser.add_argument('--nomotion',action='store_true',dest='nomotion')
parser.add_argument('--motionparam',action='store',dest='mpfile')
parser.add_argument('--motionparamtype',action='store',dest='mptype',choices=['spm','hcp','fsl'],default='fsl')
parser.add_argument('--hrffile',action='store',dest='hrffile')
parser.add_argument('--outlierfile',action='store',dest='outlierfile')
parser.add_argument('--verbose',action='store_true',dest='verbose')

args=parser.parse_args()
inputvol=args.inputvol
movfile=args.mpfile
movfile_type=args.mptype.lower()
outbase=args.outbase
outlierfile=args.outlierfile
skipvols=args.skipvols
bpfmode=args.filterstrategy
connmeasure=args.connmeasure
outputformat=args.outputformat
confoundfile=args.confoundfile
verbose=args.verbose
do_gsr=args.gsr
do_nocompcorr=args.nocompcorr
do_nomotion=args.nomotion
tr=args.tr
roilist=args.roilist
inputpattern=args.inputpattern
do_savets=args.savets

if connmeasure is None:
    connmeasure=[['correlation']]
    
flatlist=lambda l: [x for y in l for x in y]
connmeasure=flatlist([x.split(",") for x in flatlist(connmeasure)])
connmeasure=["partial correlation" if x=="partialcorrelation" else x for x in connmeasure]
connmeasure=["partial correlation" if x=="partial" else x for x in connmeasure]
connmeasure=list(set(connmeasure))
if 'none' in connmeasure:
    connmeasure=['none']
    
connmeasure_shortname={"correlation":"corr", "partial correlation":"pcorr", "precision": "prec", "tangent":"tan", "covariance":"cov"}
    
hrffile=None
if args.hrffile:
    #since nipy isn't working with numpy 1.18
    hrffile=args.hrffile

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

print("Input time series: %s" % (inputvol))
print("Confound file: %s" % (confoundfile))
print("Motion parameter file (%s-style): %s" % (movfile_type,movfile))
print("Outlier timepoint file: %s" % (outlierfile))
print("Ignore first N volumes: %s" % (skipvols))
print("Filter strategy: %s" % (bpfmode))
print("Filter band-pass Hz: [%s,%s]" % (bpf[0],bpf[1]))
print("Output basename: %s" % (outbase))
print("Skip CompCorr (WM+CSF): %s" % (do_nocompcorr))
print("Skip motion regressors: %s" % (do_nomotion))
print("Save denoised time series: %s" % (do_savets))
print("Connectivity measures: ", connmeasure)

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

def loadinput(filename):
    tr=None
    if filename.lower().endswith(".mat"):
        M=loadmat(filename)
        Dt=M['ts']
        if 'repetition_time' in M:
            tr=M['repetition_time'][0]
        roivals=M['roi_labels'][0]
        roisizes=M['roi_sizes'][0]
    else:
        Dt=np.loadtxt(filename)
    
        fid = open(filename, 'r') 
        roivals=np.arange(Dt.shape[1])+1
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
    return Dt,roivals,roisizes,tr
    
if inputpattern and roilist:
    inputvol_list=[]
    roiname_list=[]
    for roi in roilist.split(","):
        if not roi:
            continue
        roiname=roi.split("=")[0]
        inputvol_list+=[inputpattern % (roiname)]
        roiname_list+=[roiname]
else:
    inputvol_list=[inputvol]
    if roilist:
        roiname=roilist.split(",")[0].split("=")[0]
    else:
        roiname=""
    roiname_list=[roiname]
    
for roiname,inputvol in zip(roiname_list,inputvol_list):
    Dt,roivals,roisizes,tr_input = loadinput(inputvol)
    if tr_input:
        tr=tr_input

    numvols=Dt.shape[0]

    outliermat=np.zeros((numvols,1))
    resteffect=np.zeros((numvols,0))
    gmreg=np.zeros((numvols,0))
    wmreg=np.zeros((numvols,0))
    csfreg=np.zeros((numvols,0))
    mp=np.zeros((numvols,0))

    ############################################
    confoundlist=None
    confoundnames=None
    if confoundfile:
        if confoundfile.lower().endswith(".mat"):
            M=loadmat(confoundfile)
            confoundlist=M['confounds']
            confoundnames=M['confoundnames']
        else:
            confoundlist=np.loadtxt(confoundfile)
    
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

        if len(gmidx)>0:
            gmreg=confoundlist[:,gmidx]
        if len(wmidx)>0:
            wmreg=confoundlist[:,wmidx]
        if len(csfidx)>0:
            csfreg=confoundlist[:,csfidx]
        if len(mpidx)>0:
            mp=confoundlist[:,mpidx]
            if mp.shape[1]>6:
                mp=mp[:,:6]
        if len(restidx)>0:
            resteffect=confoundlist[:,restidx[0]][:,None]
        if len(outlieridx)>0:
            outliermat=confoundlist[:,outlieridx]

    #read in motion parameters (HCP saved mmx,mmy,mmz, degx,degy,degz)
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

    if hrffile:
        hrf=np.loadtxt(hrffile)[:,None]
        if hrf.shape[0] < D.shape[-1]:
            hrf=np.vstack(hrf,np.zeros((numvols-hrf.shape[0],1)))
        elif hrf.shape[0] > numvols:
            hrf=hrf[:numvols,:]
        resteffect=np.convolve(np.ones(numvols),hrf[:,0])[:numvols,None]
    elif resteffect.shape[-1]==0:
        import nipy.modalities.fmri.hrf
        hrf=nipy.modalities.fmri.hrf.spmt(np.arange(numvols)*tr)[:,None]
        #np.savetxt("hrf_%d.txt" % (numvols),hrf,fmt="%.18f");
        resteffect=np.convolve(np.ones(numvols),hrf[:,0])[:numvols,None]
        
    if outlierfile:
        outliermat=np.loadtxt(outlierfile)>0

    outliermat[:skipvols,:]=True

    #if outlier mat is a masking vector, split into columns        
    if outliermat.shape[1]==1:
        if np.sum(outliermat,axis=0)>1:

            outliermat_new=np.zeros((outliermat.shape[0],np.sum(outliermat)))
            for i,t in enumerate(np.nonzero(outliermat>0)[0]):
                outliermat_new[t,i]=1
            outliermat=outliermat_new
        else:
            outliermat=np.zeros((numvols,0))
        
        
    onesmat=np.ones(mp.shape[0])[:,None]
    detrendmat=np.arange(mp.shape[0])[:,None]

    ########################################
    if not do_gsr:
        gmreg=np.zeros((numvols,0))
    
    if do_nocompcorr:
        wmreg=np.zeros((numvols,0))
        csfreg=np.zeros((numvols,0))
        
    if do_nomotion:
        mp=np.zeros((numvols,0))
        
    confounds=np.hstack([onesmat,addderiv(gmreg),wmreg,csfreg,addsquare(addderiv(mp)),addderiv(resteffect),outliermat,detrendmat])
    
    if bpfmode=="orth":
        Dt_clean=nilearn.signal.clean(Dt, confounds=confounds, standardize='zscore', t_r=tr, detrend=False,low_pass=bpf[1], high_pass=bpf[0])
    else:
        Dt_clean=nilearn.signal.clean(Dt, confounds=confounds, standardize='zscore', t_r=tr, detrend=False)

    if bpfmode=="connregbp":
        #Dt=nilearn.signal.clean(Dt.copy(), detrend=False, standardize=False, low_pass=bpf[1], high_pass=bpf[0], t_r=tr)
        Dt_clean=fftfilt(Dt_clean, tr, bpf)

    roiheadertxt="ROI_Labels:\n"
    roiheadertxt+=" ".join(["%d" % (x) for x in roivals])
    roiheadertxt+="\nROI_Sizes(voxels):\n"
    roiheadertxt+=" ".join(["%d" % (x) for x in roisizes])
    roiheadertxt+="\nRepetition_time(sec): %g" % (tr)
   
    if do_gsr:
        gsrsuffix="_gsr"
    else:
        gsrsuffix=""

    if roiname:
        roisuffix="_"+roiname
    else:
        roisuffix=""
    
    if do_savets:
        if outputformat == "mat":
            savemat(outbase+roisuffix+gsrsuffix+"_tsclean.mat",{"ts":Dt_clean,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr},format='5',do_compression=True)
        else:
            np.savetxt(outbase+roisuffix+gsrsuffix+"_tsclean.txt",Dt_clean,fmt="%.18f",header=roiheadertxt,comments="# ")
        print("Saved %s%s%s_tsclean.%s" % (outbase,roisuffix,gsrsuffix,outputformat))
    
    for cm in connmeasure:
        if cm == 'none':
            continue
        E=nilearn.connectome.ConnectivityMeasure(kind=cm, vectorize=False, discard_diagonal=True)
        C=E.fit_transform([Dt_clean[skipvols:,:]])[0]
        if outputformat == "mat":
            savemat(outbase+roisuffix+gsrsuffix+"_FC%s.mat" % (connmeasure_shortname[cm]),{"C":C,"roi_labels":roivals,"roi_sizes":roisizes},format='5',do_compression=True)
        else:
            np.savetxt(outbase+roisuffix+gsrsuffix+"_FC%s.txt" % (connmeasure_shortname[cm]),C,fmt="%.18f",header=roiheadertxt,comments="# ")
            
        print("Saved %s%s%s_FC%s.%s" % (outbase,roisuffix,gsrsuffix,connmeasure_shortname[cm],outputformat))
