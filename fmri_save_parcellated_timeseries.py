import nibabel as nib
import numpy as np
import sys
import argparse
from scipy.io import savemat
import os.path
from scipy import sparse
 
parser=argparse.ArgumentParser(description='fMRI Parcellation')

parser.add_argument('--hcpmnidir',action='store',dest='hcpmnidir',help='Optional: HCP-style MNINonLinear directory to find input files')
parser.add_argument('--hcpscanname',action='store',dest='hcpscanname',help='Optional: Scan name within MNINonLinear/Results')
parser.add_argument('--input',action='store',dest='inputvol')
parser.add_argument('--mask',action='store',dest='maskfile')
parser.add_argument('--roifile',action='store',dest='roifile',help='ROI atlas label volume. can be --roifile myname=myname_parc.nii.gz')
parser.add_argument('--outbase',action='store',dest='outbase')
parser.add_argument('--outputformat',action='store',dest='outputformat',choices=['mat','txt'],default='txt')
parser.add_argument('--verbose',action='store_true',dest='verbose')

args=parser.parse_args()
hcpmnidir=args.hcpmnidir
hcpscanname=args.hcpscanname
inputvol=args.inputvol
maskfile=args.maskfile
roifile=args.roifile
outbase=args.outbase
outputformat=args.outputformat
verbose=args.verbose

if hcpmnidir and hcpscanname:
    if inputvol is None:
        inputvol="%s/Results/%s/%s.nii.gz" % (hcpmnidir,hcpscanname,hcpscanname)
    #maskfile="%s/Results/%s/RibbonVolumeToSurfaceMapping/goodvoxels.nii.gz" % (hcpmnidir,hcpscanname)
    print("HCP input shortcut: %s/Results/%s" % (hcpmnidir,hcpscanname))


print("Input time series: %s" % (inputvol))
print("Input voxel mask: %s" % (maskfile))
print("Atlas parcellation list: %s" % (roifile))
print("Output basename: %s" % (outbase))


D=nib.load(inputvol)
tr=D.header['pixdim'][4]
numvols=D.shape[-1]

if maskfile:
    goodvox=nib.load(maskfile)
    goodvoxmask=goodvox.get_fdata()>0
else:
    #goodvoxmask=~np.all(D.get_fdata()==0,axis=3)
    blocksize=200
    goodvoxmask=np.ones(D.shape[:3])
    for i in range(0,numvols,blocksize):
        blockstop=min(i+blocksize,numvols)
        goodvoxmask=goodvoxmask * ~np.all(D.slicer[...,i:blockstop].get_fdata(dtype=np.float32,caching="unchanged")==0,axis=3)
    goodvoxmask=goodvoxmask>0
    goodvox=nib.Nifti1Image(goodvoxmask,affine=D.affine,header=D.header)


   

#build a mask of all the voxels in any of the label volumes we input
#and then read the data for those voxels only 
#(that way we have a common matrix to parcellate quickly)
labelmask=np.zeros(D.shape[:3],dtype=np.float32)
for roi in roifile.split(","):
    if not roi:
        continue
    if roi.find("=") >= 0:
        roifilename=roi.split("=")[1]
    else:
        roifilename=roi
    labels_img=nib.load(roifilename)
    labelmask+=labels_img.get_fdata()
labelmask=((labelmask>0) * (goodvoxmask>0))>0

if verbose:
    print("starting to load data from labelmask: %d voxels" % (np.sum(labelmask)))
blocksize=200
Dmasked=np.zeros((D.shape[3],np.sum(labelmask)),dtype=np.float32)
for i in range(0,D.shape[3],blocksize):
    blockstop=min(i+blocksize,D.shape[3])
    Dmasked[i:blockstop,:]=D.slicer[...,i:blockstop].get_fdata(dtype=np.float32,caching="unchanged")[labelmask].T
if verbose:
    print("done loading data from labelmask. final size %dx%d" % (Dmasked.shape[0],Dmasked.shape[1]))
    
for roi in roifile.split(","):
    if not roi:
        continue
    
    if roi.find("=") >= 0:
        roiname=roi.split("=")[0]
        roifilename=roi.split("=")[1]
        roisuffix="_"+roiname
    else:
        roisuffix=""
        roifilename=roi
    
    labels_img=nib.load(roifilename)
    labelvol=labels_img.get_fdata()
    labelvolmask=((labelvol!=0) * (goodvoxmask>0))>0
    roivals, roivoxidx=np.unique(labelvol[labelvolmask],return_inverse=True)
    roisizes=np.bincount(roivoxidx)
    #header text to print for .txt output option
    roiheadertxt="ROI_Labels:\n"
    roiheadertxt+=" ".join(["%d" % (x) for x in roivals])
    roiheadertxt+="\nROI_Sizes(voxels):\n"
    roiheadertxt+=" ".join(["%d" % (x) for x in roisizes])
    roiheadertxt+="\nRepetition_time(sec): %g" % (tr)
    Pdata=np.round(labelvol[labelmask]).flatten()

    numvoxels_in_roivol=Pdata.size
    pmaskidx=np.where(Pdata!=0)[0]
    uroi, uidx=np.unique(Pdata[Pdata!=0],return_inverse=True)
    uroisize=np.bincount(uidx)
    uroisize_denom=1./uroisize
    unumroi=len(uroi)

    Psparse=sparse.csr_matrix((uroisize_denom[uidx],(pmaskidx,uidx)),shape=(numvoxels_in_roivol,unumroi),dtype=np.float32)

    Dparc=Dmasked @ Psparse

            
    if outputformat == "mat":
        savemat(outbase+roisuffix+"_ts.mat",{"ts":Dparc,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr},format='5',do_compression=True)
    else:
        np.savetxt(outbase+roisuffix+"_ts.txt",Dparc,fmt="%.18f",header=roiheadertxt,comments="# ")

    print("Saved %s_ts.%s" % (outbase+roisuffix,outputformat))
