#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat, savemat
import sys
import subprocess
import argparse
import tempfile
import os
import glob

from utils import flatlist


def main(argv):
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputpattern',action='store',dest='inputpattern',required=True)
    parser.add_argument('--subjectlist',action='append',dest='subjectlist',nargs='*',required=True)
    parser.add_argument('--output',action='append',dest='outputfiles',nargs='*',required=True)
    parser.add_argument('--scannames',action='append',dest='scannames',nargs='*')

    args=parser.parse_args(argv)
    
    subjlistfiles=args.subjectlist
    outputfiles=args.outputfiles
    inputpattern=args.inputpattern
    scannames=args.scannames

    is_s3 = inputpattern.lower().startswith("s3://")
    
    subjlistfiles=flatlist([x.split(",") for x in flatlist(subjlistfiles)])
    outputfiles=flatlist([x.split(",") for x in flatlist(outputfiles)])
    
    if len(subjlistfiles) != len(outputfiles):
        raise Exception("Number of output files must match number of subject lists!")
    
    scannames=flatlist([x.split(",") for x in flatlist(scannames)])
    
    if len(scannames) == 0:
        scannames=['*']
    
    subject_list_names=[]
    subject_lists=[]
    for slf in subjlistfiles:
        
        fid=open(slf,'r')
        sl=fid.readlines()
        fid.close()
        subject_lists+=[[x.strip() for x in sl]]
    
    full_subject_list=list(set(flatlist(subject_lists)))
    full_subject_list.sort()
    
    #full_subject_list=full_subject_list[:10]
    
    with tempfile.TemporaryDirectory() as tempdir:
        #print(tempdir)
        matshape=None
        roi_labels=None
        Mnew=[]
        for i in range(len(subject_lists)):
            Mnew.append({})
        for sc in scannames:
            Csum=[]
            Ccount=[]
            for i in range(len(subject_lists)):
                Csum.append(None)
                Ccount.append(0)
            for s in full_subject_list:
                #download all scans for this pattern (could be 4, could be 1)
                #load it in
                filenew=inputpattern.replace("(subject)",s).replace("(scanname)",sc)
                if is_s3:
                    s3new_dir="/".join(filenew.split("/")[:-1])
                    s3new_file=filenew.split("/")[-1]
                    s3cmd=["aws","s3","sync",s3new_dir,tempdir,"--exclude","*","--include",s3new_file,"--quiet"]
                    subprocess.run(s3cmd)
                    filelist=glob.glob(tempdir+"/"+s3new_file)
                else:
                    filelist=glob.glob(filenew)
                    
                for f in filelist:
                    #print(f)
                    M=loadmat(f)
                    C=M['C']
                    if matshape is None:
                        matshape=C.shape
                    elif C.shape != matshape:
                        raise Exception("Matrix size does not match. Expected [%dx%d], loaded [%dx%d] from %s" % (matshape[0],matshape[1],C.shape[0],C.shape[1],f))
                
                    if roi_labels is None:
                        roi_labels=M['roi_labels']
                    elif np.any(M['roi_labels'] != roi_labels):
                        raise Exception("roi_labels does not match for %s" % (f))

                    for i in range(len(subject_lists)):
                        if s in subject_lists[i]:
                            if Csum[i] is None:
                                Csum[i]=np.zeros(matshape)
                            Csum[i]+=C
                            Ccount[i]+=1
                        
                    if is_s3 and os.path.exists(f):
                        os.remove(f)
                #break
            #break
            for i in range(len(subject_lists)):
                if Csum[i] is None:
                    continue
                scstr=sc
                if scstr == "*":
                    scstr="all"
                Mnew[i][scstr]=Csum[i]/Ccount[i]
                Mnew[i]['roi_labels']=roi_labels
                Mnew[i]['count']=Ccount[i]
                Mnew[i]['subjects']=subject_lists[i]
        
        for i in range(len(subject_lists)):
            M=Mnew[i]
            print("Saving %s, %d matrices, C=[%dx%d], count=%d" % (outputfiles[i], len(scannames), matshape[0],matshape[1], M["count"]))
            savemat(outputfiles[i],dict(M),format='5',do_compression=True)
    
if __name__ == "__main__":
    main(sys.argv[1:])