import numpy as np
import sys
import argparse
from scipy.io import loadmat,savemat
import os.path

from utils import *

#this will concat ROIs 1-18 from in1_ts.mat and 1-N in in2_ts.mat

#python fmri_reorder_parcellated_timeseries.py --output out_ts.mat --input in1_ts.mat 1-18 --input in2_ts.mat 
def argument_parse(argv):
    parser=argparse.ArgumentParser(description='Reorder and/or merge ROIs in time series')
    
    parser.add_argument('--input',action='append',dest='inputlist',nargs='*')
    parser.add_argument('--output',action='store',dest='outputfile')
    
    return parser.parse_args(argv)

    
#########################################################

#this old version splits by "," first, but really we want to use commas within an roi input set eg: 1,2=4 and just use spaces for separate inputs
#but keep this here for now for reference in case another script used it and I forgot
def parse_roilist_oldtype(roilist):
    inlist=[]
    outlist=[]
    for s in roilist:
        for p in s.split(","):
            if not p:
                continue
            outval=None
            if "=" in p:
                p_eq=p.split("=")
                p=p_eq[0]
                outval=float(p_eq[-1])
            if "-" in p:
                pp=p.split("-")
                ppval=np.arange(float(pp[0]),float(pp[-1])+1).tolist()
            else:
                ppval=[float(p)]
            if outval is None:
                outval=ppval
            else:
                outval=[outval]*len(ppval)
            inlist+=ppval
            outlist+=outval
    return inlist,outlist
    
def parse_roilist(roilist):
    inlist=[]
    outlist=[]
    for p in roilist:
        outval=None
        if "=" in p:
            p_eq=p.split("=")
            p=p_eq[0]
            outval=float(p_eq[-1])
        if "-" in p:
            pp=p.split("-")
            ppval=np.arange(float(pp[0]),float(pp[-1])+1).tolist()
        elif "," in p:
            pp=p.split(",")
            ppval=[float(ppi) for ppi in pp]
        else:
            ppval=[float(p)]
        if outval is None:
            outval=ppval
        else:
            outval=[outval]*len(ppval)
        inlist+=ppval
        outlist+=outval
    return inlist,outlist
    
def fmri_reorder_parcellated_timeseries(argv):
    args = argument_parse(argv)
    
    outputfile=args.outputfile
    
    inputlist=args.inputlist
    
    inputstruct_list=[]
    
    for i in inputlist:
        inputfile=i[0]
        input_labels, output_labels=parse_roilist(i[1:])

        inputstruct_list+=[{'file':inputfile,'input_labels':input_labels,'output_labels':output_labels}]
    
    Mnew=None
    tsdict_valshift=0
    tsdict_vals={}
    tsdict_sizelist={}
    for inputinfo in inputstruct_list:
        M=loadmat(inputinfo['file'])
        roivals=M['roi_labels'].flatten()
        #roisizes=M['roi_sizes'].astype(np.float).flatten()
        roisizes=M['roi_sizes'].astype(float).flatten()
        if Mnew is None:
            Mnew=M.copy()
        
        if len(inputinfo['input_labels']) == 0:
            inputinfo['input_labels']=roivals
            inputinfo['output_labels']=roivals + tsdict_valshift
        for i,vi in enumerate(inputinfo['input_labels']):
            vo=inputinfo['output_labels'][i]
            m=roivals==vi
            if not any(m):
                #fill zeros if this roi label is not found in input
                rts=np.zeros(M['ts'][:,0].shape)
                rsize=0
            else:
                midx=np.where(m)[0][0]
                rts=M['ts'][:,midx]
                rsize=roisizes[midx]
            if vo in tsdict_vals:
                tsdict_vals[vo]=tsdict_vals[vo] + rts*rsize
                tsdict_sizelist[vo]+=rsize
            else:
                tsdict_vals[vo]=rts*rsize
                tsdict_sizelist[vo]=rsize
                
        tsdict_valshift=max(list(tsdict_vals.keys()))
    
    uvals=np.sort(np.unique(list(tsdict_vals.keys())))
    Mnew['ts']=np.zeros((Mnew['ts'].shape[0], len(uvals)))
    Mnew['roi_labels']=uvals.astype(np.int32)
    Mnew['roi_sizes']=np.array([tsdict_sizelist[u] for u in uvals],dtype=np.int64)
    for i, u in enumerate(uvals):
        if tsdict_sizelist[u] > 0:
            sz=tsdict_sizelist[u]
        else:
            sz=1
        Mnew['ts'][:,i]=tsdict_vals[u]/sz
    
    savemat(outputfile,Mnew,format='5',do_compression=True)
    print('Saved %s [%d x %d]' % (outputfile,Mnew['ts'].shape[0],Mnew['ts'].shape[1]))
    
if __name__ == "__main__":
    fmri_reorder_parcellated_timeseries(sys.argv[1:])