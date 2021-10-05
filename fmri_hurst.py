import numpy as np
import nibabel as nib
import argparse
import sys
from utils import *

def argument_parse(argv):
    parser=argparse.ArgumentParser(description='Dynamic Fluctuation Analysis (Hurst Exponent)')

    parser.add_argument('--input',action='store',dest='inputvol')
    parser.add_argument('--output',action='store',dest='outputvol')
    parser.add_argument('--outputflvals',action='store',dest='outputvol_lvals')
    parser.add_argument('--logfl',action='store_true',dest='logfl')
    parser.add_argument('--lvalues',action='append',dest='lvalues',type=int,nargs='*')
    parser.add_argument('--skipvols',action='store',dest='skipvols',type=int,default=5)
    parser.add_argument('--subsegmentlength',action='store',dest='segmentlength',type=int, help="Default=Full timeseries")
    parser.add_argument('--allowsubsegmentoverlap',action='store_true',dest='allowoverlap')
    parser.add_argument('--verbose',action='store_true',dest='verbose')

    return parser.parse_args(argv)

def flatlist(l):
    if l is None:
        return []
    return [x for y in l for x in y]
    
def flatarglist(l):
    if l is None:
        return []
    return flatlist([x.split(",") for x in flatlist(l)])

def dfa_fast(vdata, istart, iend, L_all):
    istart=np.round(istart).astype(np.int)
    iend=np.round(iend).astype(np.int)
    
    #function takes in your time series (as columns), start and end time points, and the 
    #different L values you want to use in the implementation of DFA
    
    #takes the cumulative sum
    vdata=np.cumsum(vdata,axis=0)
    
    FL_all=np.zeros((len(L_all),vdata.shape[1]))
    
    #iterating through the L values you want to use
    for il,L in enumerate(L_all):
        X=np.hstack([np.atleast_2d(np.arange(L)).T,np.ones((L,1))])
        
        #nice thing about this approach is if your data isn't an integer
        #multiple of the length L, it will just average as many windows as can fit
        c=0
        FL=np.zeros((1,vdata.shape[1]))
        for i in range(istart,min(iend+1,vdata.shape[0])-L+1,L):
            vtmp=vdata[i:i+L,:]
            #b=X\vtmp
            #y=X*b
            #r=vtmp-X*(X\vtmp)
            #calculates rms for that window
            #lstsq(a,b) is ax=b, so x=inv(a)*b
            b,sse,_,_=np.linalg.lstsq(X,vtmp,rcond=None)
            #rms=np.sqrt(np.mean((vtmp-X@b)**2,axis=0))
            rms=np.sqrt(sse/vtmp.shape[0])
            FL+=rms
            c+=1
        FL/=c
        FL_all[il,:]=FL
    
    #F(L) ~ bL^a, so fit log(FL)=a*log(L)+log(b)
    logFL=np.log(FL_all)

    X=np.array(L_all).flatten()
    logX=np.hstack([np.atleast_2d(np.log(X)).T,np.ones((len(L_all),1))])
    b,_,_,_=np.linalg.lstsq(logX,logFL,rcond=None)
    alpha=b[0,:]
    return alpha,FL_all
    
def run_dfa(argv):
    args=argument_parse(argv)
    inputfile=args.inputvol
    outputfile=args.outputvol
    outputfile_lvals=args.outputvol_lvals
    skipvols=args.skipvols
    lvalues=np.sort(flatlist(args.lvalues))
    logfl=args.logfl
    seglength=args.segmentlength
    allowoverlap=args.allowoverlap
    if skipvols is None:
        skipvols=0
    
    if seglength is None or seglength<=0:
        seglength=0
    
    
    ####
    Dt,roivals,roisizes,tr_input,vol_info,input_extension = load_input(inputfile)
    print("Loaded input file: %s (%dx%d)" % (inputfile,Dt.shape[0],Dt.shape[1]))
    
    Dt=Dt[skipvols:,:]
    
    if seglength == 0:
        seglength=Dt.shape[0]
    
    if allowoverlap:
        #alternatively, we could allow overlap....
        numseg=np.ceil(Dt.shape[0]/seglength)
        segstarts=np.floor(np.linspace(0,Dt.shape[0]-seglength,numseg))
    else:
        #no overlap, might not catch all timepoints
        segstarts=np.arange(0,Dt.shape[0]-seglength+1,seglength)
    
    hurst_allsegs=[]
    fl_allsegs=[]
    for s in segstarts:
        hurst,fl = dfa_fast(Dt,s,s+seglength-1,lvalues)
        hurst_allsegs.append(hurst)
        if outputfile_lvals is not None:
            fl_allsegs.append(fl)
        
    if len(hurst_allsegs)==1:
        segstr="1 segment"
    else:
        segstr="%d segments" % (len(hurst_allsegs))
    
    print("Computed Hurst for %s of length %d with L=%s" % (segstr,seglength,str(lvalues)))
    if outputfile is not None:
        hurst_allsegs=np.stack(hurst_allsegs).mean(axis=0).astype(np.float32)
        
        savedfilename, shapestring = save_timeseries(outputfile, None, {"ts":hurst_allsegs,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr_input}, vol_info)
        print("Saved %s (%s)" % (savedfilename,shapestring))
    
    if outputfile_lvals is not None:
        fl_allsegs=np.stack(fl_allsegs).mean(axis=0)
        if logfl:
            fl_allsegs=np.log(fl_allsegs)
        fl_allsegs=fl_allsegs.astype(np.float32)
        
        savedfilename, shapestring = save_timeseries(outputfile_lvals, None, {"ts":fl_allsegs,"roi_labels":roivals,"roi_sizes":roisizes,"repetition_time":tr_input}, vol_info)
        print("Saved %s (%s)" % (savedfilename,shapestring))
    
if __name__ == "__main__":
    run_dfa(sys.argv[1:])
