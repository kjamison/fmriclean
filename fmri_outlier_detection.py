import nibabel as nib
import numpy as np
import sys
import argparse
from pathlib import Path 
from utils import prepadZero, params2matrix, read_motion_params, read_fmriclean_outlier_params, package_version_dict

def argument_parse(argv):
    parser=argparse.ArgumentParser(description='ART-style outlier detection on motion-corrected fMRI time series and motion parameter estimates')

    parser.add_argument('--input','-i',action='store',dest='inputvol')
    parser.add_argument('--motionparam','--motionparams','-p',action='store',dest='mpfile')
    parser.add_argument('--motionparamtype','--motionparamstype','-pt',action='store',dest='mptype',choices=['spm','hcp','fsl','fmriprep'])
    parser.add_argument('--mask','-m',action='store',dest='maskvol',help='If not provided, compute mask automatically from time series')
    parser.add_argument('--output','-o',action='store',dest='outfile')
    parser.add_argument('--outputparams','-op',action='store',dest='outfile_params',help='Can be .mat (matlab format) or .txt')
    parser.add_argument('--globalthresh','-gt',action='store',dest='globalthresh',type=float,default=3)
    parser.add_argument('--motionthresh','-mt',action='store',dest='motionthresh',type=float,default=1)
    parser.add_argument('--rotationthresh','-rt',action='store',dest='rotationthresh',type=float,default=.02, help='(in radians)')
    parser.add_argument('--motioncombined','-mc',action='store_true',dest='motioncombined')
    parser.add_argument('--globalderiv','-gd',action='store_true',dest='globalderiv')
    parser.add_argument('--motionderiv','-md',action='store_true',dest='motionderiv')
    parser.add_argument('--connstandard',action='store_true',dest='connstandard',help='(use CONN toolbox default parameters: -gt 5 -mt .9 -mc -gd -md)')
    parser.add_argument('--connstrict',action='store_true',dest='connstrict',help='(use CONN toolbox strict parameters: -gt 3 -mt .5 -mc -gd -md)')
    parser.add_argument('--connloose',action='store_true',dest='connloose',help='(use CONN toolbox loose parameters: -gt 9 -mt 2 -mc -gd -md)')
    parser.add_argument('--excludevols','-ex',action='store',dest='excludevols',type=int,default=0,help='Number of volumes to AUTOMATICALLY mark as outliers at start of scan')
    
    parser.add_argument('--inputparams','-ip',action='store',dest='infile_params',help='Use previous outlier_params, but apply new thresholds. Can be .mat (matlab format) or .txt')
    
    parser.add_argument('--version', action='version',version=package_version_dict(as_string=True))
    
    return parser.parse_args(argv)
    
def fmri_outlier_detection(argv):
    args=argument_parse(argv)

    tsfile=args.inputvol
    movfile=args.mpfile
    movfile_type=args.mptype
    maskfile=args.maskvol
    outfile=args.outfile
    outfile_params=args.outfile_params
    exclude_vols=args.excludevols
    
    infile_params=args.infile_params

    if args.connstandard:
        print("Using CONN toolbox standard parameters")
        args.globalthresh=5
        args.motionthresh=0.9
        args.globalderiv=True
        args.motionderiv=True
        args.motioncombined=True
    
    elif args.connloose:
        print("Using CONN toolbox loose parameters")
        args.globalthresh=9
        args.motionthresh=2
        args.globalderiv=True
        args.motionderiv=True
        args.motioncombined=True
    
    elif args.connstrict:
        print("Using CONN toolbox strict parameters")
        args.globalthresh=3
        args.motionthresh=0.5
        args.globalderiv=True
        args.motionderiv=True
        args.motioncombined=True
    
    global_signal_thresh=args.globalthresh
    mvmt_thresh=args.motionthresh
    rot_thresh=args.rotationthresh
    do_composite_motion=args.motioncombined

    do_diff_globalmean=args.globalderiv
    do_diff_motionparams=args.motionderiv

    print("Global signal threshold (std): %g" % (global_signal_thresh))
    print("Global signal derivative: %s" % (do_diff_globalmean))
    print("Motion threshold (mm): %g" % (mvmt_thresh))
    print("Motion derivative: %s" % (do_diff_motionparams))
    print("Ignore first N volumes: %s" % (exclude_vols))
    if do_composite_motion:
        print("Motion outlier mode: Compute composite motion (bounding-box points)")
    else:
        print("Rotation threshold (rad): %g" % (rot_thresh))
        print("Motion outlier mode: All 6 parameters considered independently")
    
    if tsfile is None:
        Vimg=None
        print("No 4D input time series provided. Using motion only.")
    else:
        Vimg=nib.load(tsfile)
        print("Input volume %s is (%d,%d,%d,%d)" % (tsfile,Vimg.shape[0],Vimg.shape[1],Vimg.shape[2],Vimg.shape[3]))

    if infile_params:
        print("Loading outlier-related timeseries from previous output %s (Overriding other timeseries or motion files)" % (infile_params))
        mpdict=read_fmriclean_outlier_params(infile_params)
        
        g=mpdict['g']
        mv_data=mpdict['mv_data']
        dvars=mpdict['dvars']
        fd_power=mpdict['fd_power']
        
        tsfile=mpdict['input_options']['inputvol']
        movfile=mpdict['input_options']['mpfile']
        movfile_type=mpdict['input_options']['mptype']
        maskfile=mpdict['input_options']['maskvol']
        exclude_vols=mpdict['input_options']['excludevols']
        
        args.inputvol=tsfile
        args.mpfile=movfile
        args.mptype=movfile_type
        args.maskvol=maskfile
        args.excludevols=exclude_vols
    else:
        movfile_type=movfile_type.lower()
        #V=Vimg.get_fdata(dtype=np.float32,caching='unchanged')[:,:,:,exclude_vols:]

        #read in motion parameters (HCP saved mmx,mmy,mmz, degx,degy,degz)
        mp, mp_names = read_motion_params(movfile, movfile_type)
        mp=mp[exclude_vols:,:]
    
        if Vimg is None:
            maskedmean=np.zeros(mp.shape[0])
            maskedmedian=np.zeros(mp.shape[0])
            dvars_orig=np.zeros(mp.shape[0]-1)
        else:
            if maskfile:
                Maskimg=nib.load(maskfile)
                M=Maskimg.get_fdata()>0
                #keep only the masked voxels
                #V=V[M>0]
                blocksize=200
                maskedmean=np.zeros(Vimg.shape[3],dtype=np.float32)
                maskedmedian=np.zeros(Vimg.shape[3],dtype=np.float32) #need this so we can approximate np.median(V[M]) later
                dvars_orig=np.zeros(Vimg.shape[3],dtype=np.float32)
    
                #V=np.zeros((np.sum(M),Vimg.shape[3]),dtype=np.float32)
    
                Vblock=np.zeros((np.sum(M),blocksize),dtype=np.float32)
                for i in range(0,Vimg.shape[3],blocksize):
                    blockstop=min(i+blocksize,Vimg.shape[3])
                    #V[:,i:blockstop]=Vimg.slicer[...,i:blockstop].get_fdata(dtype=np.float32,caching="unchanged")[M]
                    vprev=Vblock[:,-1][:,None] #rotate last timepoint of previous block to front of this block
                    Vblock=Vimg.slicer[...,i:blockstop].get_fdata(dtype=np.float32,caching="unchanged")[M]
                    maskedmean[i:blockstop]=np.mean(Vblock,axis=0)
                    maskedmedian[i:blockstop]=np.median(Vblock,axis=0)
                    dvars_orig[i:blockstop]=np.mean(np.diff(np.hstack([vprev,Vblock]),axis=1)**2,axis=0)
                dvars_orig=dvars_orig[1:]
                dvars_orig=np.sqrt(dvars_orig)
    
                dvars_orig=dvars_orig[exclude_vols:]
                maskedmean=maskedmean[exclude_vols:]
                maskedmedian=maskedmedian[exclude_vols:]
                #V=V[:,exclude_vols:]
                print("Mask volume %s contains %d masked voxels" % (maskfile,np.sum(M)))

            else:
                V=Vimg.get_fdata(dtype=np.float32,caching='unchanged')[:,:,:,exclude_vols:]
                M=np.reshape(V,[-1,V.shape[-1]])
                M=M>(np.nanmean(M,axis=0)/8.)
                M=np.reshape(np.all(M,axis=1),Vimg.shape[:3])
                V=V[M>0]
                maskedmean=np.mean(V,axis=0)
                maskedmedian=np.median(V,axis=0)
                dvars_orig=np.mean(np.diff(V,axis=1)**2,axis=0) #/np.mean(V)
                dvars_orig=np.sqrt(dvars_orig)
                print("Computed mask contains %d voxels" % (np.sum(M)))
    
    
        #############################
        # calculate global signal


        #g=np.atleast_2d(np.mean(V,axis=0)).T
        g=np.atleast_2d(maskedmean).T
        gsigma=.7413*np.diff(np.percentile(g,[25,75]))
        gsigma[gsigma==0]=1
        gmean=np.median(g)
        gnorm=(g-gmean)/gsigma;
        dg=np.vstack([[0], np.diff(g,axis=0)])
        dgsigma=.7413*np.diff(np.percentile(dg,[25,75]))
        dgsigma[dgsigma==0]=1
        dgmean=np.median(dg)
        dgnorm=(dg-dgmean)/dgsigma
        g=np.hstack([g,gnorm,dg,dgnorm])


        ##############################
        # dvars

        #dvars=1000*dvars_orig/np.median(V)
        #dvars=1000*dvars_orig/np.median(maskedmean) #how similar is this?
        if np.all(maskedmedian==0):
            dvars=dvars_orig
        else:
            dvars=1000*dvars_orig/np.median(maskedmedian) #this is very cloes to np.median(V)


        #prepend a minimal value (could be 0 but makes plot look ugly)
        #to make it the right size and ensure it won't exceed threshold
        preval=np.min(dvars)
        #preval=0
        dvars=np.hstack([preval,dvars])[:,None]

        ###############################
        # FD (Power 2011)
        mpdiff=np.diff(mp,axis=0)
        mpdiff[:,3:]=mpdiff[:,3:]*50 #multiply radians by 50 to approximate mm displacement around a sphere with radius 50mm
        fd_power=np.sum(np.abs(mpdiff),axis=1)
        fd_power=np.hstack([0,fd_power])[:,None]

        ###############################
        # calculate motion-related parameters
        respos=np.diag([70,70,75]).astype(np.float64)
        resneg=np.diag([-70,-110,-45]).astype(np.float64)

        z34=np.zeros([3,4])
        z31=np.zeros([3,1])
        e3=np.eye(3)
        res=np.vstack([np.hstack([respos,z31,z34,z34,e3,z31]), #; % 6 control points: [+x,+y,+z,-x,-y,-z];
            np.hstack([z34,respos,z31,z34,e3,z31]),
            np.hstack([z34,z34,respos,z31,e3,z31]),
            np.hstack([resneg,z31,z34,z34,e3,z31]),
            np.hstack([z34,resneg,z31,z34,e3,z31]),
            np.hstack([z34,z34,resneg,z31,e3,z31])])

        mv_data=mp.copy()
        mv_data=np.hstack([mv_data,np.zeros([mv_data.shape[0],51-mv_data.shape[1]])]);
        for i in range(mp.shape[0]):
            Pflat=params2matrix(mp[i,:]).T.flatten()
            mv_data[i,13:31]=Pflat@(res.T)

        #resposneg=np.hstack([np.vstack([respos,resneg]),np.ones((6,1))]).T
        #for i in range(mp.shape[0]):
        #    Pmat=params2matrix(mp[i,:])
        #    mv_data[i,13:31]=(Pmat*resposneg)[:-1,:].T.flatten()

        mv_data[:,6]=np.sqrt(np.sum(np.abs(mv_data[:,:3]**2),axis=1))
        mv_data[1:,7:13]=np.diff(mv_data[:,:6],axis=0)
        mv_data[:,31]=np.sqrt(np.mean(np.abs(mv_data[:,13:31]-np.mean(mv_data[:,13:31],axis=0))**2,axis=1))
        mv_data[1:,32:50]=np.diff(mv_data[:,13:31],axis=0)
        mv_data[1:,50]=np.max(np.sqrt(np.sum(np.reshape(np.abs(mv_data[1:,32:50])**2,[-1,6,3]),axis=2)),axis=1)



        ######################################


        #pad outlier param timecourses with zeros for all excluded vols
        g=prepadZero(g,exclude_vols)
        mv_data=prepadZero(mv_data,exclude_vols)
        dvars=prepadZero(dvars,exclude_vols)
        fd_power=prepadZero(fd_power,exclude_vols)

    ######################################
    # identity outliers


    dvars_iqr=np.percentile(dvars,[25,75])
    dvars_threshv=dvars_iqr[1]+1.5*(dvars_iqr[1]-dvars_iqr[0])
    
    fd_iqr=np.percentile(fd_power,[25,75])
    fd_threshv=fd_iqr[1]+1.5*(fd_iqr[1]-fd_iqr[0])
    fd_outliers=fd_power>fd_threshv

    if do_diff_globalmean:
        gidx=3 #normalized(diff(globalmean))
        zoutliers=(np.abs(g[:,gidx,None])>global_signal_thresh) | (np.abs(np.vstack([g[1:,gidx,None],[0]]))>global_signal_thresh)
    else:
        gidx=1 #normalized(globalmean)
        zoutliers=np.abs(g[:,gidx,None])>global_signal_thresh
    
    zoutliers=zoutliers[:,0]

    ######
    #for diff (in MATLAB 1-based!)
    #swap 1:6 and 8:13
    #swap 14:32 and 33:51
    #
    
    if do_diff_motionparams:
        mvmt_idx=[7,8,9]
        rot_idx=[10,11,12]
        normv_idx=50
        mvmt_outliers=(np.abs(mv_data[:,mvmt_idx]) > mvmt_thresh) | (np.abs(np.vstack([mv_data[1:,mvmt_idx],[0,0,0]])) > mvmt_thresh) 
    else:
        mvmt_idx=[0,1,2]
        rot_idx=[3,4,5]
        normv_idx=31
        mvmt_outliers=np.abs(mv_data[:,mvmt_idx]) > mvmt_thresh
    
    mvmt_outliers_x=mvmt_outliers[:,0]
    mvmt_outliers_y=mvmt_outliers[:,1]
    mvmt_outliers_z=mvmt_outliers[:,2]

    normv=mv_data[:,normv_idx,None]

    if do_diff_motionparams:
        mvmt_outliers_norm=(normv>mvmt_thresh) | (np.vstack([normv[1:,0,None],[0]])>mvmt_thresh)
    else:
        mvmt_outliers_norm=normv>mvmt_thresh
    
    mvmt_outliers_norm=mvmt_outliers_norm[:,0]


    rot_outliers=np.abs(mv_data[:,rot_idx]) > rot_thresh
    rot_outliers_x=rot_outliers[:,0]
    rot_outliers_y=rot_outliers[:,1]
    rot_outliers_z=rot_outliers[:,2]
    
    dvars_outliers=dvars>dvars_threshv
    fd_outliers=fd_power>fd_threshv
    
    print("%d global signal outliers: " % (len(np.nonzero(zoutliers)[0])),end='',flush=True)
    print([x.item() for x in np.nonzero(zoutliers)[0]])

    outlier_mask=zoutliers #from global signal
    if do_composite_motion:
        print("%d combined-motion outliers: " % (len(np.nonzero(mvmt_outliers_norm)[0])),end='',flush=True)
        print([x.item() for x in np.nonzero(mvmt_outliers_norm)[0]])
    
        outlier_mask = outlier_mask | mvmt_outliers_norm
    
    else:
        mvmt_outliers_all=mvmt_outliers_x | mvmt_outliers_y | mvmt_outliers_z | rot_outliers_x | rot_outliers_y | rot_outliers_z
        print("%d separate motion outliers: " % (len(np.nonzero(mvmt_outliers_all)[0])),end='',flush=True)
        print([x.item() for x in np.nonzero(mvmt_outliers_all)[0]])
    
        outlier_mask = outlier_mask | mvmt_outliers_all

    print("%d starting volumes considered outliers: " % (exclude_vols),end='',flush=True)
    print([x.item() for x in np.arange(exclude_vols)])

    outlier_mask[:exclude_vols]=True

    print("%d total unique outliers: " % (len(np.nonzero(outlier_mask)[0])),end='',flush=True)
    print([x.item() for x in np.nonzero(outlier_mask)[0]])

    print("Mean FD: %g" % (np.mean(fd_power)))
    print("Mean DVARS: %g" % (np.mean(dvars)))
    
    if outfile:
        print("Saving binarized outlier mask (1=outlier) to %s" % (outfile))
        np.savetxt(outfile,outlier_mask,"%d")

    if outfile_params:
        print("Saving time series used for estimating outliers to %s" % (outfile_params))
        if outfile_params.lower().endswith(".mat"):
            from scipy.io import savemat
            argdict=vars(args)
            #need to swap None for empty string so it can save as matlab
            for k,v in argdict.items():
                if v is None:
                    argdict[k]=''
            mpdict={'input_options':argdict,'g':g.astype(np.float32),'mv_data':mv_data.astype(np.float32), \
                'dvars':dvars.astype(np.float32),'fd_power':fd_power.astype(np.float32)}
            savemat(outfile_params,mpdict,format='5',do_compression=True)
        else:
            argdict=vars(args)
            hdrtxt="\n".join(['%s:%s' % (k,v) for k,v in argdict.items()])
            np.savetxt(outfile_params,np.hstack([g,mv_data,dvars,fd_power]),fmt='%f',header=hdrtxt,comments='# ')

if __name__ == "__main__":
    fmri_outlier_detection(sys.argv[1:])