movfiletype=hcp
gmfile=inputdir/mysubj_graymatter.nii.gz
wmfile=inputdir/mysubj_whitematter.nii.gz
csffile=inputdir/mysubj_csf.nii.gz
erosionvoxels=1
filtname="hpf"
filtarg="--lowfreq 0.008"
skipvols="5"
gsrarg="--gsr"
connarg="--connmeasure covariance --shrinkage 0"
roiname=fs86
roifile=fs86.nii.gz


for scan in scan1 scan2; do
    scanfile=inputdir/mysubj_${scan}.nii.gz
    movfile=inputdir/mysubj_${scan}_Movement_Regressors.txt
    outputbase=outputdir/mysubj_${scan}


    python fmri_outlier_detection.py --input ${scanfile} --mask ${scandir}/brainmask_fs.2.nii.gz --motionparam ${movfile} --motionparamtype ${movfiletype} --connstandard --output ${outputbase}_outliers.txt --outputparams ${outputbase}_outlier_parameters.mat

    python fmri_save_confounds.py --input ${scanfile} --motionparam ${movfile} --motionparamtype ${movfiletype} --gmmask ${gmfile} --wmmask ${wmfile} --csfmask ${csffile} --erosionvoxels ${erosionvoxels} --outlierfile ${outputbase}_outliers.txt --skipvols ${skipvols} --output ${outputbase}_fmriclean_confounds.mat

    python fmri_save_parcellated_timeseries.py --input ${scanfile} --roifile ${roifile} --outbase ${outputbase}_${roiname} --outputformat mat

    python fmri_clean_parcellated_timeseries.py --input "${outputbase}_${roiname}_ts.mat" --confoundfile ${outputbase}_fmriclean_confounds.mat $filtarg $gsrarg --outbase ${outputbase}_fmriclean_${filtname} --skipvols ${skipvols} ${connarg} --outputformat mat --sequentialroi
done

# for concatenation
concatbase=outputdir/mysubj
inputlist="${concatbase}_scan1_${roiname}_ts.mat ${concatbase}_scan2_${roiname}_ts.mat"
confoundlist="${concatbase}_scan1_fmriclean_confounds.mat ${concatbase}_scan2_fmriclean_confounds.mat"
python fmri_clean_parcellated_timeseries.py --input ${inputlist} --confoundfile ${confoundlist} $filtarg $gsrarg --outbase ${concatbase}_concat_fmriclean_${filtname} --skipvols ${skipvols} --outputformat mat --sequentialroi --concat ${connarg}


# for alff and falff, assumes fmri_outlier_detection.py and fmri_save_confounds.py have been run
# Need to run this on a DENOISED time series, or at least DETRENDED
scanfile=inputdir/mysubj_scan1.nii.gz
outputbase=outputdir/mysubj_scan1
tr=0.8 #if left out, can be inferred from scanfile header
skipvols=5
alff_lowfreq="0.008"
alff_highfreq="0.09"

#output will be ${outputbase}_fmriclean${filtname}${gsrname}_tsclean.nii.gz
python fmri_clean_parcellated_timeseries.py --input ${scanfile} --confoundfile ${outputbase}_fmriclean_confounds.mat $gsrarg --outbase ${outputbase}_fmriclean --skipvols ${skipvols} --connmeasure none --savets
	
python fmri_alff.py --input ${outputbase}_fmriclean${filtname}${gsrname}_tsclean.nii.gz --confoundfile ${outputbase}_fmriclean_confounds.mat --outbase ${outputbase}_fmriclean${filtname}${gsrname}_tsclean --lffrange ${alff_lowfreq} ${alff_highfreq} --totalfreqrange 0 inf

#final output will be:
#${outputbase}_fmriclean${filtname}${gsrname}_tsclean_alff.nii.gz
#${outputbase}_fmriclean${filtname}${gsrname}_tsclean_falff.nii.gz