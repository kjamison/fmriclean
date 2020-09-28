# fmriclean
fMRI outlier detection, and parcellation, and denoising

To be be run AFTER preprocessing. The expected order of operations would be: 

1. fmri_save_parcellated_timeseries.py: Save parcellated timeseries with NO denoising, filtering, etc
    * Inputs = full NIfTI time series, ROI labeled volume
    * Outputs = *_ts.mat file with [timepoints x ROIs]
2. fmri_outlier_detection.py: Save information about motion and global signal outliers (comparable to ART in CONN toolbox)
    * Inputs = full NIfTI time series and motion parameter file, and a brain mask file for computing global signal (recommended if data is very large)
    * Outputs = outliers.txt (Tx1 binary vector of 1s and 0s where 1 = outlier timepoint) and outlier_parameters.mat (timeseries used for outlier estimation including global signal, FD, dvars, and individual motion regressors)
3. fmri_save_confounds.py: Compute nuisance regressors from data
    * Inputs = full NIfTI time series, motion parameter file, outlier.txt file, masks for WM, CSF, and GM (for GSR and CompCor regressors)
    * Outputs = confounds.mat (TxM) time series with columns for GSR, CompCor, motion, and outlier nuisance regressors
4. fmri_clean_parcellated_timesries.py: Denoise and/or filter the parcellated time series from (1), and save FC matrices
    * Inputs = parcellated _ts.mat from (1), confounds.mat from (3)
    * Options = Perform bandpass filtering after denoising, compute FC matrix using correlation (Pearson), covariance, precision (inv(cov)), or partial correlation
    * Outputs = FC(type).mat (RxR connectivity matrix), and/or _tsclean.mat (TxR denoised time series)