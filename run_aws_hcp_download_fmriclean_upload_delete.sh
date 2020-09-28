#!/bin/bash

set -e
#set -x

#try to stop python/numpy from secretly using extra cores sometimes
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export FSLOUTPUTTYPE=NIFTI_GZ
export FSLDIR=$HOME/fsl
source $FSLDIR/etc/fslconf/fsl.sh
	
mrtrixdir=${HOME}/mrtrix3/bin

s3root=s3://kuceyeski-wcm-temp/kwj2001
s3roipath=s3://kuceyeski-wcm-temp/kwj2001/nemo2/nemo_atlases

if [ ! -e "$( which labelconvert )" ]; then
	export PATH=$PATH:$mrtrixdir
fi

if [ ! -e "$( which fslmaths )" ]; then
	export PATH=$PATH:$FSLDIR/bin
fi

do_preproc=1
do_connmeasure=0

#preproc for ALL atlases (fs86,cc200,cc400,shen268,cocoA,cocoB) takes 22min/subj when using 16 jobs on m5a.8xlarge (32 vCPU), or 8 jobs on m5a.4xlarge(16 CPU), etc..
# (note: if we use more than half the vCPU things start taking a lot longer)
#so running a batch of 48 (3 separate 8xlarge instances) takes (997/48)*22/60=7.6hrs

#connmeasure takes ~3 min for ALL atlases, flavors: atlas=(fs86,cc200,cc400,shen268,cocoA,cocoB), gsr=(yes,no), filt=(nofilt,bpf,hpf), conn=(corr,pcorr,cov,prec)
#and can run on all cores of a single 8xlarge instace, so (997/31)*3/60=95min

subject=$1
if [ "$2" = "-onlyconn" ]; then
	newroi=""
	do_preproc=0
	do_connmeasure=1
	newroi="$3"
else
	newroi="$2"
fi

if [ "${do_preproc}" = "0" ]; then
	outzipname=output_${subject}_fmriclean
	studydir=${HOME}/${outzipname}
	mkdir -p ${studydir}
	cd ${studydir}
	if [ "x${newroi}" = "x" ]; then
		roilist="fs86,cc400,cc200,shen268,cocommp438,cocommpsuit439"
	else
		roilist="${newroi}"
	fi
else
	if [ "x$newroi" = "x" ]; then
	
		outzipname=output_${subject}_fmriclean
		studydir=${HOME}/${outzipname}
		mkdir -p ${studydir}
		cd ${studydir}

		ziplistfile_todelete=todelete.txt


		aws s3 cp ${s3root}/HCP/hrf_1200.txt ./
		aws s3 cp ${s3root}/FreeSurferWMRegLut_withcerebellum.txt ./
		aws s3 cp ${s3root}/FreeSurferCSFRegLut.txt ./
		aws s3 cp ${s3root}/FreeSurferCorticalLabelTableLut.txt ./
		aws s3 cp ${s3root}/FreeSurferColorLUT.txt ./
		aws s3 cp ${s3root}/fs_default86.txt ./

		wmlabel=./FreeSurferWMRegLut_withcerebellum.txt
		csflabel=./FreeSurferCSFRegLut.txt
		gmlabel=./FreeSurferCorticalLabelTableLut.txt

		wbc=/usr/bin/wb_command

		mnidir=$studydir/$subject/MNINonLinear
		roidir=$mnidir/ROIs

		mkdir -p $mnidir
		mkdir -p $roidir
	
		aws s3 --profile hcp cp s3://hcp-openaccess/HCP_1200/${subject}/MNINonLinear/ROIs/wmparc.2.nii.gz $roidir/
		aws s3 --profile hcp cp s3://hcp-openaccess/HCP_1200/${subject}/MNINonLinear/ROIs/ROIs.2.nii.gz $roidir/
		aws s3 --profile hcp cp s3://hcp-openaccess/HCP_1200/${subject}/MNINonLinear/ribbon.nii.gz $mnidir/
		aws s3 --profile hcp cp s3://hcp-openaccess/HCP_1200/${subject}/MNINonLinear/aparc+aseg.nii.gz $mnidir/
		aws s3 --profile hcp cp s3://hcp-openaccess/HCP_1200/${subject}/MNINonLinear/aparc.a2009s+aseg.nii.gz $mnidir/
	
		aws s3 cp s3://kuceyeski-wcm-temp/kwj2001/HCP/${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_cerebSUIT.zip ./
		aws s3 cp s3://kuceyeski-wcm-temp/kwj2001/HCP/${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7.zip ./
		unzip -qo ${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_cerebSUIT.zip ${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_cerebSUIT_seq.mni2mm.nii.gz
		unzip -qo ${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7.zip ${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_seq.mni2mm.nii.gz
		rm -f ${subject}_hcpmmp_aseg_*.zip

		find ${studydir}/ -type f > ${ziplistfile_todelete}
	
		applywarp -i $mnidir/ribbon.nii.gz -r $roidir/wmparc.2.nii.gz -o $roidir/ribbon.2.nii.gz --interp=nn

		$wbc -volume-label-import $roidir/wmparc.2.nii.gz $wmlabel $roidir/WMReg.2.nii.gz -discard-others -drop-unused-labels
		$wbc -volume-label-import $roidir/wmparc.2.nii.gz $csflabel $roidir/CSFReg.2.nii.gz -discard-others -drop-unused-labels
		$wbc -volume-label-import $roidir/ribbon.2.nii.gz $gmlabel $roidir/GMReg.2.nii.gz -discard-others -drop-unused-labels

	
	
	
		fslmaths $roidir/WMReg.2.nii.gz -bin $roidir/WMReg.2.nii.gz
		fslmaths $roidir/CSFReg.2.nii.gz -bin $roidir/CSFReg.2.nii.gz

		fslmaths $roidir/GMReg.2.nii.gz -add $roidir/ROIs.2.nii.gz -bin $roidir/GMReg.2.nii.gz
		#fslmaths $roidir/GMReg.2.nii.gz -dilM -dilM -bin $roidir/GMReg_dil.2.nii.gz
		fslmaths $roidir/GMReg.2.nii.gz -dilM -bin $roidir/GMReg_dil.2.nii.gz
		fslmaths $roidir/GMReg_dil.2.nii.gz -binv -mul $roidir/WMReg.2.nii.gz -bin $roidir/WMReg_avoid.2.nii.gz
		fslmaths $roidir/GMReg_dil.2.nii.gz -binv -mul $roidir/CSFReg.2.nii.gz -bin $roidir/CSFReg_avoid.2.nii.gz

		labelconvert $mnidir/aparc+aseg.nii.gz ./FreeSurferColorLUT.txt ./fs_default86.txt $mnidir/fs86.nii.gz -force
		fslmaths $mnidir/aparc+aseg.nii.gz -mul 0 -add $mnidir/fs86.nii.gz $mnidir/fs86.nii.gz
		applywarp -i $mnidir/fs86.nii.gz -r $roidir/wmparc.2.nii.gz -o $mnidir/fs86.2.nii.gz --interp=nn

		roilist=""
	
		roilist+=" fs86=${mnidir}/fs86.2.nii.gz cc400=cc400_new1mm_seq392.nii.gz cc200=cc200_new1mm.nii.gz shen268=shen268_MNI1mm_dil1.nii.gz"
		roilist+=" cocommp438=${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_seq.mni2mm.nii.gz cocommpsuit439=${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_cerebSUIT_seq.mni2mm.nii.gz"
	
		if [ -e $HOME/yeona/${subject}/fs86_yeona_mni2.nii.gz ]; then
			roilist+=" fs86yeona=$HOME/yeona/${subject}/fs86_yeona_mni2.nii.gz"
		fi
	else
	
		roiname=${newroi/=*/""}
		roilist="${newroi}"
	
		inzipname=output_${subject}_fmriclean
		outzipname=output_${subject}_fmriclean_${roiname}
		studydir=${HOME}/output_${subject}_fmriclean
		mkdir -p ${studydir}
		cd ${studydir}

		ziplistfile_todelete=todelete.txt


		aws s3 cp ${s3root}/HCP/hrf_1200.txt ./

		aws s3 cp ${s3root}/HCP/${inzipname}/${inzipname}.zip  ./
		unzip -o ./${inzipname}.zip 
		find ${studydir}/ -type f > ${ziplistfile_todelete}

		rm -f ./${inzipname}.zip

		mnidir=$studydir/$subject/MNINonLinear
		roidir=$mnidir/ROIs
	fi

	roilist_new=""
	for roi in $roilist; do
		roiname=${roi/=*/""}
		roifile=${roi/*=/""}
	
		if [ -e "${roifile}" ]; then
			roilist_new+="${roi},"
			continue
		fi
	
		aws s3 cp ${s3roipath}/$roifile ./
	
		#resample everything to 2mm
		roifile_new=${roifile/.nii.gz/.2.nii.gz} 
		applywarp -i ${studydir}/$roifile  -r $roidir/wmparc.2.nii.gz -o ${studydir}/${roifile_new} --interp=nn
		rm -f ${studydir}/$roifile
	
		roilist_new+="${roiname}=${studydir}/${roifile_new},"
	
		#ls ${studydir}/$roifile >> ${ziplistfile_todelete}
		ls ${studydir}/${roifile_new} >> ${ziplistfile_todelete}
	done
	roilist="${roilist_new}"
fi

opts=
for r in rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST2_RL; do
	
	resultsdir=${studydir}/${subject}/MNINonLinear/Results/${r}
	
	if [ "$do_preproc"  = 1 ]; then 
		mkdir -p ${resultsdir}

		#for the manually downloaded cases that weren't available on S3
		is_manual=$(aws s3 ls ${s3root}/HCP/downloaded_data/${subject}/${r}/${r}_hp2000_clean.nii.gz | wc -l)
		if [ "${is_manual}" = 1 ]; then
			aws s3 cp  ${s3root}/HCP/downloaded_data/${subject}/${r}/${r}_hp2000_clean.nii.gz ${resultsdir}/
		else
			aws s3 --profile hcp cp s3://hcp-openaccess/HCP_1200/${subject}/MNINonLinear/Results/${r}/${r}_hp2000_clean.nii.gz ${resultsdir}/
		fi
		aws s3 --profile hcp cp s3://hcp-openaccess/HCP_1200/${subject}/MNINonLinear/Results/${r}/Movement_Regressors.txt ${resultsdir}/
		aws s3 --profile hcp cp s3://hcp-openaccess/HCP_1200/${subject}/MNINonLinear/Results/${r}/brainmask_fs.2.nii.gz ${resultsdir}/
		aws s3 --profile hcp cp s3://hcp-openaccess/HCP_1200/${subject}/MNINonLinear/Results/${r}/RibbonVolumeToSurfaceMapping/goodvoxels.nii.gz ${resultsdir}/RibbonVolumeToSurfaceMapping/

		find ${studydir}/${subject}/MNINonLinear/Results/${r} -type f >> ${ziplistfile_todelete}
	
		cleanlog=${studydir}/${subject}_${r}_fmriclean.log
		rm -f ${cleanlog}
	
		/bin/date  >> ${cleanlog}
		
		scandir=${mnidir}/Results/${r}
		pigz -p 2 -df ${scandir}/${r}_hp2000_clean.nii.gz
		python $HOME/fmri_outlier_detection.py --input ${scandir}/${r}_hp2000_clean.nii --mask ${scandir}/brainmask_fs.2.nii.gz --motionparam ${scandir}/Movement_Regressors.txt --motionparamtype hcp --connstandard --output ${studydir}/${subject}_${r}_outliers.txt --outputparams ${studydir}/${subject}_${r}_outlier_parameters.mat >> ${cleanlog} 2>&1
	
		python $HOME/fmri_save_confounds.py --input ${scandir}/${r}_hp2000_clean.nii --hcpmnidir ${mnidir} --hcpscanname ${r} --outlierfile ${studydir}/${subject}_${r}_outliers.txt --hrffile ${studydir}/hrf_1200.txt --skipvols 5 --output ${studydir}/${subject}_${r}_fmriclean_confounds.mat >> ${cleanlog} 2>&1

		python $HOME/fmri_save_parcellated_timeseries.py --input ${scandir}/${r}_hp2000_clean.nii --roifile ${roilist} --outbase ${studydir}/${subject}_${r} --outputformat mat >> ${cleanlog} 2>&1
		
		aws s3 sync ${studydir} ${s3root}/HCP/${subject}_fmriclean/ --exclude "*" --include "${subject}_${r}_*confound*" --include "${subject}_${r}_*ts.txt" --include "${subject}_${r}_*_ts.mat" --include "${subject}_${r}_outlier*" --include "${subject}_${r}_fmriclean_*"
		
		rm -rf ${scandir}/
	else
		
		#need to exclude the old version of _ts.mat with "fmriclean" in the filename
		aws s3 sync ${s3root}/HCP/${subject}_fmriclean ${studydir}/ --exclude "*" --include "${subject}_${r}_*_ts.mat" --exclude "${subject}_${r}_fmriclean*"
		aws s3 sync ${s3root}/HCP/${subject}_fmriclean ${studydir}/ --exclude "*" --include "${subject}_${r}_fmriclean_confounds.mat"
		
		cleanlog=${studydir}/${subject}_${r}_fmriclean_connmat.log
		rm -f ${cleanlog}
		/bin/date >> ${cleanlog}
	fi

	if [ "$do_connmeasure"  = 1 ]; then
		bpfarg="--lowfreq 0.008 --highfreq 0.09"
		hpfarg="--lowfreq 0.008"
		for filtargname in "nofilt@--nocompcor" "bpf@$bpfarg" "hpf@$hpfarg"; do
			for gsrarg in "" "--gsr"; do
				filtname=${filtargname/@*/""}
				filtarg=${filtargname/*@/""}
				python $HOME/fmri_clean_parcellated_timeseries.py --inputpattern "${studydir}/${subject}_${r}_%s_ts.mat" --roilist ${roilist} --confoundfile ${studydir}/${subject}_${r}_fmriclean_confounds.mat --filterstrategy connregbp $filtarg $gsrarg --outbase ${studydir}/${subject}_${r}_fmriclean_${filtname} --skipvols 5 --connmeasure precision partialcorrelation correlation covariance --outputformat mat >> ${cleanlog} 2>&1
			done
		done
		aws s3 sync ${studydir} ${s3root}/HCP/${subject}_fmriclean/ --exclude "*" --include "${subject}_${r}_fmriclean_*_ts.txt" --include "${subject}_${r}_*FC*.mat" --include "${subject}_${r}_*FC*.mat" --include "${subject}_${r}_fmriclean_*.log"
	fi

	rm -f ${studydir}/${subject}_${r}_*
	
	#outzipname_scan=${studydir}/${subject}_${r}_fmriclean.zip
	#zip ${outzipname_scan} ${subject}_${r}_outlier* ${subject}_${r}_fmriclean_*
done

#for f in $( cat ${ziplistfile_todelete} ); do
#	rm -f $f
#done

#zip -r ${outzipname}.zip * -x "*/" > ${outzipname}.zip.log

#cd $(dirname $studydir)
#aws s3 sync ${studydir}/ ${s3root}/HCP/$(basename $studydir) --exclude "*" --include "*_outlier*"

#aws s3 sync ${studydir} ${s3root}/HCP/$(basename $studydir) --exclude "*" --include "${outzipname}.zip" --include "${outzipname}*.tar" --include "*.log"

cd $HOME
rm -rf ${studydir}
