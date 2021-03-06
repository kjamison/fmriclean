#!/bin/bash

set -e
set -x


do_preproc=1
do_connmeasure=0
do_concat=0

#set these after determining if its HCP or not first
#bpfarg="--lowfreq 0.008 --highfreq 0.09"
#hpfarg="--lowfreq 0.008"

#bpfarg="--lowfreq 0.01 --highfreq 0.15"
#hpfarg="--lowfreq 0.01"
#skipvolarg="--skipvols 10"

#connarg="--connmeasure precision partialcorrelation correlation covariance"
connarg="--connmeasure covariance --shrinkage 0"

isAWSpipeline=0
if [ -e ${HOME}/startup_tags.json ]; then
    isAWSpipeline=1
fi

#################################


#tagfile=${HOME}/startup_tags.json
#instanceid=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id)
#region=$(curl --silent --fail http://169.254.169.254/latest/dynamic/instance-identity/document/ | grep region | cut -d\" -f4)
#aws ec2 describe-tags --region $region --filter "Name=resource-id,Values=$instanceid" | jq --raw-output ".Tags[]" > ${tagfile}

#Study=$(jq --raw-output 'select(.Key=="Study") | .Value' ${tagfile} | head -n1)
#Subject=$(jq --raw-output 'select(.Key=="Subject") | .Value' ${tagfile} | head -n1)
#ScanName=$(jq --raw-output 'select(.Key=="ScanName") | .Value' ${tagfile} | head -n1)
#PipelineStep=$(jq --raw-output 'select(.Key=="PipelineStep") | .Value' ${tagfile} | head -n1)

s3root=s3://kuceyeski-wcm-temp/kwj2001

if [ "x$Subject" = "x" ]; then
	subject=$1
	Study=HCP
	if [ "$2" = "-study" ]; then
		Study="$3"
		shift; shift;
	fi
	if [ "$2" = "-onlyconn" ]; then
		newroi=""
		do_preproc=0
		do_connmeasure=1
		newroi="$3"
	elif [ "$2" = "-conn" ]; then
		newroi=""
		do_connmeasure=1
		newroi="$3"
	elif [ "$2" = "-concat" ]; then
		newroi=""
		do_preproc=0
		do_connmeasure=0
		do_concat=1
		newroi="$3"
    elif [ "$2" = "-conn_and_concat" ]; then
        do_connmeasure=1
        do_concat=1
        newroi="$3"
	elif [ "$2" = "-concatday" ]; then
		newroi=""
		do_preproc=0
		do_connmeasure=0
		do_concat=2
		newroi="$3"
	elif [ "$2" = "-args" ]; then
		extra_args="$3"
		newroi="$4"
	else
		newroi="$2"
	fi
	
	
else
	subject=${Subject}
fi

s3studyroot=${s3root}/$Study

#results are uploaded to s3root/HCP/${subject}${uploadsuffix}
downloadsuffix="_fmriclean"
uploadsuffix="_fmriclean"
outputprefix=""

#################################

#try to stop python/numpy from secretly using extra cores sometimes
if [ "$isAWSpipeline" = 0 ]; then
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
fi

export FMRICLEANDIR=$HOME/fmriclean

export FSLDIR=$HOME/fsl
#aws s3 sync s3://kuceyeski-wcm-temp/kwj2001/fsl/ $FSLDIR/ && chmod -R +x $FSLDIR/bin
export FSLOUTPUTTYPE=NIFTI_GZ
source $FSLDIR/etc/fslconf/fsl.sh

mrtrixdir=${HOME}/mrtrix3/bin
#aws s3 cp s3://kuceyeski-wcm-temp/kwj2001/mrtrix3.tar.gz $HOME/ && (cd $HOME; tar -xf mrtrix3.tar.gz ) && rm -f $HOME/mrtrix3.tar.gz && chmod -R +x $HOME/mrtrix3/bin

if [ ! -e $FMRICLEANDIR/fmri_clean_parcellated_timeseries.py ]; then
    ( cd $HOME; git clone https://github.com/kjamison/fmriclean.git )
fi

export PATH=$PATH:$FSLDIR/bin
export PATH=$PATH:$mrtrixdir

export CONDAPATH=$HOME/miniconda
export PATH=$CONDAPATH/bin:$PATH
source $CONDAPATH/etc/profile.d/conda.sh
conda activate base

s3roipath=s3://kuceyeski-wcm-temp/kwj2001/nemo2/nemo_atlases




#preproc for ALL atlases (fs86,cc200,cc400,shen268,cocoA,cocoB) takes 22min/subj when using 16 jobs on m5a.8xlarge (32 vCPU), or 8 jobs on m5a.4xlarge(16 CPU), etc..
# (note: if we use more than half the vCPU things start taking a lot longer)
#so running a batch of 48 (3 separate 8xlarge instances) takes (997/48)*22/60=7.6hrs

#connmeasure takes ~3 min for ALL atlases, flavors: atlas=(fs86,cc200,cc400,shen268,cocoA,cocoB), gsr=(yes,no), filt=(nofilt,bpf,hpf), conn=(corr,pcorr,cov,prec)
#and can run on all cores of a single 8xlarge instace, so (997/31)*3/60=95min

if [ "${Study}" = "HCP" ]; then
	ScanList="rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST2_RL"
	bpfarg="--lowfreq 0.01 --highfreq 0.15"
	hpfarg="--lowfreq 0.01"
	skipvolarg="--skipvols 10"
else
	#HCP didn't use the output_ thing, but we should use it for other studies for consistency
	uploadprefix="output_"
	ScanList="rfMRI_REST_AP rfMRI_REST_PA"
	bpfarg="--lowfreq 0.008 --highfreq 0.09"
	hpfarg="--lowfreq 0.008"
	skipvolarg="--skipvols 5"
fi


if [[ "$subject" == *_Retest ]]; then
	hcps3dir=HCP_Retest
	hcps3subject=${subject/_Retest/""}
else
	hcps3dir=HCP_1200
	hcps3subject=${subject}
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
	
		if [ "${Study}" = "HCP" ]; then
			aws s3 --profile hcp cp s3://hcp-openaccess/${hcps3dir}/${hcps3subject}/MNINonLinear/ROIs/wmparc.2.nii.gz $roidir/
			aws s3 --profile hcp cp s3://hcp-openaccess/${hcps3dir}/${hcps3subject}/MNINonLinear/ROIs/ROIs.2.nii.gz $roidir/
			aws s3 --profile hcp cp s3://hcp-openaccess/${hcps3dir}/${hcps3subject}/MNINonLinear/ribbon.nii.gz $mnidir/
			aws s3 --profile hcp cp s3://hcp-openaccess/${hcps3dir}/${hcps3subject}/MNINonLinear/aparc+aseg.nii.gz $mnidir/
			aws s3 --profile hcp cp s3://hcp-openaccess/${hcps3dir}/${hcps3subject}/MNINonLinear/aparc.a2009s+aseg.nii.gz $mnidir/
	
			aws s3 cp ${s3studyroot}/${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_cerebSUIT.zip ./
			aws s3 cp ${s3studyroot}/${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7.zip ./
			unzip -qo ${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_cerebSUIT.zip ${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_cerebSUIT_seq.mni2mm.nii.gz
			unzip -qo ${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7.zip ${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7_seq.mni2mm.nii.gz
			rm -f ${subject}_hcpmmp_aseg_*.zip
		else
			aws s3 cp ${s3studyroot}/output_${subject}/${subject}_hcpstruct.zip ./
			unzip -qo ${subject}_hcpstruct.zip "${subject}/MNINonLinear/*"
			rm -rf ${subject}_hcpstruct.zip
            
            aws s3 sync ${s3studyroot}/output_${subject} ./ --exclude "*" --include "${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7*.zip"
            for z in ${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7*.zip; do
    			unzip -qo $z "${subject}_hcpmmp_aseg_aal3_sgmfix_thalFS7*_seq.mni2mm.nii.gz"
                rm -f $z
            done

		fi
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

		aws s3 cp ${s3studyroot}/${inzipname}/${inzipname}.zip  ./
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
	
		aws s3 cp ${s3roipath}/$roifile ./ || continue
	
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


inputpattern_list=""
confoundfile_list=""

opts=
for r in ${ScanList}; do
	
	resultsdir=${studydir}/${subject}/MNINonLinear/Results/${r}

	if [ "$do_preproc"  = 1 ]; then 
		mkdir -p ${resultsdir}

		if [ "$Study" = "HCP" ]; then
			scanfile=${r}_hp2000_clean
			
			#for the manually downloaded cases that weren't available on S3
			is_manual=$(aws s3 ls ${s3root}/HCP/downloaded_data/${subject}/${r}/${r}_hp2000_clean.nii.gz | wc -l)
			if [ "${is_manual}" = 1 ]; then
				aws s3 cp  ${s3studyroot}/downloaded_data/${subject}/${r}/${r}_hp2000_clean.nii.gz ${resultsdir}/
			else
				aws s3 --profile hcp cp s3://hcp-openaccess/${hcps3dir}/${hcps3subject}/MNINonLinear/Results/${r}/${r}_hp2000_clean.nii.gz ${resultsdir}/
			fi
			aws s3 --profile hcp cp s3://hcp-openaccess/${hcps3dir}/${hcps3subject}/MNINonLinear/Results/${r}/Movement_Regressors.txt ${resultsdir}/
			aws s3 --profile hcp cp s3://hcp-openaccess/${hcps3dir}/${hcps3subject}/MNINonLinear/Results/${r}/brainmask_fs.2.nii.gz ${resultsdir}/
			aws s3 --profile hcp cp s3://hcp-openaccess/${hcps3dir}/${hcps3subject}/MNINonLinear/Results/${r}/RibbonVolumeToSurfaceMapping/goodvoxels.nii.gz ${resultsdir}/RibbonVolumeToSurfaceMapping/
		else
			aws s3 cp ${s3studyroot}/output_${subject}_${r}/${subject}_${r}_hcpfunc.zip ./
			unzip -qo ${subject}_${r}_hcpfunc.zip "${subject}/MNINonLinear/Results/${r}/*"
			rm -f ${subject}_${r}_hcpfunc.zip
			
			scanfile=${r}
		fi
		find ${studydir}/${subject}/MNINonLinear/Results/${r} -type f >> ${ziplistfile_todelete}
	
		cleanlog=${studydir}/${subject}_${r}_fmriclean.log
		rm -f ${cleanlog}
	
		/bin/date  >> ${cleanlog}
		
		scandir=${mnidir}/Results/${r}
		pigz -p 2 -df ${scandir}/${scanfile}.nii.gz
		python $FMRICLEANDIR/fmri_outlier_detection.py --input ${scandir}/${scanfile}.nii --mask ${scandir}/brainmask_fs.2.nii.gz --motionparam ${scandir}/Movement_Regressors.txt --motionparamtype hcp --connstandard --output ${studydir}/${subject}_${r}_outliers.txt --outputparams ${studydir}/${subject}_${r}_outlier_parameters.mat >> ${cleanlog} 2>&1
	
		python $FMRICLEANDIR/fmri_save_confounds.py --input ${scandir}/${scanfile}.nii --hcpmnidir ${mnidir} --hcpscanname ${r} --outlierfile ${studydir}/${subject}_${r}_outliers.txt --skipvols 5 --output ${studydir}/${subject}_${r}_fmriclean_confounds.mat >> ${cleanlog} 2>&1

		python $FMRICLEANDIR/fmri_save_parcellated_timeseries.py --input ${scandir}/${scanfile}.nii --roifile ${roilist} --outbase ${studydir}/${subject}_${r} --outputformat mat >> ${cleanlog} 2>&1
		
		aws s3 sync ${studydir} ${s3studyroot}/${uploadprefix}${subject}${uploadsuffix}/ --exclude "*" --include "${subject}_${r}_*confound*" --include "${subject}_${r}_*ts.txt" --include "${subject}_${r}_*_ts.mat" --include "${subject}_${r}_outlier*" --include "${subject}_${r}_fmriclean_*"
		
		rm -rf ${scandir}/
	else
		
		#need to exclude the old version of _ts.mat with "fmriclean" in the filename
		aws s3 sync ${s3studyroot}/${subject}${downloadsuffix} ${studydir}/ --exclude "*" --include "${subject}_${r}_*_ts.mat" --exclude "${subject}_${r}_fmriclean*"
		aws s3 sync ${s3studyroot}/${subject}${downloadsuffix} ${studydir}/ --exclude "*" --include "${subject}_${r}_fmriclean_confounds.mat"
		
		cleanlog=${studydir}/${subject}_${r}_fmriclean_connmat.log
		rm -f ${cleanlog}
		/bin/date >> ${cleanlog}
	fi

	if [ "$do_connmeasure"  = 1 ]; then
		for filtargname in "nofilt@--nocompcor" "bpf@$bpfarg" "hpf@$hpfarg"; do
			for gsrarg in "" "--gsr"; do
				filtname=${filtargname/@*/""}
				filtarg=${filtargname/*@/""}
				python $FMRICLEANDIR/fmri_clean_parcellated_timeseries.py --inputpattern "${studydir}/${subject}_${r}_%s_ts.mat" --roilist ${roilist} --confoundfile ${studydir}/${subject}_${r}_fmriclean_confounds.mat $filtarg $gsrarg --outbase ${studydir}/${subject}_${r}_fmriclean_${filtname} ${skipvolarg} ${connarg} --outputformat mat --sequentialroi >> ${cleanlog} 2>&1
			done
		done
		aws s3 sync ${studydir} ${s3studyroot}/${uploadprefix}${subject}${uploadsuffix}/ --exclude "*" --include "${subject}_${r}_fmriclean_*_ts.txt" --include "${subject}_${r}_fmriclean_*_ts.mat" --include "${subject}_${r}_*FC*.mat" --include "${subject}_${r}_fmriclean_*.log"
	fi
	
	inputpattern_list+=" ${studydir}/${subject}_${r}_%s_ts.mat"
	confoundfile_list+=" ${studydir}/${subject}_${r}_fmriclean_confounds.mat"
	
	if [ "${do_concat}" = "0" ]; then
		rm -f ${studydir}/${subject}_${r}_*
	fi
	#outzipname_scan=${studydir}/${subject}_${r}_fmriclean.zip
	#zip ${outzipname_scan} ${subject}_${r}_outlier* ${subject}_${r}_fmriclean_*
done

if [ "${do_concat}" = "1" ]; then
	cleanlog=${studydir}/${subject}_concat_fmriclean_connmat.log
	rm -f ${cleanlog}
	/bin/date >> ${cleanlog}

	#for filtargname in "nofilt@--nocompcor" "bpf@$bpfarg --filterstrategy connregbp" "hpf@$hpfarg --filterstrategy connregbp" "bpfseq@$bpfarg --filterstrategy seq" "hpfseq@$hpfarg --filterstrategy seq"; do
	#for filtargname in "nofilt@--nocompcor" "bpf@$bpfarg --filterstrategy connregbp" "hpf@$hpfarg --filterstrategy connregbp" "bpfpar@$bpfarg --filterstrategy parallel" "hpfpar@$hpfarg --filterstrategy parallel"; do
	for filtargname in "nofilt@--nocompcor" "bpf@$bpfarg --filterstrategy connregbp" "hpf@$hpfarg --filterstrategy connregbp"; do
		for gsrarg in "" "--gsr"; do
			filtname=${filtargname/@*/""}
			filtarg=${filtargname/*@/""}
			python $FMRICLEANDIR/fmri_clean_parcellated_timeseries.py --inputpattern ${inputpattern_list} --roilist ${roilist} --confoundfile ${confoundfile_list} $filtarg $gsrarg --outbase ${studydir}/${subject}_concat_fmriclean_${filtname} ${skipvolarg} --outputformat mat --sequentialroi --concat ${connarg}  >> ${cleanlog} 2>&1
		done
	done
	
	aws s3 sync ${studydir} ${s3studyroot}/${uploadprefix}${subject}${uploadsuffix}/ --exclude "*" --include "${subject}_concat_fmriclean_*_ts.txt" --include "${subject}_concat_fmriclean_*_ts.mat" --include "${subject}_concat_*FC*.mat" --include "${subject}_concat_fmriclean_*.log"
	
elif [ "${do_concat}" = "2" ]; then
	inputpattern_list_alldays="${inputpattern_list}"
	confoundfile_list_alldays="${confoundfile_list}"
	for d in day1 day2; do
		if [ "$d" = "day1" ]; then
			inputpattern_list=$(echo "${inputpattern_list_alldays}" | tr " " "\n" | grep rfMRI_REST1 | tr "\n" " ")
			confoundfile_list=$(echo "${confoundfile_list_alldays}" | tr " " "\n" | grep rfMRI_REST1 | tr "\n" " ")
		else
			inputpattern_list=$(echo "${inputpattern_list_alldays}" | tr " " "\n" | grep rfMRI_REST2 | tr "\n" " ")
			confoundfile_list=$(echo "${confoundfile_list_alldays}" | tr " " "\n" | grep rfMRI_REST2 | tr "\n" " ")
		fi

		
		cleanlog=${studydir}/${subject}_concat${d}_fmriclean_connmat.log
		rm -f ${cleanlog}
		/bin/date >> ${cleanlog}

		#for filtargname in "nofilt@--nocompcor" "bpf@$bpfarg --filterstrategy connregbp" "hpf@$hpfarg --filterstrategy connregbp" "bpfseq@$bpfarg --filterstrategy seq" "hpfseq@$hpfarg --filterstrategy seq"; do
		#for filtargname in "nofilt@--nocompcor" "bpf@$bpfarg --filterstrategy connregbp" "hpf@$hpfarg --filterstrategy connregbp" "bpfpar@$bpfarg --filterstrategy parallel" "hpfpar@$hpfarg --filterstrategy parallel"; do
		for filtargname in "nofilt@--nocompcor" "bpf@$bpfarg --filterstrategy connregbp" "hpf@$hpfarg --filterstrategy connregbp"; do
			for gsrarg in "" "--gsr"; do
				filtname=${filtargname/@*/""}
				filtarg=${filtargname/*@/""}
				python $FMRICLEANDIR/fmri_clean_parcellated_timeseries.py --inputpattern ${inputpattern_list} --roilist ${roilist} --confoundfile ${confoundfile_list} $filtarg $gsrarg --outbase ${studydir}/${subject}_concat${d}_fmriclean_${filtname} ${skipvolarg} --outputformat mat --sequentialroi --concat ${connarg}  >> ${cleanlog} 2>&1
			done
		done
	
		aws s3 sync ${studydir} ${s3studyroot}/${uploadprefix}${subject}${uploadsuffix}/ --exclude "*" --include "${subject}_concat${d}_fmriclean_*_ts.txt" --include "${subject}_conca${d}t_fmriclean_*_ts.mat" --include "${subject}_concat${d}_*FC*.mat" --include "${subject}_concat${d}_fmriclean_*.log"
	done
fi

#for f in $( cat ${ziplistfile_todelete} ); do
#	rm -f $f
#done

#zip -r ${outzipname}.zip * -x "*/" > ${outzipname}.zip.log

#cd $(dirname $studydir)
#aws s3 sync ${studydir}/ ${s3root}/HCP/$(basename $studydir) --exclude "*" --include "*_outlier*"

#aws s3 sync ${studydir} ${s3root}/HCP/$(basename $studydir) --exclude "*" --include "${outzipname}.zip" --include "${outzipname}*.tar" --include "*.log"

cd $HOME
rm -rf ${studydir}
