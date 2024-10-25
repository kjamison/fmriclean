import numpy as np
import sys
import argparse
from scipy.io import loadmat,savemat
import re
from matplotlib import pyplot as plt

from utils import *

#python fmri_regularized_inverse.py --combinedinput ~/Research/HCP/fc_fs86_FCcov_hpf_993subj.mat --output_target_precision testtarg_fs86_hpf.mat
#python fmri_regularized_inverse.py --target_precision_file testtarg_fs86_hpf.mat --combinedinput ~/Research/MS/msdata_103subj_FC_fs86_hpf.mat --outputfig test.png --verbose --saveprec --savepcorr

def argument_parse_reginv(argv):
    parser=argparse.ArgumentParser(description='Regularized Inverse Covariance Estimation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #option 1: --inputpattern with {subject} placeholders 
    #          --subjects with list of subjects in command line
    #           OR --subjectfile with a file containing a list of subjects
    #option 2: --input with list of input files
    #option 3: --combinedinput with a single input file containing all subjects

    input_group=parser.add_argument_group('FC input options')
    input_group.add_argument('--inputpattern',action='store',dest='inputpattern',help='Input filename pattern with {subject} placeholders. Eg "myfolder/{subject}_FC.mat"')
    input_group.add_argument('--input','--inputs',action='append',dest='inputlist',nargs='*',help='List of input files')
    input_group.add_argument('--combinedinput',action='store',dest='combinedinput',help='Single .mat input file containing all subjects, with "C" field with cell array of connectivity data for each subject')
    
    input_group.add_argument('--subjects',action='append',dest='subjectlist',nargs='*',help='List of subject IDs for inputpattern OR to  be used to subset combinedinput')
    input_group.add_argument('--subjectfile',action='store',dest='subjectfile',help='plaintext (or .mat with "subject" field) with list of subject IDs for inputpatternor to subset combinedinput')
    
    reg_group=parser.add_argument_group('Regularization options')
    reg_group.add_argument('--lambda_range',action='store',dest='lambda_range',nargs=2,type=float,default=[0,1],help='Range of lambda values to search')
    reg_group.add_argument('--lambda_steps',action='store',dest='lambda_steps',type=int,default=10,help='Number of steps in lambda search')
    reg_group.add_argument('--lambda_loops',action='store',dest='lambda_loops',type=int,default=5,help='Number of loops to expand lambda search')
    reg_group.add_argument('--lambda_expand_loops',action='store',dest='lambda_expand_loops',type=int,default=5,help='Number of times to expand lambda search range')
    reg_group.add_argument('--lambda_expand_scale',action='store',dest='lambda_expand_scale',type=float,default=.5,help='Scale factor for expanding lambda search range')
    reg_group.add_argument('--target_precision_file',action='store',dest='target_precision',help='File with target precision matrix for regularization (otherwise compute mean precision matrix from data)')
    reg_group.add_argument('--lambda',action='store',dest='lambda_value',type=float,help='Single lambda value to use (overrides lambda search)')
    reg_group.add_argument('--lambda_round_scale',action='store',dest='lambda_round_scale',type=float,default=.01,help='Round lambda value to nearest multiple of this scale')
    
    output_group=parser.add_argument_group('Output options')
    output_group.add_argument('--output_target_precision',action='store',dest='output_target_precision',help='Output unregularized mean precision matrix (as target for others), then exit')
    output_group.add_argument('--outputfig',action='store',dest='outputfig',help='Output filename for lambda search plot')
    output_group.add_argument('--saveprec',action='store_true')
    output_group.add_argument('--savepcorr',action='store_true')
    
    parser.add_argument('--verbose', action='store_true',dest='verbose')                                                 
    parser.add_argument('--version', action='version',version=package_version_dict(as_string=True))
    
    return parser.parse_args(argv)

def reginv_file_extension(filename):
    extension_list=['.mat','.txt','.csv','.tsv']
    for ext in extension_list:
        if filename.lower().endswith(ext.lower()):
            return ext
    return None

def reginv_load_inputs(combinedinput_arg=None, inputpattern=None, subjectlist_arg=None, subjectfile=None, inputlist_arg=None):
    M_allsubj={}
    
    input_info={}
    
    subjectlist=[]
    if subjectfile:
        if reginv_file_extension(subjectfile)=='.mat':
            Msubj=loadmat(subjectfile,simplify_cells=True)
            if 'subject' in Msubj:
                subjectlist=[s for s in Msubj['subject']]
            elif 'subjects' in Msubj:
                subjectlist=[s for s in Msubj['subjects']]
            else:
                raise Exception("No subject list found in %s. Must include field 'subject' or 'subjects'" % (subjectfile))
        else:
            with open(subjectfile,'r') as fid:
                subjectlist=[x.strip() for x in fid.readlines]
    elif subjectlist_arg:
        subjectlist=[s for s in subjectlist_arg]
    subjectlist=[str(s) for s in subjectlist] #in case some parsing turned them into integers
                
    if combinedinput_arg:
        if reginv_file_extension(combinedinput_arg) != '.mat':
            raise Exception("Combined input must be a .mat file")
        M=loadmat(combinedinput_arg,simplify_cells=True)
        mfield_search=['C','FC','data']
        mfield_search=[x.upper() for x in mfield_search]
        mfield=None
        for k in M:
            if k.upper() in mfield_search:
                mfield=k
                break
        if mfield is None:
            raise Exception("No field found in %s with names %s" % (combinedinput_arg,",".join(mfield_search)))
        #build new M_allsubj with C field for data
        #and add any additional fields from the input file that are not __* internal fields
        M_allsubj['C']=[x for x in M[mfield]]
        for k in M:
            if k.startswith("__"):
                continue
            if k == mfield:
                continue
            M_allsubj[k]=M[k]
        
        if subjectlist:
            msubjfield=None
            if 'subject' in M_allsubj:
                msubjfield='subject'
            elif 'subjects' in M_allsubj:
                msubjfield='subjects'
            if msubjfield:
                M_allsubj[msubjfield]=[str(s) for s in M_allsubj[msubjfield]] #in case some parsing turned them into integers
                #filter out subjects not in subjectlist
                subjmask=[s in subjectlist for s in M_allsubj[msubjfield]]
                for k in M_allsubj:
                    if k.startswith("__"):
                        continue
                    if len(M_allsubj[k])==len(subjmask):
                        M_allsubj[k]=[M_allsubj[k][i] for i in range(len(subjmask)) if subjmask[i]]
                          
        input_info['combined_input_file']=combinedinput_arg
    else:
        inputlist=[]
        if inputpattern:
            inputpattern=re.sub('\{(s|subj|subject)\}','{SUBJECT}',inputpattern,flags=re.IGNORECASE)
            if not subjectlist:
                raise Exception("Must specify --subjects or --subjectfile with --inputpattern")
            
            inputlist=[inputpattern.format(SUBJECT=s) for s in subjectlist]
        elif inputlist_arg:
            inputlist=[s for s in inputlist_arg]
            subjectlist=[s for s in inputlist_arg] #if subjectlist was not provided, just save the filename as the subject name
            
        M_allsubj={}
        M_allsubj['subject']=subjectlist
        M_allsubj['C']=[]
        for inputfile in inputlist:
            conn_dict=load_connmatrix(inputfile)
            M_allsubj['C'].append(conn_dict['C'])
            
        input_info['subjects_list']=subjectlist
        input_info['input_file_list']=inputlist
            
    return M_allsubj, input_info


def reginv_save_outputs(M_allsubj, input_info, output_suffix='_new', reg_info_dict={}):
    if 'combined_input_file' in input_info:
        if reginv_file_extension(input_info['combined_input_file'])=='.mat':
            outname=input_info['combined_input_file'][:-4]+output_suffix+'.mat'
        else:
            raise Exception("Combined input must be a .mat file")
        
        #need to convert C to a cell array in mat file
        Cnew=np.empty((len(M_allsubj['C'])),dtype=object)
        Cnew[:]=[C for C in M_allsubj['C']]
        
        M_new={}
        for k in M_allsubj:
            if k=='C':
                M_new['C']=Cnew
            else:
                M_new[k]=M_allsubj[k]
        for k in reg_info_dict:
            M_new[k]=reg_info_dict[k]
        
        savemat(outname,M_new,format='5',do_compression=True)
        
        print("Saved %s (%d x [%dx%d])" % (outname, len(M_allsubj['C']),M_allsubj['C'][0].shape[0],M_allsubj['C'][0].shape[1]))
        #save as outname
    else:
        #save each of the input files with the output suffix
        for i,inputfile in enumerate(input_info['input_file_list']):
            inputsubj=input_info['subjects_list'][i]
            fext=reginv_file_extension(inputfile)
            if fext is None:
                raise Exception("Input file format not recognized for %s" % (inputfile))
            elif fext=='.mat':
                outname_noext=outname=inputfile[:-4]+output_suffix
                outname_ext=fext
            else:
                outname_noext=inputfile[:-len(fext)]+output_suffix
                outname_ext=fext
            outdict=reg_info_dict.copy()
            outdict['C']=M_allsubj['C'][i]
            outname, outshapestr=save_connmatrix(filename_noext=outname_noext,outputformat=outname_ext,output_dict=outdict)
            print("Saved %s (%s)" % (outname,outshapestr))
    
def invert_tikhonov(C,lam):
    return np.linalg.inv(C+(lam*np.trace(C)/C.shape[0])*np.eye(C.shape[0]))

def prec2pcorr(Cprec):
    dsqrt=np.diag(np.sqrt(np.diag(np.abs(Cprec))))
    Cpcorr=dsqrt @ (-Cprec) @ dsqrt
    np.fill_diagonal(Cpcorr,0)
    return Cpcorr

def find_optimal_precision_lambda(C_list, C_targ, lambda_range=[0,1], lambda_steps=10, lambda_loops=3, lambda_expand_loops=5, lambda_expand_scale=.5, normtype='euclidean', verbose=False):
    mask=np.triu(np.ones(C_targ.shape),1)>0 #skip diag
    
    if normtype=='euclidean':
        normfun=lambda x: np.linalg.norm(x,ord=2)
    else:
        normfun=lambda x: np.linalg.norm(x,ord=normtype)
    
    lambda_match_tol=1e-6 #avoid computing for the same lambda twice
    
    lambda_full=[]
    reg_err_full=[]
    
    max_loop_count=lambda_loops
    iloop=0
    lambda_opt=None
    while iloop<max_loop_count:
        lambda_list=np.linspace(lambda_range[0],lambda_range[1],lambda_steps)
        #define additional ranges for expanding the grid to the left or right
        #dv=lambda_list[1]-lambda_list[0]
        dv=(lambda_range[1]-lambda_range[0])*lambda_expand_scale
        lambda_range_shifted_L=[lambda_range[0]-dv,lambda_range[1]-dv]
        lambda_range_shifted_R=[lambda_range[0]+dv,lambda_range[1]+dv]
        
        reg_err=np.zeros(len(lambda_list))
        for i,lam in enumerate(lambda_list):
            i_match=[ii for ii,l in enumerate(lambda_full) if np.isclose(l,lam,atol=lambda_match_tol)]
            if(len(i_match)>0):
                reg_err[i]=reg_err_full[i_match[0]]
                continue
            Cprec=[invert_tikhonov(C,lam) for C in C_list]
            reg_err[i]=np.mean([normfun(C[mask]-C_targ[mask]) for C in Cprec])
        midx=np.argmin(reg_err)
        lambda_opt=lambda_list[midx]
        did_expand_range=False
        if midx==0:
            if lambda_expand_loops>0:
                did_expand_range=True
                #max_loop_count+=2
                iloop=-1
                lambda_expand_loops-=1
                new_lambda_range=lambda_range_shifted_L
            else:
                new_lambda_range=[lambda_range[0],lambda_list[1]]
        elif midx==len(lambda_list)-1:
            if lambda_expand_loops>0:
                did_expand_range=True
                #max_loop_count+=2
                iloop=-1
                lambda_expand_loops-=1
                new_lambda_range=lambda_range_shifted_R
            else:
                new_lambda_range=[lambda_range[-2],lambda_list[-1]]
        else:
            new_lambda_range=[lambda_list[midx-1],lambda_list[midx+1]]
        
        lambda_full=np.concatenate([lambda_full,lambda_list])
        reg_err_full=np.concatenate([reg_err_full,reg_err])
        
        if verbose:
            if did_expand_range:
                print(f'loop {iloop+1}(shift range), range {lambda_range[0]}-{lambda_range[1]}, best lambda={lambda_opt}, best err={reg_err[midx]}')
            else:
                print(f'loop {iloop+1}, range {lambda_range[0]}-{lambda_range[1]}, best lambda={lambda_opt}, best err={reg_err[midx]}')
        
        lambda_range=new_lambda_range
        iloop+=1
    
    #just get the unique, sorted lambda values and their errors
    uidx=np.unique(lambda_full,return_index=True)[1]
    lambda_full=lambda_full[uidx]
    reg_err_full=reg_err_full[uidx]
    sidx=np.argsort(lambda_full)
    lambda_full=lambda_full[sidx]
    reg_err_full=reg_err_full[sidx]
    
    midx=np.argmin(reg_err_full)
    lambda_opt=lambda_full[midx]
    
    lambda_search_info={'lambda':lambda_full,'reg_err':reg_err_full, 'lambda_opt':lambda_opt, 'reg_err_opt':reg_err_full[midx]}
    return lambda_opt, lambda_search_info
    
def run_fmri_regularized_inverse(argv):
    args=argument_parse_reginv(argv)

    inputpattern=args.inputpattern
    subjectlist_arg=flatarglist(args.subjectlist)
    subjectfile=args.subjectfile
    
    inputlist_arg=flatarglist(args.inputlist)
    combinedinput_arg=args.combinedinput
    
    output_unreg_precision_arg=args.output_target_precision
    target_precision_arg=args.target_precision
    
    output_fig_filename=args.outputfig
    
    #######
    lambda_range=args.lambda_range
    lambda_steps=args.lambda_steps
    lambda_loops=args.lambda_loops
    lambda_expand_loops=args.lambda_expand_loops
    lambda_expand_scale=args.lambda_expand_scale
    lambda_range=[min(lambda_range),max(lambda_range)]
    lambda_range[1]=max(lambda_range[1],lambda_range[0]+0.01)
    lambda_expand_scale=min(.9,max(.1,lambda_expand_scale))
    
    lambda_round_scale=args.lambda_round_scale
    lambda_value_arg=args.lambda_value
    #######
    verbose=args.verbose
    
    #load input data
    M_allsubj, input_info=reginv_load_inputs(combinedinput_arg, inputpattern, subjectlist_arg, subjectfile, inputlist_arg)
    subject_count=len(M_allsubj['C'])
    C_shape=M_allsubj['C'][0].shape
    print("Loaded input data (%d subjects, %dx%d)" % (subject_count,C_shape[0],C_shape[1]))
    
    target_precision=None
    #if target precision matrix was provided, just load it
    if target_precision_arg:
        print("Using target precision matrix from %s" % (target_precision_arg))
        target_precision=load_connmatrix(target_precision_arg)['C']
    

    if lambda_value_arg is None:

        #otherwise, compute the mean precision matrix for input data
        if target_precision is None:
            inv_success=False
            for invtype in ['inv','pinv']:
                if invtype=='inv':
                    invfun=np.linalg.inv
                elif invtype=='pinv':
                    invfun=np.linalg.pinv
                target_precision=np.zeros(C_shape)
                inv_success=True
                for C in M_allsubj['C']:
                    try:
                        target_precision+=invfun(C)
                    except np.linalg.LinAlgError:
                        print("Singular matrix for %s(%dx%d)" % (invtype,C_shape[0],C_shape[1]))
                        inv_success=False
                        break
                target_precision/=subject_count
                if inv_success:
                    break
            if not inv_success:
                raise Exception("Unable to compute inv or pinv for all subjects")
            
            target_invtype=invtype
            if output_unreg_precision_arg:
                if output_unreg_precision_arg.lower().endswith(".mat"):
                    output_unreg_precision_arg=output_unreg_precision_arg[:-4]
                unreg_outname, unreg_outshapestr=save_connmatrix(filename_noext=output_unreg_precision_arg,outputformat='mat',output_dict={'C':target_precision, 'subject':M_allsubj['subject'], 'cov_estimator':target_invtype})
                print("Saved unregularized precision matrix to %s (%s)" % (unreg_outname,unreg_outshapestr))
                sys.exit(0)
        
        print("Searching for optimal lambda for regularization...")
        lambda_opt, lambda_search_info = find_optimal_precision_lambda(M_allsubj['C'], 
                                                                    target_precision,
                                                                    lambda_range=lambda_range, 
                                                                    lambda_steps=lambda_steps,
                                                                    lambda_loops=lambda_loops,
                                                                    lambda_expand_loops=lambda_expand_loops,
                                                                    lambda_expand_scale=lambda_expand_scale,
                                                                    verbose=verbose)
        
        print("lambda_opt: %f, reg_err_opt: %f" % (lambda_opt,lambda_search_info['reg_err_opt']))
        if output_fig_filename:
            fig=plt.figure()
            plt.plot(lambda_search_info['lambda'],lambda_search_info['reg_err'],'-o',markerfacecolor='none',markersize=3)
            plt.plot(lambda_search_info['lambda_opt'],lambda_search_info['reg_err_opt'],'ro',markersize=3)
            plt.plot(lambda_search_info['lambda_opt'],lambda_search_info['reg_err_opt'],'ro',markerfacecolor='none',markersize=9)
            yl=[np.min(lambda_search_info['reg_err']),np.max(lambda_search_info['reg_err'])]
            if (lambda_search_info['reg_err'][0]/np.max(lambda_search_info['reg_err'][1:]))>2:
                yl[1]=np.max(lambda_search_info['reg_err'][1:])
            yl[0]=yl[0]-(yl[1]-yl[0])*.05
            plt.ylim(yl)
            plt.grid()
            plt.title('lambda_opt = %.6f' % (lambda_opt))
            plt.savefig(output_fig_filename,dpi=fig.dpi)
            print("Saved %s" % (output_fig_filename))
    else:
        print("Using user-specified lambda value: %f" % (lambda_value_arg))
        lambda_opt=lambda_value_arg
    
    lambda_opt_raw=lambda_opt
    lambda_opt=np.round(lambda_opt/lambda_round_scale)*lambda_round_scale
    
    print("Final rounded lambda: %f" % (lambda_opt))
    
    if args.saveprec or args.savepcorr:
        #convert to precision matrix
        M_allsubj['C']=[invert_tikhonov(C,lambda_opt) for C in M_allsubj['C']]
    
    if args.saveprec:
        reg_info_dict={'cov_estimator':'precision','lambda':lambda_opt, 'lambda_raw':lambda_opt_raw}
        reginv_save_outputs(M_allsubj, input_info, output_suffix='_FCprec', reg_info_dict=reg_info_dict)
    
    if args.savepcorr:
        M_allsubj['C']=[prec2pcorr(C) for C in M_allsubj['C']]
        reg_info_dict={'cov_estimator':'partialcorrelation','lambda':lambda_opt, 'lambda_raw':lambda_opt_raw}
        reginv_save_outputs(M_allsubj, input_info, output_suffix='_FCpcorr', reg_info_dict=reg_info_dict)
    
if __name__ == "__main__":
    run_fmri_regularized_inverse(sys.argv[1:])