# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:43:32 2023
######################################################################################################################################
Copyright (c)  2023 Mahsa Dadar

Script for 9 Class Tissue Segmentation from FLAIR images
This version of the pipeline has the option to perform its own preprocessing and works with .nii images.
The pipeline provides the option of using other classifiers, but is has been tested and validated with random forests 
Input arguments:
BISON.py -c <Classifier (Default: RF)> -i <Input CSV File> 
 -m <Template Mask File>  -f <Number of Folds in K-fold Cross Validation (Default=10)>
 -o <Output Path> -t <Temp Files Path> -e <Classification Mode> -n <New Data CSV File> 
 -p <Pre-trained Classifiers Path> -d  <Do Preprocessing> -l < The Number of Classes>
 
CSV File Column Headers: Subjects, XFMs, FLAIRs, Labels, Masks
Subjects:   Subject ID
FLAIR:      Path to preprocessed FLAIR image, coregistered with primary modality
XFMs:       Nonlinear transformation from template to primary modality image 
Masks:      Brain mask or mask of region of interest
Labels:     Labels (For Training, not necessary if using pre-trained classifiers)
 
Preprocessing Options: 
 Y:   Perform Preprocessing 

Classification Mode Options: 
 CV:   Cross Validation (On The Same Dataset) 
 TT:   Train-Test Model (Training on Input CSV Data, Segment New Data, Needs an extra CSV file)
 PT:   Using Pre-trained Classifiers  
 
Classifier Options:
 NB:   Naive Bayes
 LDA:  Linear Discriminant Analysis
 QDA:  Quadratic Discriminant Analysis
 LR:   Logistic Regression
 KNN:  K Nearest Neighbors 
 RF:   Random Forest 
 SVM:  Support Vector Machines 
 Tree: Decision Tree
 Bagging
 AdaBoost
#####################################################################################################################################
@author: mdadar
"""
import os
import numpy as np
import sys
import getopt
import SimpleITK as sitk

try:
    import joblib
except ModuleNotFoundError:
    # for old scikit-learn
    from sklearn.externals import joblib
import tempfile

def run_command(cmd_line):
    """
    Execute command and check the return status
    throw an exception if command failed
    """
    r=os.system(cmd_line)
    if r!=0:
        raise OSError(r,cmd_line)

#DEBUG
def draw_histograms(hist,out,modality='',dpi=100 ):
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
   
    x=np.arange(hist.shape[0])
    for c in range(hist.shape[1]):
        plt.plot(x, hist[:,c], label=f'{c+1}')  # Plot some data on the (implicit) axes.

    plt.xlabel('Intensity')
    plt.ylabel('Density')
    plt.legend()
    if modality is not None:
        plt.title(modality)

    plt.savefig(out, bbox_inches='tight', dpi=dpi)
    plt.close()
    plt.close('all')
#DEGUB

def doPreprocessing(path_nlin_mask,path_Temp, ID_Test, Label_Files_Test , Label, flr_Files_Test , flr , path_av_flr ):
    nlmf = 'Y'
    nuf = 'Y'
    volpolf = 'Y'
    if '.nii' in flr_Files_Test[0]: 
        fileFormat = 'nii'
    else:
        fileFormat = 'mnc'
    preprocessed_list = {}
    str_flr_proc = ''

    preprocessed_list_address=path_Temp+'Preprocessed.csv'
    print('Preprocessing The Images ...')
    for i in range(0 , len(flr_Files_Test)):
        if (flr != ''):
            str_File_flr = str(flr_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):
                new_command = 'nii2mnc ' + str_File_flr + ' ' + path_Temp + str(ID_Test[i]) + '_flr.mnc'      
            else:
                new_command = 'cp ' + str_File_flr + ' ' + path_Temp + str(ID_Test[i]) + '_flr.mnc'
            os.system(new_command)
            new_command = 'bestlinreg_s2 ' +  path_Temp + str(ID_Test[i]) + '_flr.mnc ' +  path_av_flr + ' ' +  path_Temp + str(ID_Test[i]) + '_flrtoTemplate.xfm'
            os.system(new_command)
            new_command = 'mincresample ' +  path_nlin_mask + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_flrtoTemplate.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_flr_Mask.mnc -invert_transform -like ' + path_Temp + str(ID_Test[i]) + '_flr.mnc -nearest -clobber'
            os.system(new_command)
            str_flr_proc = path_Temp + str(ID_Test[i]) + '_flr.mnc'
            str_main_modality = str_flr_proc
            if (nlmf == 'Y'):
                new_command = 'mincnlm -clobber -mt 1 ' + path_Temp + str(ID_Test[i]) + '_flr.mnc ' + path_Temp + str(ID_Test[i]) + '_flr_NLM.mnc -beta 0.7 -clobber'
                os.system(new_command)
                str_flr_proc = path_Temp + str(ID_Test[i]) + '_flr_NLM.mnc'
            if (nuf == 'Y'):
                new_command = 'nu_correct ' + path_Temp + str(ID_Test[i]) + '_flr_NLM.mnc '  + path_Temp + str(ID_Test[i]) + '_flr_N3.mnc -mask '+ path_Temp + str(ID_Test[i]) + '_flr_Mask.mnc  -iter 200 -distance 50 -clobber'
                os.system(new_command)
                str_flr_proc = path_Temp + str(ID_Test[i]) + '_flr_N3.mnc'
            if (volpolf == 'Y'):
                new_command = 'volume_pol ' + path_Temp + str(ID_Test[i]) + '_flr_N3.mnc '  + path_av_flr + ' --order 1 --noclamp --expfile ' + path_Temp + str(ID_Test[i]) + '_flr_norm --clobber ' + path_Temp + str(ID_Test[i]) + '_flr_VP.mnc --source_mask '+ path_Temp + str(ID_Test[i]) + '_flr_Mask.mnc --target_mask '+ path_nlin_mask
                os.system(new_command)
                str_flr_proc = path_Temp + str(ID_Test[i]) + '_flr_VP.mnc'
                
            new_command = 'bestlinreg_s2 ' +  str_flr_proc + ' ' +  path_av_flr + ' ' +  path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_lin.xfm'
            os.system(new_command)
            new_command = 'mincresample ' +  str_flr_proc + ' -transform ' +  path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_lin.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_flr_lin.mnc -like ' + path_av_flr + ' -clobber'
            os.system(new_command)
            new_command = 'nlfit_s ' +  path_av_flr + ' ' + path_Temp + str(ID_Test[i]) + '_flr_lin.mnc ' +   path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_nlininv.xfm -level 2 -clobber'
            os.system(new_command)
            new_command = 'xfminvert ' + path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_lin.xfm ' + path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_lininv.xfm'
            os.system(new_command)
            new_command = 'xfmconcat '  + path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_nlininv.xfm '+ path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_lininv.xfm '+ path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_both.xfm'
            os.system(new_command)

        if (Label != ''):
            str_File_Label = str(Label_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            if (fileFormat == 'nii'):   
                new_command = 'nii2mnc ' + str_File_Label + ' ' + path_Temp + str(ID_Test[i]) + '_Label.mnc' 
            else:
                new_command = 'cp ' + str_File_Label + ' ' + path_Temp + str(ID_Test[i]) + '_Label.mnc'
            os.system(new_command)
            str_File_Label = path_Temp + str(ID_Test[i]) + '_Label.mnc' 
        
        new_command = 'mincresample ' +  path_nlin_mask + ' -transform ' + path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_both.xfm' + ' ' +  path_Temp + str(ID_Test[i]) + '_Mask_nl.mnc -like ' + str_main_modality + ' -nearest -clobber'
        os.system(new_command) 
        str_Mask = path_Temp + str(ID_Test[i]) + '_Mask_nl.mnc'
        nl_xfm = path_Temp + str(ID_Test[i]) + '_flrtoTemplate_pp_both.xfm'
        print('.')
        preprocessed_list[0,0]= 'Subjects,FLAIRs,Masks,XFMs'
        preprocessed_list[i+1,0]= str(ID_Test[i]) + ',' + str_flr_proc + ',' + str_Mask + ',' + nl_xfm

        if (Label != ''):
            preprocessed_list[0,0]=  preprocessed_list[0,0] + ',Labels'
            preprocessed_list[i+1,0]=  preprocessed_list[i+1,0] + ',' + str_File_Label
            
    outfile = open( preprocessed_list_address, 'w' )
    for key, value in sorted( preprocessed_list.items() ):
        outfile.write(  str(value) + '\n' )
    outfile = open( preprocessed_list_address, 'w' )
    for key, value in sorted( preprocessed_list.items() ):
        outfile.write(  str(value) + '\n' )
    return [preprocessed_list_address]
###########################################################################################################################################################################
def Calculate_Tissue_Histogram(Files_Train , Masks_Train , Label_Files_Train , image_range , n_labels):
    PDF_Label = np.zeros(shape = (image_range , n_labels),dtype=np.float64)
    print(('Calculating Histograms of Tissues: .'), end=' ',flush=True)
    for i in range(0 , len(Files_Train)):
        print(('.'), end='',flush=True)
        str_File = str(Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        str_Mask = str(Masks_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
        str_Label = str(Label_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')

        manual_segmentation = sitk.GetArrayFromImage(sitk.ReadImage(str_Label))
        image_vol = sitk.GetArrayFromImage(sitk.ReadImage(str_File))      
        brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))> 0
        
        image_vol  = np.round(image_vol).astype(np.int32)

        for nl in range(0 , n_labels):
            masked_vol = image_vol[ (manual_segmentation==(nl+1)) * brain_mask]
            for j in range(1 , image_range):
                PDF_Label[j,nl] = PDF_Label[j,nl] + np.sum( masked_vol == j, dtype=np.float64)

    # VF: normalize after all files are processed
    for nl in range(0 , n_labels):
        PDF_Label[:,nl] = PDF_Label[:,nl] / np.sum(PDF_Label[:,nl])
    print(' Done.')
    return PDF_Label
###########################################################################################################################################################################
def load_csv(csv_file):
    import csv
    data = {}
    with open(csv_file , 'r') as f:
        for r in csv.DictReader(f):
            for k in r.keys():
                try:
                    data[k].append(r[k])
                except KeyError:
                    data[k] = [r[k]]
    return data
###########################################################################################################################################################################
def get_Train_Test(Indices_G , K , IDs):
    i_train = 0
    i_test = 0
    ID_Train = np.empty(shape = (np.sum(Indices_G != K) , 1) , dtype = list , order = 'C')
    ID_Test = np.empty(shape = (np.sum(Indices_G == K) , 1) , dtype = list , order = 'C')        
    for i in range(0 , len(Indices_G)):
        if (Indices_G[i] != K):
            ID_Train[i_train] = IDs[i]
            i_train = i_train + 1
        if (Indices_G[i] == K):
            ID_Test[i_test] = IDs[i]
            i_test = i_test + 1
    return [ID_Train , ID_Test]
###########################################################################################################################################################################
def get_addressess(TestList):
    InputListInfo_Test = load_csv(TestList)    
    ID_Test = InputListInfo_Test['Subjects']
    if 'XFMs' in InputListInfo_Test:    
        XFM_Files_Test = InputListInfo_Test['XFMs']
        xfmf = 'exists'
    else:
        xfmf = ''
        XFM_Files_Test = ''
    if 'Masks' in InputListInfo_Test:    
        Mask_Files_Test = InputListInfo_Test['Masks']
        maskf = 'exists'
    else:
        maskf = ''
        Mask_Files_Test = ''
    if 'FLAIRs' in InputListInfo_Test:    
        flr_Files_Test = InputListInfo_Test['FLAIRs']
        flr = 'exists'
    else:
        flr =''
        flr_Files_Test = ''
    if 'Labels' in InputListInfo_Test:    
        Label_Files_Test = InputListInfo_Test['Labels']
        Label = 'exists'
    else:
        Label = ''
        Label_Files_Test = ''

    return [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, flr_Files_Test, flr, Label_Files_Test, Label]
###########################################################################################################################################################################

def warp_and_read_prior(prior, ref_scan, xfm, tmp_file_location,clobber=False):
    """ Apply INVERSE xfm to prior , like ref_scan and store into tmp_file_location
    then read into numpy array. 
    Parameters:
        prior - input scan
        ref_scan - reference space
        xfm - transformation
        tmp_file_location - output file
        
    Returns: 
        numpy array of floats of the contents of output
    """
    run_command(f'itk_resample {prior} --like  {ref_scan} --transform {xfm} {tmp_file_location} --clobber')
    return sitk.GetArrayFromImage(sitk.ReadImage(tmp_file_location))

def main(argv):   
    # Default Values    
    n_folds=10
    image_range = 256
    subject = 0
    Classifier='RF'    
    doPreprocessingf = False
    path_trained_classifiers=''
    InputList=''
    TestList=''
    path_Temp=None
    path_nlin_files = ''
    ClassificationMode = ''
    path_output = ''
    TestList = ''
    n_labels = 3
    n_jobs = 1
    temp_dir = None
    nifti = False


    _help="""
BISON.py -c <Classifier (Default: LDA)> 
         -i <Input CSV File>  - for training only (TT or CV mode)
         -m <Templates directory> - location of template files, unless -p option is used 
         -f <Number of Folds in K-fold Cross Validation (Default=10)>
         -o <Output Path>
         -t <Temp Files Path>
         -e <Classification Mode> (CV/TT/PT)
         -n <New Data CSV File>  - for segmenting only (TT or PT mode)
         -p <Pre-trained Classifiers Path> - pretrained classfier and templates directory
         -d <Do Preprocessing>  - run nonlinear registration
         -l <The Number of Classes, default 3> 
         -j <n> maximum number of jobs (CPUs) to use for classification default 1, -1 - all possible
         --nifti output a nifti file

CSV File Column Headers: Subjects, XFMs, FLAIRs, Labels, Masks
Preprocessing Options: 
    Y:   Perform Preprocessing 
Classification Mode Options:
    CV:   Cross Validation (On The Same Dataset) 
    TT:   Train-Test Model (Training on Input CSV Data, Segment New Data, Needs an extra CSV file)
    PT:   Using Pre-trained Classifiers 

Classifier Options:
    NB:   Naive Bayes
    LDA:  Linear Discriminant Analysis
    QDA:  Quadratic Discriminant Analysis
    LR:   Logistic Regression
    KNN:  K Nearest Neighbors 
    RF:   Random Forest 
    SVM:  Support Vector Machines 
    Tree: Decision Tree
    Bagging
    AdaBoost
    """

    try:
        opts, args = getopt.getopt(argv,"hc:i:m:o:t:e:n:f:p:dl:j:",["cfile=","ifile=","mfile=","ofile=","tfile=","efile=","nfile=","ffile=","pfile=","dfile","lfile=","jobs=","nifti"])
    except getopt.GetoptError:
        print(_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(_help)
            sys.exit()
        elif opt in ("-c", "--cfile"):
            Classifier = arg
        elif opt in ("-i", "--ifile"):
            InputList = arg
        elif opt in ("-m", "--mfile"):
            path_nlin_files = arg
        elif opt in ("-o", "--ofile"):
            path_output = arg
        elif opt in ("-t", "--tfile"):
            if not os.path.exists(arg):
                os.makedirs(arg)
            path_Temp = arg+str(np.random.randint(1000000, size=1)).replace("[",'').replace("]",'').replace(" ",'').replace(" ",'')+'_BISON_'
        elif opt in ("-e", "--efile"):
            ClassificationMode = arg
        elif opt in ("-n", "--nfile"):
            TestList = arg
        elif opt in ("-f", "--ffile"):
            n_folds = int(arg)
        elif opt in ("-p", "--pfile"):
            path_trained_classifiers = arg
        elif opt in ("-d", "--dfile"):
            doPreprocessingf = True
        elif opt in ("-l", "--lfile"):
            n_labels = int(arg)
        elif opt in ("-j", "--jobs"):
            n_jobs = int(arg)
        elif opt in ("--nifti"):
            nifti = True
        else:
            print("Unknown option:",opt)
            print(_help)
            sys.exit(1)

    if path_Temp is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="BISON_")
        path_Temp = temp_dir.name + os.sep + 'temp_'

    print('The Selected Input CSV File is ', InputList)
    print('The Selected Test CSV File is ', TestList)
    print('The Selected Classifier is ', Classifier)
    print('The Classification Mode is ', ClassificationMode)
    print('The Selected Template Mask is ', path_nlin_files)
    print('The Selected Output Path is ', path_output)    
    print('The Assigned Temp Files Path is ', path_Temp)

    if doPreprocessingf:
        print('Preprocessing:  Yes')

    if (Classifier == 'NB'):
        # Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
    elif (Classifier == 'LDA'):
        # Linear Discriminant Analysis
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis(solver = "svd" )  
    elif (Classifier == 'QDA'):
        # Quadratic Discriminant Analysis
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        clf = QuadraticDiscriminantAnalysis()
    elif (Classifier == 'LR'):
        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C = 200 , penalty = 'l2', tol = 0.01)
    elif (Classifier == 'KNN'):
        # K Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors = 10)
    elif (Classifier == 'Bagging'):
        # Bagging
        from sklearn.ensemble import BaggingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        clf = BaggingClassifier(KNeighborsClassifier() , max_samples = 0.5 , max_features = 0.5)
    elif (Classifier == 'AdaBoost'):
        # AdaBoost
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators = 100)
    elif (Classifier == 'RF'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 100, n_jobs=n_jobs)
    elif (Classifier == 'RF0'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=10,verbose=True)
    elif (Classifier == 'RF1'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=20,verbose=True)
    elif (Classifier == 'RF2'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=40,verbose=True)
    elif (Classifier == 'RF3'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=100,verbose=True)
    elif (Classifier == 'RF4'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=1000,verbose=True)
    elif (Classifier == 'RF5'):
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators = 128, n_jobs=n_jobs, max_depth=100, min_samples_split=100, min_samples_leaf=100, verbose=True)
    elif (Classifier == 'SVM'):
        # Support Vector Machines
        from sklearn import svm
        clf = svm.LinearSVC()
    elif (Classifier == 'Tree'):
        # Decision Tree
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
    else:
        print('The Selected Classifier Was Not Recognized')
        sys.exit()

    if (InputList != ''):
    	[IDs, XFM_Files, xfmf, Mask_Files, maskf, flr_Files, flr, Label_Files, Label] = get_addressess(InputList)

####################### Preprocessing ####################################################################################################################################
    if (path_nlin_files == ''):
	    print('No path has been defined for the template files')
	    sys.exit()
    if (path_nlin_files != ''):
    	path_nlin_mask = path_nlin_files + os.sep + 'Mask.mnc'
    	path_av_flr = path_nlin_files +  os.sep + 'Av_flr.mnc'

    if ((path_trained_classifiers == '') & (ClassificationMode == 'PT')):
	    print('No path has been defined for the pretrained classifiers')
	    sys.exit()
    if (path_trained_classifiers != ''):    
        path_av_flr = path_trained_classifiers + os.sep + 'Av_flr.mnc'

    if (n_labels == 0):
        print('The number of classes has not been determined')
        sys.exit()

    if (ClassificationMode == ''):
        print('The classification mode has not been determined')
        sys.exit()
###########################################################################################################################################################################
    if ClassificationMode == 'CV':
        if doPreprocessingf:
            doPreprocessing(path_nlin_mask,path_Temp, IDs, Label_Files , Label, flr_Files , flr , path_av_flr )
            [IDs, XFM_Files, xfmf, Mask_Files, maskf, flr_Files, flr, Label_Files, Label] = get_addressess( path_Temp + 'Preprocessed.csv')

        if Label == '':    
            print('No Labels to Train on')
            sys.exit()

        Indices_G = np.random.permutation(len(IDs)) * n_folds / len(IDs)
        Kappa = np.zeros(shape = (len(IDs) , n_labels))
        ID_Subject = np.empty(shape = (len(IDs),1) , dtype = list, order = 'C')       

        for K in range(0 , n_folds):
            [ID_Train , ID_Test] = get_Train_Test(Indices_G , K , IDs)    
            [XFM_Files_Train , XFM_Files_Test] = get_Train_Test(Indices_G , K , XFM_Files)    
            [Label_Files_Train , Label_Files_Test] = get_Train_Test(Indices_G , K , Label_Files)    
            [Mask_Files_Train , Mask_Files_Test] = get_Train_Test(Indices_G , K , Mask_Files)    
            
            n_features=n_labels           
            if (flr != ''):        
                [flr_Files_Train , flr_Files_Test] = get_Train_Test(Indices_G , K , flr_Files)    
                flr_PDF_Label = Calculate_Tissue_Histogram(flr_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                n_features = n_features + n_labels + 2
                
                       
            path_sp = path_nlin_files + os.sep + 'SP_'
            
            X_All = np.empty(shape = (0 , n_features) , dtype = float , order = 'C')
            Y_All = np.empty(shape = (0 , ) , dtype = np.int32 , order = 'C')
            
            for i in range(0 , len(ID_Train)):
                str_Train = str(ID_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                print(('Extracting The Features: Subject ID = ' + str_Train))
                
                str_Mask = str(Mask_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
                ind_Mask = (Mask > 0)
                N=int(np.sum(Mask))
                Label = str(Label_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'') 
                WMT = sitk.GetArrayFromImage(sitk.ReadImage(Label))

                nl_xfm = str(XFM_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    spatial_prior = warp_and_read_prior(path_sp + str(nl+1) + '.mnc',Label,nl_xfm, path_Temp + 'train_' + str(i) + '_' + str(K) + '_tmp_sp_'+str(nl+1)+'.mnc')
                    spatial_priors[0:N,nl] = spatial_prior[ind_Mask]

                if (flr != ''):
                    str_flr = str(flr_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    flr = sitk.GetArrayFromImage(sitk.ReadImage(str_flr))
                    av_flr = warp_and_read_prior(path_av_flr, Label, nl_xfm, path_Temp + 'train_' + str(i)+'_' + str(K) + '_tmp_flr.mnc')
                    flr[flr < 1] = 1
                    flr[flr > (image_range - 1)] = (image_range - 1)
                    flr_Label_probability = np.empty(shape = (N, n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        flr_Label_probability[:,nl] = flr_PDF_Label[np.round(flr[ind_Mask]).astype(np.int),nl]
                    X_flr = np.zeros(shape = (N , 2))
                    X_flr[0 : N , 0] = flr[ind_Mask]
                    X_flr[0 : N , 1] = av_flr[ind_Mask]
                    X_flr = np.concatenate((X_flr , flr_Label_probability) , axis = 1)

                

                else:
                    X = np.zeros(shape = (N , 0))
                    X = np.concatenate((X , spatial_priors) , axis = 1)
                if (flr != ''):
                    X = np.concatenate((X , X_flr) , axis = 1)
                
                X_All = np.concatenate((X_All , X) , axis = 0)
                Y = np.zeros(shape = (N , ),dtype=np.int32)
                Y[0 : N , ] = (WMT[ind_Mask])
                Y_All = np.concatenate((Y_All , Y) , axis = 0)

            print('Training The Classifier ...')
            clf = clf.fit(X_All , Y_All)

            for i in range(0 , len(ID_Test)):
                str_Test = str(ID_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                print(('Segmenting Volumes: Subject: ID = ' + str_Test))
                Label = str(Label_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                WMT = sitk.GetArrayFromImage(sitk.ReadImage(Label))
                str_Mask = str(Mask_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
                ind_Mask = (Mask > 0)
                N=int(np.sum(Mask))
                nl_xfm = str(XFM_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    spatial_prior = warp_and_read_prior(path_sp + str(nl+1) + '.mnc', Label,nl_xfm, path_Temp + 'test_' + str(i) + '_' + str(K) + '_tmp_sp_'+str(nl+1)+'.mnc')
                    spatial_priors[0:N,nl] = spatial_prior[ind_Mask]                
                               
                if (flr != ''):
                    str_flr = str(flr_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                    flr = sitk.GetArrayFromImage(sitk.ReadImage(str_flr))
                    av_flr = warp_and_read_prior(path_av_flr, Label, str_flr, path_Temp + 'test_' +  str(i) + '_' + str(K) + '_tmp_flr.mnc')
                    flr[flr < 1] = 1
                    flr[flr > (image_range - 1)] = (image_range - 1)
                    flr_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                    for nl in range(0 , n_labels):
                        flr_Label_probability[:,nl] = flr_PDF_Label[np.round(flr[ind_Mask]).astype(np.int32),nl]
                    N = len(flr_Label_probability)
                    X_flr = np.zeros(shape = (N , 2))
                    X_flr[0 : N , 0] = flr[ind_Mask]
                    X_flr[0 : N , 1] = av_flr[ind_Mask]
                    X_flr = np.concatenate((X_flr , flr_Label_probability) , axis = 1)
                
                else:
                    X = np.zeros(shape = (N , 0))
                    X = np.concatenate((X , spatial_priors) , axis = 1)    
                if (flr != ''):
                    X = np.concatenate((X , X_flr) , axis = 1)

                        
                Y = np.zeros(shape = (N , ))
                Y[0 : N] = WMT[ind_Mask]
                Binary_Output = clf.predict(X)
                for nl in range(0 , n_labels):
                    Kappa[subject,nl] = 2 * np.sum((Y==(nl+1)) * (Binary_Output==(nl+1))) / (np.sum(Y==(nl+1)) + np.sum(Binary_Output==(nl+1)))
                    ID_Subject[subject] = ID_Test[i]
                if (np.sum(Y) + np.sum(Binary_Output)) == 0:
                    Kappa[subject] = 1
                print(Kappa[subject])
                subject = subject + 1
                        
                WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0,:]) , len(Mask[0 , 0 , :])),dtype=np.int32)
                WMT_auto[ind_Mask] = Binary_Output[0 : N]
                
                inputImage = sitk.ReadImage(str_Mask)
                result_image = sitk.GetImageFromArray(WMT_auto)
                result_image.CopyInformation(inputImage)
                sitk.WriteImage(result_image,  path_output + os.sep +  Classifier + '_' + str_Test + '_Label.mnc')             
        
        print('Cross Validation Successfully Completed. \nKappa Values:\n')        
        print(Kappa)
        print('Indices')
        print(Indices_G) 
        print(('Mean Kappa: ' + str(np.mean(Kappa)) + ' - STD Kappa: ' + str(np.std(Kappa))))  
###########################################################################################################################################################################    
    elif ClassificationMode == 'TT':
        K=0
        Indices_G=np.ones(shape = (len(IDs) , 1))
        ID_Train = IDs
        XFM_Files_Train = XFM_Files    
        Label_Files_Train = Label_Files   
        Mask_Files_Train = Mask_Files  
        if (flr != ''):
            flr_Files_Train = flr_Files
        
        if doPreprocessingf:
            doPreprocessing(path_nlin_mask,path_Temp, IDs, Label_Files , Label, flr_Files , flr , path_av_flr )
            [IDs, XFM_Files, xfmf, Mask_Files, maskf, flr_Files, flr, Label_Files, Label] = get_addressess(path_Temp+'Preprocessed.csv')

        [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, flr_Files_Test, flr, Label_Files_Test, Label] = get_addressess(TestList)

        n_features=n_labels           
        if (flr != ''):        
            [flr_Files_Train , tmp] = get_Train_Test(Indices_G , K , flr_Files)
            if os.path.exists(path_output+os.sep + 'flr_Label.pkl'):
                flr_PDF_Label = joblib.load(path_output+os.sep +'flr_Label.pkl')
            else:
                flr_PDF_Label = Calculate_Tissue_Histogram(flr_Files_Train , Mask_Files_Train , Label_Files_Train , image_range , n_labels)
                draw_histograms(flr_PDF_Label,path_output+os.sep +'flr_Label.png',"FLAIR")
            n_features = n_features + n_labels + 2
                
                             
        path_sp = path_nlin_files + 'SP_'
            
        X_All = np.empty(shape = (0 , n_features ) , dtype = float , order = 'C')
        Y_All = np.empty(shape = (0 , ) , dtype = np.int32 , order = 'C')
    
        for i in range(0 , len(ID_Train)):
            str_Train = str(ID_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            print(('Extracting The Features: Subject: ID = ' + str_Train))
                
            str_Mask = str(Mask_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
            ind_Mask = (Mask > 0)
            N=int(np.sum(Mask))
            Label = str(Label_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'') 
            WMT = sitk.GetArrayFromImage(sitk.ReadImage(Label))
                
            nl_xfm = str(XFM_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
            for nl in range(0 , n_labels):
                spatial_prior = warp_and_read_prior(f"{path_sp}{nl+1}.mnc", Label, nl_xfm, f"{path_Temp}train_{i}_{K}_sp_{nl+1}.mnc")
                spatial_priors[0:N,nl] = spatial_prior[ind_Mask]
                
            if (flr != ''):
                str_flr = str(flr_Files_Train[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                flr = sitk.GetArrayFromImage(sitk.ReadImage(str_flr))
                av_flr = warp_and_read_prior(path_av_flr, Label,nl_xfm, f"{path_Temp}train_{i}_{K}_av_flr.mnc")
                flr[flr < 1] = 1
                flr[flr > (image_range - 1)] = (image_range - 1)
                flr_Label_probability = np.empty(shape = (N, n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    flr_Label_probability[:,nl] = flr_PDF_Label[np.round(flr[ind_Mask]).astype(np.int32),nl]
                X_flr = np.zeros(shape = (N , 2))
                X_flr[0 : N , 0] = flr[ind_Mask]
                X_flr[0 : N , 1] = av_flr[ind_Mask]
                X_flr = np.concatenate((X_flr , flr_Label_probability) , axis = 1)
                
            X = np.zeros(shape = (N , 0))
            X = np.concatenate((X , spatial_priors) , axis = 1)  
            if (flr != ''):
                X = np.concatenate((X , X_flr) , axis = 1)

                
            X_All = np.concatenate((X_All , X) , axis = 0)
            Y = np.zeros(shape = (N , ),dtype=np.int32)
            Y[0 : N , ] = (WMT[ind_Mask])    
            Y_All = np.concatenate((Y_All , Y) , axis = 0)
            
        print('Training The Classifier ...')
        clf = clf.fit(X_All , Y_All)
        print('Training Successfully Completed.')        

        saveFlag=1

        if saveFlag == 1:
            print('Saving the Classifier ...')  
            path_trained_classifiers = path_output 
            path_save_classifier = path_trained_classifiers + os.sep + Classifier + '.pkl'    
            

            joblib.dump(clf,path_save_classifier)
            if (flr != ''):
                joblib.dump(flr_PDF_Label,path_trained_classifiers+os.sep+'flr_Label.pkl')
            
            print("Trained Classifier Successfully Saved in: ",path_trained_classifiers)
            sys.exit()
            
        for i in range(0 , len(ID_Test)):
            str_Test = str(ID_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            print(('Segmenting Volumes: Subject: ID = ' + str_Test))
            str_Mask = str(Mask_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'') .replace(" ",'')
            Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
            ind_Mask = (Mask > 0)
            nl_xfm = str(XFM_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            N=int(np.sum(Mask))
              
            if (flr != ''):
		
                str_flr = str(flr_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    spatial_prior = warp_and_read_prior(f"{path_sp}{nl+1}.mnc", str_flr, nl_xfm, f"{path_Temp}test_{i}_{K}_sp_{nl+1}.mnc")
                    spatial_priors[0:N,nl] = spatial_prior[ind_Mask]
                
                flr = sitk.GetArrayFromImage(sitk.ReadImage(str_flr))
                av_flr = warp_and_read_prior(path_av_flr, str_flr, nl_xfm, f"{path_Temp}test_{i}_{K}_av_flr.mnc")
                flr[flr < 1] = 1
                flr[flr > (image_range - 1)] = (image_range - 1)
                flr_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    flr_Label_probability[:,nl] = flr_PDF_Label[np.round(flr[ind_Mask]).astype(np.int32),nl]
                N = len(flr_Label_probability)
                X_flr = np.zeros(shape = (N , 2))
                X_flr[0 : N , 0] = flr[ind_Mask]
                X_flr[0 : N , 1] = av_flr[ind_Mask]
                X_flr = np.concatenate((X_flr , flr_Label_probability) , axis = 1)
    
                           
            else:
                X = np.zeros(shape = (N , 0))
                X = np.concatenate((X , spatial_priors) , axis = 1)
            if (flr != ''):
                X = np.concatenate((X , X_flr) , axis = 1)
   
            Y = np.zeros(shape = (N , ), dtype=np.int32)
            Binary_Output = clf.predict(X)       
            Prob_Output=clf.predict_proba(X)            
            #### Saving results #########################################################################################################################            
            WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])),dtype=np.int32)
            WMT_auto[ind_Mask] = Binary_Output[0 : N]
            
            str_Labelo= path_output +os.sep + Classifier + '_' + str_Test
            
            inputImage = sitk.ReadImage(str_Mask)
            result_image = sitk.GetImageFromArray(WMT_auto)
            result_image.CopyInformation(inputImage)
            sitk.WriteImage(result_image,  str_Labelo + '_Label.mnc')
            
            Prob_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])))
            Prob_auto[ind_Mask] = Prob_Output[0 : N,1]
            
            result_image = sitk.GetImageFromArray(Prob_auto)
            result_image.CopyInformation(inputImage)
            sitk.WriteImage(result_image,  str_Labelo + '_P.mnc')
            
            if (flr != ''):            
                new_command = 'minc_qc.pl ' + str_flr + ' --mask ' + str_Labelo + '_Label.mnc ' + str_Labelo + '_Label.jpg --big --clobber --spectral-mask  --image-range 0 200 --mask-range 0 ' + str(n_labels)
                os.system(new_command)

###########################################################################################################################################################################    
    elif ClassificationMode == 'PT':
        path_sp   =path_trained_classifiers + os.sep + 'SP_'
        path_av_flr=path_trained_classifiers + os.sep + 'Av_flr.mnc'
        [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, flr_Files_Test, flr, Label_Files_Test, Label] = get_addressess( TestList )
        path_saved_classifier = path_trained_classifiers + os.sep + Classifier+ '.pkl'
############## Preprocessing ####################################################################################################################################
        if doPreprocessingf:
            doPreprocessing(path_nlin_mask,path_Temp, ID_Test, Label_Files_Test , Label, flr_Files_Test , flr , path_av_flr )
            [ID_Test, XFM_Files_Test, xfmf, Mask_Files_Test, maskf, flr_Files_Test, flr, Label_Files_Test, Label] = get_addressess(path_Temp+'Preprocessed.csv')
########## Loading Trained Classifier ##########################################################################################################################            
        print('Loading the Pre-trained Classifier from: ' + path_saved_classifier)
        clf = joblib.load(path_saved_classifier)
        # set maximum jobs to run in parallel
        clf.n_jobs = n_jobs

        K=0
        if (flr != ''):
            flr_PDF_Label=joblib.load(path_trained_classifiers+os.sep +'FLAIR_Label.pkl')
        
        for i in range(0 , len(ID_Test)):
            str_Test = str(ID_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            print(('Segmenting Volumes: Subject: ID = ' + str_Test))
            str_Mask = str(Mask_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'') .replace(" ",'')
            Mask = sitk.GetArrayFromImage(sitk.ReadImage(str_Mask))
            ind_Mask = (Mask > 0)
            nl_xfm = str(XFM_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
            N=int(np.sum(Mask))
            print('Extracting The Features ...')
            if (flr != ''):
                str_flr = str(flr_Files_Test[i]).replace("[",'').replace("]",'').replace("'",'').replace(" ",'')
                spatial_priors = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    spatial_prior = warp_and_read_prior(f"{path_sp}{nl+1}.mnc", str_flr, nl_xfm, f"{path_Temp}test_{i}_{K}_sp_{nl+1}.mnc")
                    spatial_priors[0:N,nl] = spatial_prior[ind_Mask]

                flr = sitk.GetArrayFromImage(sitk.ReadImage(str_flr))
                av_flr = warp_and_read_prior(path_av_flr, str_flr, nl_xfm, f"{path_Temp}test_{i}_{K}_av_flr.mnc")
                flr[flr < 1] = 1
                flr[flr > (image_range - 1)] = (image_range - 1)
                flr_Label_probability = np.empty(shape = (N , n_labels) , dtype = float , order = 'C')
                for nl in range(0 , n_labels):
                    flr_Label_probability[:,nl] = flr_PDF_Label[np.round(flr[ind_Mask]).astype(np.int32),nl]
                N = len(flr_Label_probability)
                X_flr = np.zeros(shape = (N , 2))
                X_flr[0 : N , 0] = flr[ind_Mask]
                X_flr[0 : N , 1] = av_flr[ind_Mask]
                X_flr = np.concatenate((X_flr , flr_Label_probability) , axis = 1)
    

            X = np.zeros(shape = (N , 0))
            X = np.concatenate((X , spatial_priors) , axis = 1)
            if (flr != ''):
                X = np.concatenate((X , X_flr) , axis = 1)
            
            Y = np.zeros(shape = (N , ))
            print("Applying The Classifier ...")
            Binary_Output = clf.predict(X)
            Prob_Output=clf.predict_proba(X)
            #### Saving results #########################################################################################################################            
            WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])), dtype=np.int32)
            WMT_auto[ind_Mask] = Binary_Output[0 : N]
            str_Labelo = path_output + os.sep + Classifier + '_' + str_Test 
            
            inputImage = sitk.ReadImage(str_Mask)
            result_image = sitk.GetImageFromArray(WMT_auto)
            result_image.CopyInformation(inputImage)
            sitk.WriteImage(result_image,  str_Labelo + '_Label.mnc')
                   
            run_command('mincresample ' + str_Labelo + '_Label.mnc -like ' + str_flr + ' ' + str_Labelo + '_Labelr.mnc -clobber')
            for nl in range(0 , n_labels):
                WMT_auto = np.zeros(shape = (len(Mask) , len(Mask[0 , :]) , len(Mask[0 , 0 , :])), dtype=np.float32)
                WMT_auto[ind_Mask] = Prob_Output[0 : N,nl]
                inputImage = sitk.ReadImage(str_flr)
                result_image = sitk.GetImageFromArray(WMT_auto)
                result_image.CopyInformation(inputImage)
                sitk.WriteImage(result_image,  str_Labelo + '_Prob_Label_'+str(nl+1) +'.mnc')
              
            if nifti:
                run_command('mnc2nii ' + str_Labelo + '_Label.mnc ' + str_Labelo + '_Label.nii')
            
            if (flr != ''):            
                run_command('minc_qc.pl ' + str_flr + ' --mask ' + str_Labelo + '_Labelr.mnc ' + str_Labelo + '_Label.jpg --big --clobber --spectral-mask  --image-range 0 200 --mask-range 0 ' + str(n_labels))
                run_command('minc_qc.pl ' + str_flr + ' ' +  str_Labelo + '_flr.jpg --big --clobber --image-range 0 100 ')

    print('Segmentation Successfully Completed. ')

if __name__ == "__main__":
   main(sys.argv[1:])   

