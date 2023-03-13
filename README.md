# BISON-WMH

BISON (Brain tIsue SegmentatiOn pipeliNe), modified to perform tissue + WMH segmentation using a random forests classifier and a set of intensity and location priors obtained based on T1w and FLAIR images.

# Execution Example
python ./BISON_L9.py -c RF0 -m Trained_Classifiers/ -o  Outputs/ -t Temp_Files/ -e PT -n  to_segment.csv  -p  Trained_Classifiers/ -l 9

# List of Labels
1. Ventricles
2. CSF
3. cerebellar GM
4. cerebellar WM
5. brainstem
6. deep GM
7. cortical GM
8. WM
9. WMHs

# Input csv format (to_segment.csv)
Subjects,T1s,FLAIRs,Masks,XFMs 

S1,t1.mnc,flr.mnc,mask.mnc,xfm.xfm

# Dependencies
1. minc-toolkit 
2. anaconda

# Reference
    Dadar, M., & Collins, D. L. (2021). BISON: Brain tissue segmentation pipeline using T1‐weighted magnetic resonance images and a random forest classifier. Magnetic Resonance in Medicine, 85(4), 1881-1894.
