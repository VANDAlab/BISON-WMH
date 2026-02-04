# BISON-WMH

BISON (Brain tIsue SegmentatiOn pipeliNe), modified to perform tissue + WMH segmentation using a random forests classifier and a set of intensity and location priors obtained based on T1w and FLAIR images.

## Overview

BISON-WMH is a medical image processing pipeline for automated brain tissue segmentation and White Matter Hyperintensity (WMH) detection. The project implements two main pipeline variants:

- **BISON_L9.py** - Full 9-class tissue segmentation using T1 + T2 + PD + FLAIR images
- **BISON_FLAIR.py** - Simplified FLAIR-only segmentation pipeline

### Tissue Classes (9-class model)

| # | Label |
|---|-------|
| 1 | Ventricles |
| 2 | CSF |
| 3 | Cerebellar GM |
| 4 | Cerebellar WM |
| 5 | Brainstem |
| 6 | Deep GM |
| 7 | Cortical GM |
| 8 | White Matter (WM) |
| 9 | White Matter Hyperintensities (WMHs) |

## Installation

### Prerequisites

**External Tools:**
- [minc-toolkit](https://github.com/BIC-MNI/minc-toolkit) - MINC image processing utilities (mincresample, minc_qc.pl, mnc2nii)

**Python:** 3.10 or 3.11 (scikit-learn 1.0.2 is incompatible with Python 3.12+)

### Setup with uv

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.10
uv sync
```

### Manual Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install numpy\<2 SimpleITK scikit-learn==1.0.2 matplotlib
```

## Usage

### Basic Syntax

```bash
python ./BISON_L9.py -c <CLASSIFIER> -m <TEMPLATE_DIR> -o <OUTPUT_DIR> -t <TEMP_DIR> -e <MODE> -n <INPUT_CSV> -p <CLASSIFIER_DIR> -l <NUM_CLASSES>
```

### Example: Pre-trained Model Segmentation

```bash
python ./BISON_L9.py \
  -c RF0 \
  -m Trained_Classifiers/ \
  -o Outputs/ \
  -t Temp_Files/ \
  -e PT \
  -n to_segment.csv \
  -p Trained_Classifiers/ \
  -l 9
```

### Example: FLAIR-only Pipeline

```bash
python ./BISON_FLAIR.py \
  -c RF0 \
  -m Trained_Classifiers/ \
  -o Outputs/ \
  -t Temp_Files/ \
  -e PT \
  -n to_segment.csv \
  -p Trained_Classifiers/ \
  -l 9
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-c` | Classifier type (RF0, RF, NB, LDA, QDA, LR, KNN, SVM, Tree, Bagging, AdaBoost) | RF |
| `-i` | Input CSV file for training (required for CV/TT modes) | - |
| `-m` | Template mask file/directory | - |
| `-f` | Number of folds for K-fold cross-validation | 10 |
| `-o` | Output path | - |
| `-t` | Temp files path | Auto-generated |
| `-e` | Execution mode (CV/TT/PT) | - |
| `-n` | New data CSV file for segmentation (TT/PT modes) | - |
| `-p` | Pre-trained classifiers path (PT mode) | - |
| `-d` | Enable preprocessing flag | Disabled |
| `-l` | Number of classes | 3 |
| `-j` | Number of parallel jobs (-1 for all CPUs) | 1 |
| `--nifti` | Convert output to NIfTI format | Disabled |

## Execution Modes

### CV (Cross Validation)
K-fold cross-validation on the input dataset. Requires training labels in the CSV.

```bash
python ./BISON_L9.py -i training_data.csv -e CV -o results/
```

### TT (Train-Test)
Train on input data, segment new data. Requires two CSV files.

```bash
python ./BISON_L9.py -i training_data.csv -n new_data.csv -e TT -o results/
```

### PT (Pre-trained)
Use pre-trained classifiers for segmentation only (most common for inference).

```bash
python ./BISON_L9.py -n to_segment.csv -e PT -p Trained_Classifiers/ -o results/
```

## Input CSV Format

### BISON_L9.py
```csv
Subjects,T1s,T2s,PDs,FLAIRs,Masks,XFMs,Labels
S1,t1.mnc,t2.mnc,pd.mnc,flair.mnc,mask.mnc,xfm.xfm,label.mnc
```

### BISON_FLAIR.py
```csv
Subjects,FLAIRs,Masks,XFMs,Labels
S1,flair.mnc,mask.mnc,xfm.xfm,label.mnc
```

**Notes:**
- `Labels` column is optional for PT (pre-trained) mode
- All paths should be relative or absolute paths to image files
- Images must be co-registered to the same space
- XFMs are nonlinear transformations from template to subject space

## Available Classifiers

| Code | Algorithm |
|------|-----------|
| RF0 | Random Forest (default) |
| RF | Random Forest |
| NB | Naive Bayes |
| LDA | Linear Discriminant Analysis |
| QDA | Quadratic Discriminant Analysis |
| LR | Logistic Regression |
| KNN | K Nearest Neighbors |
| SVM | Support Vector Machines |
| Tree | Decision Tree |
| Bagging | Bagging Classifier |
| AdaBoost | AdaBoost Classifier |

## Output Files

| File | Description |
|------|-------------|
| `{Classifier}_{SubjectID}_Label.mnc` | Segmentation mask |
| `{Classifier}_{SubjectID}_Prob_Label_{N}.mnc` | Probability maps for each class |
| `{Classifier}_{SubjectID}_*.jpg` | Quality control images |
| `{Tissue}_Label.pkl` | Saved tissue histograms (training) |
| `{Classifier}_T1*.pkl` | Trained classifier models (training) |

## Data Files

- `*.mnc` - MINC format medical images
- `*.pkl` - Trained classifiers and tissue histograms
- `*.zip` - Compressed trained models
- `Av_T1.mnc`, `Av_FLAIR.mnc` - Template average images
- `SP_1.mnc` through `SP_9.mnc` - Spatial priors for each tissue class

## Reference

Dadar, M., & Collins, D. L. (2021). BISON: Brain tissue segmentation pipeline using T1-weighted magnetic resonance images and a random forest classifier. Magnetic Resonance in Medicine, 85(4), 1881-1894. https://doi.org/10.1002/mrm.28495

## License

Copyright (c) 2022 Mahsa Dadar
