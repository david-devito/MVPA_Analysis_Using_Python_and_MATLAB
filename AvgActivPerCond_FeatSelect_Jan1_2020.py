# David De Vito - Last update: Jan 1, 2020

# Code used for analysis of data for SpatialSpatial Working Memory Task
# Asks for user input of categories and time periods
# Creates files containing average activations for each category during inputted time periods


# IMPORT PROCESSING PACKAGES
from sys import exit
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import image, plotting
from statistics import mean
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from nilearn.input_data import NiftiMasker
from nilearn.image import clean_img
from scipy import stats

# DIRECTORIES
baseDataFolder = '/home/data/DCM_WM/'
maskFolder = '/model/inv_ROIs/'
catLabelFolder = '/model/AvgActivationRegression/'

# SCAN INFORMATION
sessNumArray = ['1','1','1','1','2','2','2','2','3','3','3','3']
scanArray = ['08','09','10','11','07','08','09','10','07','08','09','10']
funcFilePrefix = "/Functional/CleanedScans/cleaned_ruadf00"

# INPUT VARIABLES
# Input subject Numbers
subjects = []
while True:
    if len(subjects) < 1: curSubject = input('\nInput subject numbers one at a time, pressing enter after each.\nWhen done entering subjects press enter to end list:\n')
    else:curSubject = input()
    if curSubject == "": break
    else: subjects.append(curSubject)

# Input analysis information
categoryInput = input('Enter category (e.g., Probe): ')
startTR = input('Enter start TR (1-4): ')
combinedTRs = input('Enter combined TRs (1-2): ')

# Input mask information
try:
    print('\nMasks Available: ')
    masksAvailable = [filename for filename in os.listdir(baseDataFolder + subjects[0] + maskFolder) if filename.startswith("r")]; print(*masksAvailable, sep = "\n")
    mask_name = input('Input a mask name from above list (e.g., ' + masksAvailable[0] + '): ')
    # Remove file extension from variable name
    mask_name = mask_name.split('.')[0]
except:
    print('\nInvalid Participant Number. Check Subject: ' + subjects[0]); exit()

# Feature Selection
featSelectPercent = .05
categoryToBaseFeatureSelection = 'Probe'
TRtoBaseFeatureSelection = 3
combinedTRsToBaseFeatureSelection = 1
featSelectInput = input('\nUse Feature Selection? (y or n) (currently set at p<' + str(featSelectPercent) + '): ')

# Plot Data?
makePlot = input('Plot the data? (y or n): ')

# COMPILE LISTS OF CATEGORIES
stimNames = ['ML', 'MR', 'BL', 'TP', 'BR']
catPrefixArray = {'Probe': ['LeftProbe', 'RightProbe'], 'Cued': ['Cued_Left', 'Cued_Right'], 'Uncued': ['Uncued_Left', 'Uncued_Right'],
                  'Switch_Probe': ['Switch_LeftProbe', 'Switch_RightProbe'], 'Switch_Cued': ['Switch_Cued_Left', 'Switch_Cued_Right'],
                  'Repeat_Probe': ['Repeat_LeftProbe', 'Repeat_RightProbe'], 'Repeat_Cued': ['Repeat_Cued_Left', 'Repeat_Cued_Right']}
categoriesUsed = []
try:
    for prefixNum in range(0, len(catPrefixArray[categoryInput])):
        for stimNum in range(0, len(stimNames)):
            categoriesUsed.append(catPrefixArray[categoryInput][prefixNum] + '-' + stimNames[stimNum] + '_' + str(startTR))
except:
    print('\nInvalid Category Name.'); exit()

# PRINT ANALYSIS INFO TO TERMINAL
print(f'\nSubjects: {subjects}')
print(f'Time Period: {categoryInput}_{startTR}')

# LOOP THROUGH SUBJECTS
for subjNum in subjects:

    # PRINT ANALYSIS INFO TO TERMINAL
    print(f'\nAnalyzing Subject: {subjNum}')

    # LOAD CATEGORY LABELS
    if categoryInput == 'Probe' or categoryInput == 'Cued':
        labels = pd.read_csv((baseDataFolder + str(subjNum) + catLabelFolder + "/Decoding_Timing_" + str(subjNum) + ".txt"), sep=",")
    elif categoryInput == 'Switch_Probe' or categoryInput == 'Repeat_Probe':
        labels = pd.read_csv((baseDataFolder + str(subjNum) + catLabelFolder + "/Decoding_Timing_Probe_SwRep_" + str(subjNum) + ".txt"), sep=",")
    # Separate Category Labels into Variables
    stimuli = labels['Event']
    session_labels = labels['Run']
    # Get list of runs used for analysis
    runsUsed = session_labels.unique().tolist()

    # Set up mask path and create masker
    mask_filename = (baseDataFolder + str(subjNum) + maskFolder + "/" + mask_name + ".nii")
    masker = NiftiMasker(mask_img=mask_filename, smoothing_fwhm=5, standardize = True)

    # LOAD AND MASK DATA FROM EACH RUN
    maskedActivations = []
    # NOTE: Don't need to remove runs that have a category with less than one instance, because you're standardizing by feature within each run
    for runNum in runsUsed:
        print(f'Analyzing Run: {runNum}')
        curRunFuncFileName = baseDataFolder + str(subjNum) + "/data/Session" + str(sessNumArray[runNum-1]) + funcFilePrefix + scanArray[runNum-1] + ".nii"
        curRunFuncFile_Masked = masker.fit_transform(curRunFuncFileName)
        # Standardize by Feature
        curRunMaskedActivations = StandardScaler().fit_transform(curRunFuncFile_Masked)

        if len(maskedActivations) < 1:
            maskedActivations = curRunMaskedActivations
        else:
            maskedActivations = np.vstack([maskedActivations, curRunMaskedActivations])

    # FEATURE SELECTION
    if featSelectInput == 'y':
        # If current time period is what you'd like to base feature selection on, then create new file, otherwise load selected voxels
        if categoryInput == categoryToBaseFeatureSelection and startTR == str(TRtoBaseFeatureSelection) and combinedTRs == str(combinedTRsToBaseFeatureSelection):
            condition_mask = stimuli.isin(categoriesUsed)
            stimuliAnalyzed = stimuli[condition_mask].copy()
            featSelect_maskedActivations = maskedActivations[condition_mask].copy()
            usedVox = f_classif(featSelect_maskedActivations, stimuliAnalyzed)[1] < featSelectPercent
            np.savetxt(baseDataFolder + str(subjNum) + catLabelFolder + "UsedVoxels_" + mask_name + ".csv", usedVox, delimiter=",")
        else:
            # LOAD FEATURE SELECTION DATA FILE
            usedVox = np.loadtxt((baseDataFolder + str(subjNum) + catLabelFolder + "UsedVoxels_" + mask_name + ".csv"), delimiter=",")
            # Convert to boolean
            usedVox = usedVox != 0

        maskedActivations = maskedActivations[:, usedVox]
        print('Number of voxels used after feature selection: ' + str(np.shape(maskedActivations)[1]))

    # GET AVERAGE ACTIVATION BY CONDITION
    allCatMeanActivations = []
    usedStimuli = stimuli[session_labels.isin(runsUsed)]
    for curCategory in categoriesUsed:
        curCatRows = (usedStimuli == curCategory)
        curCatMaskedActivations = maskedActivations[curCatRows]

        if combinedTRs == 2:
            #Get Next TR and Append to Trial Activations
            curCategory = curCategory[0:-1] + str((startTR+1))
            curCatRows = (usedStimuli == curCategory)
            combinedTRs_timecourses = maskedActivations[curCatRows]
            curCatMaskedActivations = np.vstack([curCatMaskedActivations, combinedTRs_timecourses])

        # Get Average Activation
        curCatMaskedActivations_mean = curCatMaskedActivations.mean(axis=0)

        if len(allCatMeanActivations) < 1:
            allCatMeanActivations = curCatMaskedActivations_mean
        else:
            allCatMeanActivations = np.vstack([allCatMeanActivations, curCatMaskedActivations_mean])

    # CORRELATION MATRIX
    if makePlot == 'y':
        dataToPlot = np.corrcoef(allCatMeanActivations)
        labels = categoriesUsed
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(dataToPlot,cmap="autumn",vmin=np.amin(dataToPlot),vmax=np.amax(dataToPlot))
        #plt.title(str(subjNum) + '_' + mask_name,fontsize=12,fontweight="bold",y=1.16)
        fig.colorbar(cax)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set_xticklabels([''] + labels,fontsize=10,rotation=50)
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set_yticklabels([''] + labels,fontsize=10)
        fig.tight_layout()
        plt.show()

    # Assign Column indices
    columnIX = np.arange(allCatMeanActivations.shape[1])
    # Add column indices to dataframe
    allCatMeanActivations_toWriteToFile = np.vstack([columnIX, allCatMeanActivations])

    np.savetxt(baseDataFolder + str(subjNum) + catLabelFolder + "/AvgActivationByCat_" + categoryInput + "_" + str(startTR) + "_" + mask_name + "_" + str(combinedTRs) + "_FeatSel_" + featSelectInput + ".csv", allCatMeanActivations_toWriteToFile, delimiter=",")
