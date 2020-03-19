# David De Vito - Last update: Mar19, 2020

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

# FORCE DEFAULTS FOR QUICK RUN
defRun = input('\nSkip unnecessary options? (y or n): ')
if defRun == 'y':
    subjects = ['515', '518', '520', '522', '525', '528', '530', '536', '543', '548', '554', '565', '571', '583', '593']
    # subjects = ['515', '518', '520', '525', '528', '530', '536', '543', '548', '554', '565', '571', '583', '593']
    combinedTRs = '1'
    mask_name = 'rFull_IPS_Binary_BinaryMask'
    featSelectInput = 'y'
    makePlot = 'n'
    trainOrTest = 'Test'

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
if defRun != 'y':
    subjects = []
    while True:
        if len(subjects) < 1: curSubject = input('\nInput subject numbers one at a time, pressing enter after each.\nWhen done entering subjects press enter to end list:\n')
        else:curSubject = input()
        if curSubject == "": break
        else: subjects.append(curSubject)

# Input analysis information
categoryInput = input('Enter category (e.g., Probe; SwitchProbe): ')
startTR = input('Enter start TR (1-4): ')
if defRun != 'y': combinedTRs = input('Enter combined TRs (1-2): ')
if categoryInput == 'SwitchProbe' or categoryInput == 'SwitchProbe':
    stanAlt = input('Analyze standard or alt stimIDs (i.e., S or A): ')

# Input mask information
if defRun != 'y':
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
if categoryInput == 'SwitchProbe' or categoryInput == 'RepeatProbe': categoryToBaseFeatureSelection = categoryInput
else: categoryToBaseFeatureSelection = 'Probe'
TRtoBaseFeatureSelection = 3
combinedTRsToBaseFeatureSelection = 1
if defRun != 'y': featSelectInput = input('\nUse Feature Selection? (y or n) (currently set at p<' + str(featSelectPercent) + '): ')

# Plot Data?
if defRun != 'y': makePlot = input('Plot the data? (y or n): ')

# COMPILE LISTS OF CATEGORIES
stimNames = ['ML', 'MR', 'BL', 'TP', 'BR']
catPrefixArray = {'Probe': ['LeftProbe', 'RightProbe'], 'Cued': ['Cued_Left', 'Cued_Right'], 'Uncued': ['Uncued_Left', 'Uncued_Right'],
                  'SwitchProbe': ['LeftSwitch', 'RightSwitch'], 'Switch_Cued': ['Switch_Cued_Left', 'Switch_Cued_Right'],
                  'RepeatProbe': ['LeftRepeat', 'RightRepeat'], 'Repeat_Cued': ['Repeat_Cued_Left', 'Repeat_Cued_Right']}
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
if defRun != 'y': trainOrTest = input('Train or Test Data Files: ')
if trainOrTest == 'Train': runsEntered = input('Enter runsToRemoveForTest 1,2,3: ') # 1,2,3
elif trainOrTest == 'Test': runsEntered = input('Enter runsUsed 1,2,3: ') # 1,2,3
for subjNum in subjects:

    # PRINT ANALYSIS INFO TO TERMINAL
    print(f'\nAnalyzing Subject: {subjNum}')

    # LOAD CATEGORY LABELS
    if categoryInput == 'Probe' or categoryInput == 'Cued':
        labels = pd.read_csv((baseDataFolder + str(subjNum) + catLabelFolder + "/Decoding_Timing_Cued_" + str(subjNum) + ".txt"), sep=",")
    elif categoryInput == 'Uncued':
        labels = pd.read_csv((baseDataFolder + str(subjNum) + catLabelFolder + "/Decoding_Timing_Uncued_" + str(subjNum) + ".txt"), sep=",")
    elif categoryInput == 'SwitchProbe' or categoryInput == 'RepeatProbe':
        if stanAlt == 'A':
            labels = pd.read_csv((baseDataFolder + str(subjNum) + catLabelFolder + "/Decoding_Timing_SwRepAlt_" + str(subjNum) + ".txt"), sep=",")
        else:
            labels = pd.read_csv((baseDataFolder + str(subjNum) + catLabelFolder + "/Decoding_Timing_SwRep_" + str(subjNum) + ".txt"), sep=",")
    # Separate Category Labels into Variables
    stimuli = labels['Event']
    session_labels = labels['Run']

    # Get list of runs used for analysis
    runsUsed = session_labels.unique().tolist()
    print(f'\nRuns Available: {runsUsed}')
    if trainOrTest == 'Train':
        runsToRemove = [int(i) for i in runsEntered.split(',') if i.isdigit()]
        runsUsed = [i for i in runsUsed if i not in runsToRemove]
    elif trainOrTest == 'Test':
        runsUsed = [int(i) for i in runsEntered.split(',') if i.isdigit()]
        runsUsed = [i for i in runsUsed]
    else:
        print('\nInvalid Entry.'); exit()

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
        if len(maskedActivations) < 1: maskedActivations = curRunMaskedActivations
        else: maskedActivations = np.vstack([maskedActivations, curRunMaskedActivations])

    # FEATURE SELECTION
    if featSelectInput == 'y':
        # If current time period is what you'd like to base feature selection on, then create new file, otherwise load selected voxels
        if categoryInput == categoryToBaseFeatureSelection and startTR == str(TRtoBaseFeatureSelection) and combinedTRs == str(combinedTRsToBaseFeatureSelection) and trainOrTest == 'Train':
            condition_mask = stimuli.isin(categoriesUsed) & session_labels.isin(runsUsed)
            stimuliAnalyzed = stimuli[condition_mask].copy()
            featSelect_maskedActivations = maskedActivations[condition_mask[session_labels.isin(runsUsed)]].copy()
            usedVox = f_classif(featSelect_maskedActivations, stimuliAnalyzed)[1] < featSelectPercent
            if categoryInput == 'SwitchProbe': np.savetxt(baseDataFolder + str(subjNum) + catLabelFolder + "UsedVoxelsSwi_" + str(runsToRemove[0]) + '_' + mask_name + ".csv", usedVox, delimiter=",")
            elif categoryInput == 'RepeatProbe': np.savetxt(baseDataFolder + str(subjNum) + catLabelFolder + "UsedVoxelsRep_" + str(runsToRemove[0]) + '_' + mask_name + ".csv", usedVox, delimiter=",")
            else: np.savetxt(baseDataFolder + str(subjNum) + catLabelFolder + "UsedVoxels_" + str(runsToRemove[0]) + '_' + mask_name + ".csv", usedVox, delimiter=",")
        else:
            # LOAD FEATURE SELECTION DATA FILE
            if trainOrTest == 'Train': usedVoxFirstRun = runsToRemove[0]
            elif trainOrTest == 'Test': usedVoxFirstRun = runsUsed[0]

            if categoryInput == 'SwitchProbe':
                usedVox = np.loadtxt((baseDataFolder + str(subjNum) + catLabelFolder + "UsedVoxelsSwi_" + str(usedVoxFirstRun) + '_' + mask_name + ".csv"), delimiter=",")
            elif categoryInput == 'RepeatProbe':
                usedVox = np.loadtxt((baseDataFolder + str(subjNum) + catLabelFolder + "UsedVoxelsRep_" + str(usedVoxFirstRun) + '_' + mask_name + ".csv"), delimiter=",")
            else:
                usedVox = np.loadtxt((baseDataFolder + str(subjNum) + catLabelFolder + "UsedVoxels_" + str(usedVoxFirstRun) + '_' + mask_name + ".csv"), delimiter=",")
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

        if len(allCatMeanActivations) < 1: allCatMeanActivations = curCatMaskedActivations_mean
        else: allCatMeanActivations = np.vstack([allCatMeanActivations, curCatMaskedActivations_mean])

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

    # Save data to file
    if categoryInput == 'SwitchProbe' or categoryInput == 'RepeatProbe' and stanAlt == 'A':
        if trainOrTest == 'Train': np.savetxt(baseDataFolder + str(subjNum) + catLabelFolder + "/AvgActivationByCat_" + categoryInput + "_" + str(startTR) + "_" + mask_name + "_" + str(combinedTRs) + "_FeatSel_" + featSelectInput + "_" + trainOrTest + "_" + str(runsToRemove[0]) + "A.csv", allCatMeanActivations_toWriteToFile, delimiter=",")
        elif trainOrTest == 'Test': np.savetxt(baseDataFolder + str(subjNum) + catLabelFolder + "/AvgActivationByCat_" + categoryInput + "_" + str(startTR) + "_" + mask_name + "_" + str(combinedTRs) + "_FeatSel_" + featSelectInput + "_" + trainOrTest + "_" + str(runsUsed[0]) + "A.csv", allCatMeanActivations_toWriteToFile, delimiter=",")
    else:
        if trainOrTest == 'Train': np.savetxt(baseDataFolder + str(subjNum) + catLabelFolder + "/AvgActivationByCat_" + categoryInput + "_" + str(startTR) + "_" + mask_name + "_" + str(combinedTRs) + "_FeatSel_" + featSelectInput + "_" + trainOrTest + "_" + str(runsToRemove[0]) + ".csv", allCatMeanActivations_toWriteToFile, delimiter=",")
        elif trainOrTest == 'Test': np.savetxt(baseDataFolder + str(subjNum) + catLabelFolder + "/AvgActivationByCat_" + categoryInput + "_" + str(startTR) + "_" + mask_name + "_" + str(combinedTRs) + "_FeatSel_" + featSelectInput + "_" + trainOrTest + "_" + str(runsUsed[0]) + ".csv", allCatMeanActivations_toWriteToFile, delimiter=",")

    print('\nComplete')