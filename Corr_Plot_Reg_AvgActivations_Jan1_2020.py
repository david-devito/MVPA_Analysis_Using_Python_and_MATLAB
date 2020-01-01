# David De Vito - Last update: Jan 1, 2020

# Code used for analysis of data for SpatialSpatial Working Memory Task
# Asks for user input of categories and time periods
# Loads previously-created files containing average activations for each category during inputted time periods
# Correlates activations and allows for plotting correlations
# Regresses final correlation matrix with activation templates


# IMPORT PROCESSING PACKAGES
from sys import exit
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import heapq
from statistics import mean

# DIRECTORIES
baseDataFolder = '/home/data/DCM_WM/'
maskFolder = '/model/inv_ROIs/'
avgActivationFolder = '/model/VolsAsImages/'

# INPUT VARIABLES
# Input subject numbers
subjects = []
while True:
    if len(subjects) < 1: curSubject = input('Input subject numbers one at a time, pressing enter after each.\nWhen done entering subjects press enter to end list:\n')
    else:curSubject = input()
    if curSubject == "": break
    else: subjects.append(curSubject)

# Input analysis information
trainingCat = input('Enter training category (e.g., Probe): ')
startTR_Train = input('Enter training category start TR (1-4): ')
trainingCombinedTRs = input('Enter training combined TRs (1-2): ')
testingCat = input('Enter testing category (e.g., Cued): ')
startTR_Test = input('Enter testing category start TR (1-4): ')
testingCombinedTRs = input('Enter testing combined TRs (1-2): ')

# Input mask information
try:
    print('\nMasks Available: ')
    masksAvailable = [filename for filename in os.listdir(baseDataFolder + subjects[0] + maskFolder) if filename.startswith("r")]; print(*masksAvailable, sep = "\n")
    mask_name = input('Input a mask name from above list (e.g., ' + masksAvailable[0] + '): ')
    # Remove file extension from variable name
    mask_name = mask_name.split('.')[0]
except:
    print('\nInvalid Participant Number. Check Subject: ' + subjects[0]); exit()

# Plot Data?
makePlot = input('Plot the data? (y or n): ')

# Run Regression?
regressData = input('Run regression? (y or n): ')

# COMPILE LISTS OF TRAINED AND TESTED CATEGORIES
stimNames = ['ML', 'MR', 'BL', 'TP', 'BR']
catPrefixArray = {'Probe': ['LeftProbe', 'RightProbe'], 'Cued': ['Cued_Left', 'Cued_Right'], 'Uncued': ['Uncued_Left', 'Uncued_Right'],
                  'Switch_Probe': ['Switch_LeftProbe', 'Switch_RightProbe'], 'Switch_Cued': ['Switch_Cued_Left', 'Switch_Cued_Right'],
                  'Repeat_Probe': ['Repeat_LeftProbe', 'Repeat_RightProbe'], 'Repeat_Cued': ['Repeat_Cued_Left', 'Repeat_Cued_Right']}
categories_train = []; categories_test = []
try:
    for prefixNum in range(0, len(catPrefixArray[trainingCat])):
        for stimNum in range(0, len(stimNames)):
            categories_train.append(catPrefixArray[trainingCat][prefixNum] + '-' + stimNames[stimNum] + '_' + str(startTR_Train))
            categories_test.append(catPrefixArray[testingCat][prefixNum] + '-' + stimNames[stimNum] + '_' + str(startTR_Test))
except:
    print('\nInvalid Category Name.'); exit()

# PRINT ANALYSIS INFO TO TERMINAL
print(f'\nSubject: {subjects}')
print(f'Trained Time Period: {trainingCat}_{startTR_Train}')
print(f'Tested Time Period: {testingCat}_{startTR_Test}')
print(f'Mask_Name: {mask_name}')

avgActAcrossSubjects = []
for subjNum in range(0, len(subjects)):
    # LOAD CSV CONTAINING AVERAGE VOXEL ACTIVATIONS FOR EACH CATEGORY
    try:
        trainingActivations = pd.read_csv((baseDataFolder + str(subjects[subjNum]) + avgActivationFolder + "/AvgRepresentationDF_" + trainingCat + "_" + str(startTR_Train) + "_" + mask_name + "_" + str(trainingCombinedTRs) + ".csv"), sep=",")
        testingActivations = pd.read_csv((baseDataFolder + str(subjects[subjNum]) + avgActivationFolder + "/AvgRepresentationDF_" + testingCat + "_" + str(startTR_Test) + "_" + mask_name + "_" + str(testingCombinedTRs) + ".csv"), sep=",")
    except:
        print('\nFile does not exist. Check Subject: ' + subjects[subjNum]); exit()

    # Read in array of average activations for each category
    # Create list that contains average activations by voxel of all training categories and all testing categories
    avgActivationArray = []
    for trainingCatNum in range(0, trainingActivations.shape[0]):
        avgActivationArray.append(list(trainingActivations.iloc[trainingCatNum]))
    for testingCatNum in range(0, testingActivations.shape[0]):
        avgActivationArray.append(list(testingActivations.iloc[testingCatNum]))

    # Create correlation matrix of all training and testing categories
    corrAvgActivations = np.corrcoef(avgActivationArray)

    # Select part of correlation matrix that you are interested in (remember: (row,columns))
    corrAvgActivations = corrAvgActivations[10:, 0:10]; corrAvgActivations = np.around(corrAvgActivations, 4)

    # Convert correlation matrix into one long vector so that you can average across participants
    corrsAvgActAsVec = []
    for corrsConcat in range(0, len(corrAvgActivations)):
        corrsAvgActAsVec = np.concatenate((corrsAvgActAsVec, corrAvgActivations[corrsConcat]), axis=None)
    corrsAvgActAsVec = corrsAvgActAsVec.tolist()

    if len(avgActAcrossSubjects) < 1:
        avgActAcrossSubjects = corrsAvgActAsVec
    else:
        avgActAcrossSubjects = np.vstack([avgActAcrossSubjects, corrsAvgActAsVec])

# Only take average across subjects if than one subject being analyzed
if len(subjects) > 1:
    avgActAcrossSubjects = np.mean(avgActAcrossSubjects, axis=0)
avgActAcrossSubjects = np.around(avgActAcrossSubjects, 4)

# PLOTTING
if makePlot == 'y':
    if trainingCat == 'Probe':
        x_labels = ['LP-ML', 'LP-MR', 'LP-BL', 'LP-TP', 'LP-BR', 'RP-ML', 'RP-MR', 'RP-BL', 'RP-TP', 'RP-BR', 'CL-ML',
                   'CL-MR', 'CL-BL', 'CL-TP', 'CL-BR', 'CR-ML', 'CR-MR', 'CR-BL', 'CR-TP', 'CR-BR']
    elif trainingCat == 'Cued':
        x_labels = ['CL-ML', 'CL-MR', 'CL-BL', 'CL-TP', 'CL-BR', 'CR-ML', 'CR-MR', 'CR-BL', 'CR-TP', 'CR-BR', 'CL-ML',
                   'CL-MR', 'CL-BL', 'CL-TP', 'CL-BR', 'CR-ML', 'CR-MR', 'CR-BL', 'CR-TP', 'CR-BR']

    plotData = avgActAcrossSubjects; plotData = np.reshape(plotData, (10, 10))

    y_labels = categories_test
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(plotData,cmap="autumn",vmin=np.amin(plotData),vmax=np.amax(plotData))
    #plt.title(str(subjNum) + '_' + mask_name,fontsize=12,fontweight="bold",y=1.16)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_xticklabels([''] + x_labels,fontsize=10,rotation=50)
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax.set_yticklabels([''] + y_labels,fontsize=10)
    fig.tight_layout()
    plt.show()

# REGRESSION TEMPLATES
if regressData == 'y':
    # Spatial Attention
    spatialAttention = [1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1]
    # Cued Item Identity
    cuedItemIdentity = [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]
    # Stimulus Idenity - Opposite Side
    stimIDOppoSide = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    # Next In Sequence
    nextInSequence = [0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0]
    # Next In Sequence - Opposite Side
    nextInSequenceOppoSide = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
    # Distance Between Stimuli
    distBetweenStim = [0, 0.34, 0.14, 0.19, 0.32, 0.65, 1, 0.71, 0.83, 0.95, 0.34, 0, 0.32, 0.19, 0.14, 0.32, 0.65, 0.38, 0.49, 0.62, 0.14, 0.32, 0, 0.26, 0.23, 0.62, 0.95, 0.65, 0.8, 0.89, 0.19, 0.19, 0.26, 0, 0.26, 0.49, 0.83, 0.58, 0.65, 0.8, 0.32, 0.14, 0.23, 0.26, 0, 0.38, 0.71, 0.41, 0.58, 0.65, 0.65, 0.32, 0.62, 0.49, 0.38, 0, 0.34, 0.14, 0.19, 0.32, 1, 0.65, 0.95, 0.83, 0.71, 0.34, 0, 0.32, 0.19, 0.14, 0.71, 0.38, 0.65, 0.58, 0.41, 0.14, 0.32, 0, 0.26, 0.23, 0.83, 0.49, 0.8, 0.65, 0.58, 0.19, 0.19, 0.26, 0, 0.26, 0.95, 0.62, 0.89, 0.8, 0.65, 0.32, 0.14, 0.23, 0.26, 0]
    #Probability of Upcoming Stimulus
    probOfUpcoming = [0, 1, 0.3334, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0, 1, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0.3334, 0, 1, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0.3334, 0.3334, 0, 1, 0, 0, 0, 0, 0, 1, 0.3334, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.3334, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0, 1, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0.3334, 0, 1, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0.3334, 0.3334, 0, 1, 0, 0, 0, 0, 0, 1, 0.3334, 0.3334, 0.3334, 0]
    #Probability of Upcoming Stimulus - Opposite Side
    probOfUpcomingOppoSide = [0, 0, 0, 0, 0, 0, 1, 0.3334, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0, 1, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0.3334, 0, 1, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0.3334, 0.3334, 0, 1, 0, 0, 0, 0, 0, 1, 0.3334, 0.3334, 0.3334, 0, 0, 1, 0.3334, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0, 1, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0.3334, 0, 1, 0.3334, 0, 0, 0, 0, 0, 0.3334, 0.3334, 0.3334, 0, 1, 0, 0, 0, 0, 0, 1, 0.3334, 0.3334, 0.3334, 0, 0, 0, 0, 0, 0]

    # REGRESSION
    templatesUsedString = 'cuedItemIdentity, probOfUpcoming' # For Terminal Output
    templatesUsed = [cuedItemIdentity, probOfUpcoming]

    ones = np.ones(len(templatesUsed[0]))
    X = np.vstack((templatesUsed,ones)).T
    regressOutput = np.linalg.lstsq(X,avgActAcrossSubjects,rcond=None)
    # Calculate R-Squared Value
    regressResid = regressOutput[1]
    r2 = 1 - regressResid / (avgActAcrossSubjects.size * avgActAcrossSubjects.var())

    # Print Regression Output
    print(f'\nRegression Templates Used: {templatesUsedString}')
    print(f'Beta Values (last value is the intercept): {[ round(elem, 4) for elem in list(regressOutput[0]) ]}')
    print(f'R-Squared value of Model: {np.around(r2,4)}')
