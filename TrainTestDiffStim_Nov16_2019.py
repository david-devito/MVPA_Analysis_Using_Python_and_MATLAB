# IMPORT PROCESSING PACKAGES
from sys import exit
import numpy as np
import random
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image
from statistics import mean
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelBinarizer, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, f_regression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score
from nilearn.input_data import NiftiMasker
from nilearn.image import clean_img
# IMPORT CLASSIFIERS
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression, RidgeClassifier, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


# 520 - remove run 4 - less than minimum in a cued item category for run 4


# INPUT VARIABLES
testingNextInSequence = 'No'  # Yes or No  ## Needs to be setup
subjNum = 536
trainingCat = 'Probe'; startTR_train = 4; combinedTRs_train = 1
testingCat = 'Probe'; startTR_test = 4; combinedTRs_test = 1
repetitions = 6  # number of decoding repetitions

side = 'Both'
MaskOrTMap = 'Mask'  # Mask or TMap
decodingResultsDir = '/home/data/DCM_WM/MVPA_Results/' + str(subjNum) + '/'
if MaskOrTMap == 'TMap': MaskFolder = 'VolsAsImages'
else: MaskFolder = 'inv_ROIs'
betaFolder = 'VolsAsImages'

# CLASSIFIERS
alphas = [0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 5, 10, 50, 100, 1000, 10000, 100000]

# SCAN INFORMATION
sessNumArray = ['1','1','1','1','2','2','2','2','3','3','3','3']
scanArray = ['08','09','10','11','07','08','09','10','07','08','09','10']
funcFilePrefix = "/Functional/CleanedScans/cleaned_ruadf00"

# MASK ARRAY
maskNames = []
#maskNames.append('rFull_Visual_Cortex_Binary_Binary' + MaskOrTMap)
#maskNames.append('rFull_IPS_Binary_Binary' + MaskOrTMap)
#maskNames.append('rFull_FEF_Binary_Binary' + MaskOrTMap)
maskNames.append('rLeft_Visual_Cortex_Binary_Binary' + MaskOrTMap)
maskNames.append('rLeft_IPS_Binary_Binary' + MaskOrTMap)
maskNames.append('rLeft_FEF_Binary_Binary' + MaskOrTMap)
maskNames.append('rRight_Visual_Cortex_Binary_Binary' + MaskOrTMap)
maskNames.append('rRight_IPS_Binary_Binary' + MaskOrTMap)
maskNames.append('rRight_FEF_Binary_Binary' + MaskOrTMap)

# FUNCTIONS
# Function creates the lists of trained and tested categories
def categoriesList(curCat, curStartTR, catPrefix, stimNames):
    categories_temp = []
    for prefixNum in range(0,len(catPrefix)):
        for stimNum in range(0,len(stimNames)):
            categories_temp.append(catPrefix[prefixNum] + '-' + stimNames[stimNum] + '_' + str(curStartTR))
    return categories_temp

# Function calculates multiclass area under the curve
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

# Function combines data from successive TRs
def combineTRs(curConditionMask, curFMRI_Masked):
    curMaskedTimeCourses = []
    for conditioni in range(0, curConditionMask["Event"].size):
        if curConditionMask["Event"].iloc[conditioni]:
            curTRActivation = curFMRI_Masked[conditioni]
            nextTRActivation = curFMRI_Masked[conditioni + 1]
            curActivation = np.mean(np.vstack([curTRActivation, nextTRActivation]), axis=0, keepdims=True)
            if len(curMaskedTimeCourses) < 1:
                curMaskedTimeCourses = curActivation
            else:
                curMaskedTimeCourses = np.vstack([curMaskedTimeCourses, curActivation])
    return curMaskedTimeCourses



# ASSEMBLE LISTS OF TRAINED AND TESTED CATEGORIES
if testingCat == 'Uncued': behavFileName = 'Decoding_Timing_Uncued_'
else: behavFileName = 'Decoding_Timing_'

stimNames = ['ML', 'MR', 'BL', 'TP', 'BR']
catPrefixArray = {'Probe': ['LeftProbe', 'RightProbe'], 'Cued': ['Cued_Left', 'Cued_Right'], 'Uncued': ['Uncued_Left', 'Uncued_Right']}
curCatArray = [trainingCat, testingCat]; startTRArray = [startTR_train, startTR_test]
for getCats in range(0, len(curCatArray)):
    curCat = curCatArray[getCats]; curStartTR = startTRArray[getCats]
    try: catPrefix = catPrefixArray[curCat]
    except: print('Incorrect Category Name'); exit()
    # Run Function to generate training and testing category lists
    if getCats == 0: categories_train = categoriesList(curCat, curStartTR, catPrefix, stimNames)
    else: categories_test = categoriesList(curCat, curStartTR, catPrefix, stimNames)


# PRINT RUN INFO TO TERMINAL
print(f'\nSubject: {subjNum}')
print(f'Trained Categories: {categories_train}')
print(f'Tested Categories: {categories_test}')

# BEHAVIORAL INFORMATION
# Load behavioral file
labels = pd.read_csv(("/home/data/DCM_WM/" + str(subjNum) + "/model/" + betaFolder + "/" + behavFileName + str(subjNum) + ".txt"), sep=",")
# Separate Behavioural Data into Variables
session_labels = labels['Run']; stimuli = labels['Event']

# COMBINING CATEGORIES IF NEEDED
'''stimuli = stimuli.replace("LeftSwitchMatch_Left-BL_1", "LeftProbe")
stimuli = stimuli.replace("LeftSwitchNonMatch_Left-BL_1", "LeftProbe")'''
# print(stimuli.unique()); exit()

# BEGIN DECODING
for currentROI in range(0, len(maskNames)):
    # GET CURRENT MASK
    mask_name = maskNames[currentROI]
    mask_filename = ("/home/data/DCM_WM/" + str(subjNum) + "/model/" + MaskFolder + "/" + mask_name + ".nii")
    masker = NiftiMasker(mask_img=mask_filename, smoothing_fwhm=2, standardize = True)

    # Visualize the mask on the subject's anatomical image
    # plotting.plot_roi(mask_filename, bg_img=("/home/data/DCM_WM/" + str(subjNum) + "/model/" + betaFolder + "/1.nii"), cmap='Paired')
    # plotting.show(); exit()

    for repeatCode in range(0, repetitions):
        print(f"\nROI: {maskNames[currentROI]}")
        print(f"Repeating decoding....Current repetition: {repeatCode+1} of {repetitions}\n")

        #Reset Arrays
        fmri_masked_train = []; fmri_masked_test = []
        fmri_masked = []

        # Get runs used for analysis
        runsUsed = session_labels.unique().tolist()

        # Create a dataframe to hold final stimuli and label information
        stimuli_final = pd.DataFrame(columns=["Event"])
        session_labels_final = pd.DataFrame(columns=["Run"])

        for runNum in runsUsed:
            print(runNum)
            funcFileName = "/home/data/DCM_WM/" + str(subjNum) + "/data/Session" + str(sessNumArray[runNum - 1]) + funcFilePrefix + scanArray[runNum - 1] + ".nii"

            # CHECK IF ANY CATEGORY HAS LESS THAN 1 INSTANCE IN THIS RUN AND SKIP RUN IF IT DOES
            catHasLessThanOne = 0
            curRunLabels = stimuli[labels["Run"] == runNum]
            categoriesArray = [categories_train, categories_test]
            for trainOrTest in range(0,len(categoriesArray)):
                curCategories = categoriesArray[trainOrTest]
                for cat in curCategories:
                    if curRunLabels.eq(cat).sum() < 1:
                        catHasLessThanOne = 1
                        print(f"\n Run Number {runNum} removed from analysis (contains category with less than 1 trial) \n")
            # Remove runs across all TR's
            if subjNum == 520:
                if runNum == 4: catHasLessThanOne = 1

            if catHasLessThanOne < 1:
                fmri_masked_run = masker.fit_transform(funcFileName)
                # Add current run trials to stimuli and session labels array
                X = curRunLabels.index.values  # Indices of all trials in this run
                stimuli_run = pd.DataFrame(columns=["Event"], data=stimuli[X])
                session_labels_run = pd.DataFrame(columns=["Run"], data=session_labels[X])
                # Append arrays for this run to compiled arrays
                if len(fmri_masked) < 1: fmri_masked = fmri_masked_run
                else: fmri_masked = np.vstack([fmri_masked, fmri_masked_run])
                stimuli_final = stimuli_final.append(stimuli_run,ignore_index=True)
                session_labels_final = session_labels_final.append(session_labels_run,ignore_index=True)

        fmri_masked_train = fmri_masked; stimuli_final_train = stimuli_final; session_labels_final_train = session_labels_final
        fmri_masked_test = fmri_masked; stimuli_final_test = stimuli_final; session_labels_final_test = session_labels_final


        # RESTRICT ANALYSIS ARRAYS TO ONLY BETAS IN RELEVANT CATEGORIES
        condition_mask_train = stimuli_final_train.isin(categories_train); condition_mask_test = stimuli_final_test.isin(categories_test)
        current_stimuli_final_train = stimuli_final_train[condition_mask_train["Event"]].copy(); current_stimuli_final_test = stimuli_final_test[condition_mask_test["Event"]].copy()

        # Compile masked timecourses arrays, either single TR or combined TRs
        if combinedTRs_train == 1: masked_timecourses_train = fmri_masked_train[condition_mask_train["Event"]].copy()
        elif combinedTRs_train == 2: masked_timecourses_train = combineTRs(condition_mask_train, fmri_masked_train)
        if combinedTRs_test == 1: masked_timecourses_test = fmri_masked_test[condition_mask_test["Event"]].copy()
        elif combinedTRs_test == 2: masked_timecourses_test = combineTRs(condition_mask_test, fmri_masked_test)


        # STANDARDIZE THE FEATURES
        scaler = StandardScaler()
        #scaler = RobustScaler()
        masked_timecourses_train = scaler.fit_transform(masked_timecourses_train); masked_timecourses_test = scaler.transform(masked_timecourses_test)

        session_labels_final_train = session_labels_final_train[condition_mask_train["Event"]].copy(); session_labels_final_train = session_labels_final_train["Run"].copy()
        session_labels_final_test = session_labels_final_test[condition_mask_test["Event"]].copy(); session_labels_final_test = session_labels_final_test["Run"].copy()

        # Altering test categories stimulus names to match the training categories
        if trainingCat == testingCat: pass
        elif trainingCat == 'Probe':
            # Change the stimulus description to match the categories of the training stimuli
            if testingCat == 'Cued':
                current_stimuli_final_test["Event"] = current_stimuli_final_test["Event"].replace({'Cued_Left': 'LeftProbe', 'Cued_Right': 'RightProbe'}, regex=True)
            elif testingCat == 'Uncued':
                current_stimuli_final_test["Event"] = current_stimuli_final_test["Event"].replace({'Uncued_Left': 'LeftProbe', 'Uncued_Right': 'RightProbe'}, regex=True)
        # Change the stimulus associated delay number to match the corresponding number of the training stimuli
        current_stimuli_final_test["Event"] = current_stimuli_final_test["Event"].replace({"\d": str(startTR_train)}, regex=True)

        # Changing tested stimulus to next in sequence
        if trainingCat == 'Probe' and testingCat == 'Cued' and testingNextInSequence == 'Yes':
            current_stimuli_final_test["Event"] = current_stimuli_final_test["Event"].replace({'ML':'MR','MR':'BL','BL':'TP','TP':'BR','BR':'ML'}, regex=True)

        # GET LIST OF RUNS USED IN ANALYSIS
        # Runs that have more than 1 value for each category for both train and test sets
        runList_train = (session_labels_final_train.unique()).tolist()
        runList_test = (session_labels_final_test.unique()).tolist()
        testGroupArray = list(set(runList_train) & set(runList_test))


        # RESET COUNTER VARIABLES EVERY RUN
        runAvgArray = [0] * len(alphas)
        for classRun in range(0, len(testGroupArray)):
            testGroup = []; testGroup.append(testGroupArray[classRun])

            trainGroup = []
            for trainGroupPopulate in range(0, len(testGroupArray)):
                if trainGroupPopulate == classRun: pass  # Skip the current test group when populating trainGroup
                else:
                    trainGroup.append(testGroupArray[trainGroupPopulate])
            # Get indices of testGroup and trainGroup
            testGroupIX = session_labels_final_test.isin(testGroup).copy()
            trainGroupIX = session_labels_final_train.isin(trainGroup).copy()


            def compileFinalArrays(curCategories, stimList, groupIX, masked_timecourses):
                cat_List = []
                for cat in curCategories:
                    cat_List.append(stimList["Event"][groupIX].eq(cat).sum())
                min_cat = min(cat_List)

                removeArray = []
                for catNum in range(0, len(curCategories)):
                    if cat_List[catNum] > min_cat:
                        # Get list of trials that have this category
                        catTrials = np.where(stimList[groupIX] == curCategories[catNum])[0]
                        # Create array of trials that need to be removed
                        removeRow = catTrials[np.random.choice(len(catTrials), size=(cat_List[catNum] - min_cat), replace=False)]
                        removeArray = np.concatenate([removeArray, removeRow])
                        removeArray = sorted(removeArray.astype(int), reverse=False)
                # Remove Rows from Trial Number Indices(X)
                curLabels = stimList["Event"][groupIX]
                X = curLabels.index.values
                X = np.delete(X, removeArray, axis=0)
                cur_masked_timecourses = masked_timecourses[groupIX]
                cur_masked_timecourses = np.delete(cur_masked_timecourses, removeArray, axis=0)
                current_stimuli_cur = pd.DataFrame(columns=["Event"], data=stimList["Event"][X])
                return cur_masked_timecourses, current_stimuli_cur

            # Compile Training and Testing Data for this Testing Group
            [cur_masked_timecourses_train, current_stimuli_cur_train] = compileFinalArrays(categories_train, current_stimuli_final_train, trainGroupIX, masked_timecourses_train)
            [cur_masked_timecourses_test, current_stimuli_cur_test] = compileFinalArrays(categories_test, current_stimuli_final_test, testGroupIX, masked_timecourses_test)


            # FEATURE SELECTION
            percentFeaturesToSelect = 0.10
            numVoxToKeep = int((np.shape(masked_timecourses_train)[1])*percentFeaturesToSelect)

            for alphaVal in range(0, len(alphas)):
                classPre = 'ridgesmooth5'
                classifiers = RidgeClassifier(alpha=alphas[alphaVal], fit_intercept=True, normalize=True)
                #classifiers = LogisticRegression(C=alphas[alphaVal], solver='sag', multi_class='multinomial', max_iter=10000)
                # classifiers = GaussianNB()
                #classifiers = SVC(C=alphas[alphaVal], kernel="linear")



                clf = Pipeline([('anova', SelectKBest(f_classif, k=numVoxToKeep)),('svc', classifiers)])

                # RUN CLASSIFICATION
                predictedLabels = clf.fit(cur_masked_timecourses_train,current_stimuli_cur_train["Event"]).predict(cur_masked_timecourses_test)
                #Multiclass AREA UNDER CURVE
                #aucArray.append(multiclass_roc_auc_score(current_stimuli_cur_test["Event"][testGroupIX], predictedLabels))
                corRunNum = (predictedLabels == current_stimuli_cur_test["Event"][testGroupIX]).sum()
                stimRunNum = len(current_stimuli_cur_test["Event"][testGroupIX])
                runAvgArray[alphaVal] = runAvgArray[alphaVal] + (corRunNum/stimRunNum)
                #print(str(round(corRunNum/stimRunNum,2)) + ' - alpha: ' + str(alphas[alphaVal]))
                #runAvgArray.append(corRunNum/stimRunNum)

        # SUMMARY DATA FOR CURRENT CLASSIFIER RUN
        runAvgArray = [x / len(testGroupArray) for x in runAvgArray]
        runAvgArray = [round(x,4) for x in runAvgArray]
        print(runAvgArray)
        #weightedACC = round(correctNum/totalNum, 4)
        #finalClass = alphas[alphaVal]
        #avgACC = round(mean(runAvgArray),4)
        #avgAUC = round(mean(aucArray),4)

        # Write results to text file
        #resultsArray = str(subjNum) + '\t' + str(avgACC) + '\t' + str(weightedACC) + '\t' + str(avgAUC) + '\t' + mask_name + '\t' + str(finalClass) + '\n'
        if trainingCat == testingCat:
            file = open(decodingResultsDir + classPre + 'Decoding_Results_TrainAndTest-' + testingCat + '_TR' + str(startTR_train) + '_' + MaskOrTMap + '_' + side + '_' + str(subjNum) + '.txt', 'a')
        else:
            if testingNextInSequence == 'Yes':
                file = open(decodingResultsDir + classPre + 'Decoding_Results_Train-' + trainingCat + '_TR' + str(startTR_train) + '_' + str(combinedTRs_train) + '_Test-' + testingCat + '_TR' + str(startTR_test) +  '_' + str(combinedTRs_test) + '_NIS_' + MaskOrTMap + '_' + side + '_' + str(subjNum) + '.txt', 'a')
            elif testingNextInSequence == 'No':
                file = open(decodingResultsDir + classPre + 'Decoding_Results_Train-' + trainingCat + '_TR' + str(startTR_train) + '_' + str(combinedTRs_train) + '_Test-' + testingCat + '_TR' + str(startTR_test) +  '_' + str(combinedTRs_test) + '_' + MaskOrTMap + '_' + side + '_' + str(subjNum) + '.txt', 'a')

        for writei in range(0,len(runAvgArray)):
            resultsArray = str(subjNum) + '\t' + str(runAvgArray[writei]) + '\t' + mask_name + '\t' + str(alphas[writei]) + '\n'
            file.write(resultsArray)
        file.close()


print(f'\nSubject: {subjNum}')
print(f'Trained Categories: {categories_train}')
print(f'Trained Combined TRs: {combinedTRs_train}')
print(f'Tested Categories: {categories_train}')
print(f'Tested Combined TRs: {combinedTRs_test}')
#print(cm)

'''labels = categories_train
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm, cmap="autumn", vmin=np.amin(cm), vmax=np.amax(cm))
plt.title(str(alphas[0]),fontsize=12,fontweight="bold",y=1.16)
fig.colorbar(cax)

#ax.xaxis.set_major_locator(plt.MaxNLocator(10))
#ax.set_xticklabels([''] + labels, fontsize=8, rotation=50)
ax.yaxis.set_major_locator(plt.MaxNLocator(10))
ax.set_yticklabels([''] + labels, fontsize=8)
fig.tight_layout()
plt.show()'''




# Output/Save as Pandas Dataframe
#pd.DataFrame(session_labels_final).to_csv("/home/data/DCM_WM/Decoding/DataSet4/308_VC_VerbalDelay_SessionsLabels.csv")
#pd.DataFrame(masked_timecourses).to_csv("/home/data/DCM_WM/Decoding/DataSet4/308_VC_VerbalDelay_Data.csv")
#pd.DataFrame(current_stimuli_final).to_csv("/home/data/DCM_WM/Decoding/DataSet4/308_VC_VerbalDelay_StimulusLabels.csv")
