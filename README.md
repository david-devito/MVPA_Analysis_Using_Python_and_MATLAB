# MVPA_Analysis_Using_Python_and_MATLAB

Scripts are listed in order of how they would likely be run.

Script: DecodingList_...
Purpose: Loads onset timing files and creates a text file where each line represents the event at every TR.

Script: AvgActivPerCond_FeatSelect...
Purpose: Loads text file created by DecodingList... Script. Creates a csv file that contains the average activation per voxel of each condition/category in the experiment.

Script: Corr_Plot_Reg_AvgActivations...
Purpose: Loads the csv file created by AvgActivPerCond_FeatSelect... Script. Allows input of two time periods in experiment. Creates a correlation matrix across the two time periods that correlates each condition/category. Runs regression of correlation matrix with templates of possible hypotheses.
