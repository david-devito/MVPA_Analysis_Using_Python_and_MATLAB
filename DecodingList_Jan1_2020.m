%% CREATING LIST OF EVENTS FOR MVPA
% Loads onset files and creates text file containing one line per TR


%% INITIAL SETUP
%-------------------------------------------------------------------------
clear all
clc

% -------- SET MODALITY AS FMRI --------
spm('defaults','FMRI')

% >>>>>>>> INPUT INFORMATION <<<<<<<<
subjectNum = 543;
runsToRemove = [4 11]; % Runs to be removed from decoding analysis for high motion etc.
sessionsAnalyzed = [1 2 3];

% -------- DIRECTORIES --------
subjectDir = (['/home/data/DCM_WM/' num2str(subjectNum) '/']);
dataDir = ([subjectDir 'data/']);
analysisDir = ([subjectDir 'model/AvgActivationRegression/']);
scriptsDir = '/home/ddevito/SpatialVerbalfMRI/AnalysisScripts/';
outputFile = ([analysisDir 'Decoding_Timing_' num2str(subjectNum) '.txt']);

% -------- SCAN INFORMATION --------
numberSessions = length(sessionsAnalyzed);
numRunsPerSession = 4;
TR = 2;
dummyScans = 4;

% -------- INFORMATION USED FOR END OF EACH RUN --------
lastOnset = 792;
numProbeScans = 4;
endOfRunScans = 4;



%% COMBINING ONSET FILES AND DELETING UNNECESSARY ROWS
%-------------------------------------------------------------------------

% ----- LOAD AND COMBINE ONSET FILES -----
OnsetTimesFull = [];
for onsetFileNum = 1:length(sessionsAnalyzed)
    OnsetTimes = readtable([subjectDir 'behav/OnsetTimes_' num2str(subjectNum) '_' num2str(sessionsAnalyzed(onsetFileNum)) '_fmri.txt']);
    OnsetTimesFull = vertcat(OnsetTimesFull,OnsetTimes);
end

% ----- ADJUSTING ONSET TIMES FOR REMOVAL OF DUMMY SCANS -----
OnsetTimesFull{:,3} = OnsetTimesFull{:,3} - (dummyScans*TR);

% ----- DELETE UNNECESSARY ROWS IN ONSET TIMES ARRAY -----
OnsetTimesFull(contains(OnsetTimesFull{:,2},'InitStim') | contains(OnsetTimesFull{:,2},'Uncued'),:) = [];

% ----- CHANGING NAMES OF CONDITIONS IN ONSET TIMES ARRAY -----
OnsetTimesFull.Var2(OnsetTimesFull.Var5(:) == 0) = cellstr('Error');

% ----- CREATE DECODING LIST - ONE LINE PER TR -----
pasteLine = 1;
for numTrials = 1:height(OnsetTimesFull)
    
    % Skip runs marked above to be removed from analysis
    if ismember(OnsetTimesFull.Var1(numTrials), runsToRemove)
    else
        % If it's a probe Onset then rename the category appropriately
        if contains(OnsetTimesFull.Var2{numTrials}, ["Repeat","Switch"])
            stimString = char(OnsetTimesFull.Var2(numTrials));
            stimLoc = extractAfter(stimString,"-");
            if contains(OnsetTimesFull.Var2{numTrials}, ["LeftSwitch","LeftRepeat"]), OnsetTimesFull.Var2{numTrials} = strcat('LeftProbe-',stimLoc);
            else, OnsetTimesFull.Var2{numTrials} = strcat('RightProbe-',stimLoc);end
        end
        
        % If line is last onset of run then add probe TRs followed by end of run scans
        if OnsetTimesFull.Var3(numTrials) == lastOnset
            for lineNum = 1:numProbeScans
                DecodingList.Var1(pasteLine,1) = OnsetTimesFull.Var1(numTrials);
                DecodingList.Var2{pasteLine,1} = [OnsetTimesFull.Var2{numTrials} '_' num2str(lineNum)];
                pasteLine = pasteLine + 1;
            end
            for lineNum = 1:endOfRunScans
                DecodingList.Var1(pasteLine,1) = OnsetTimesFull.Var1(numTrials);
                DecodingList.Var2{pasteLine,1} = 'EndOfRunScans';
                pasteLine = pasteLine + 1;
            end
        else
            % Use the current onset and the next onset to get number of scans for current line's event
            numEventScans = (OnsetTimesFull.Var3(numTrials+1) - OnsetTimesFull.Var3(numTrials))/2;
            % Add information to decoding list variables
            for lineNum = 1:numEventScans
                DecodingList.Var1(pasteLine,1) = OnsetTimesFull.Var1(numTrials);
                DecodingList.Var2{pasteLine,1} = [OnsetTimesFull.Var2{numTrials} '_' num2str(lineNum)];
                pasteLine = pasteLine + 1;
            end
        end
    end
end

cd(analysisDir)
OnsetTimes_Table = table(DecodingList.Var1,DecodingList.Var2);
OnsetTimes_Table.Properties.VariableNames = {'Run' 'Event'};
writetable(OnsetTimes_Table,outputFile);

