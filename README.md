# KernelProject
Project 2 Dylan Stewart

URGENT: MUST INSTALL PATTERN RECOGNITION TOOLBOX (prt) to run this code
 from https://github.com/covartech/PRT
and add these lines to your startup.m file (if not created just create one in your MATLAB folder)

addpath C:\Users\Dylan\Documents\MATLAB\prt
prtPath;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Kernel Methods Project Code and Results
These folders contain the code, results from training and testing with the data set we were given 
and the overview of the project.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FOLDERS: 

CODE
This contains all the code written by our group to train and validate each of the methods

DATA SETS
Data sets given to us for the project 25000 samples of 60 features and the labels

TRAINING AND TESTING RESULTS
mat files that contain our results from training on 80% and testing on 20% of the data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

To train with new data change line 8 of the Project2.m file to load the new data and labels

To train with new data uncomment the code in the training and cross validation section. If not the code
will test with new data (XTrain)
