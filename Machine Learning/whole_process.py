# This file calls all the python scripts and function required to carry out the whole Machine Learning Part of the project


print("\nEXTRACTING FEATURES ... ")
print("__________________________________________________")
# The csv files are provided in this link -> https://drive.google.com/drive/folders/1ciE6ksmzgpM0PFr-tloPrHtksggkvfMZ?usp=sharing
# Store all the csv files in the a folder named "green" to be able to smoothly run the code
# For each lesion 2 csv files are created: one from training dataset, one from testing dataset
#import segmentAllTest as segment 
print("\nThe csv files are ready.")




# print("__________________________________________________")
# In stage-1 true lesions are detected and sent to stage-2
# When you run stage-1 a .pickle file will be created which contains the dataset to be carried forward 

# When you run stage-2 a .pickle file will be created for each classifier
# This file contains the model, which is used in the testing part to predict the labels 

print("\n\nSTAGE 1 and STAGE 2 OF TRAINING IS EXECUTING ... ")
print("__________________________________________________")
# Run stage 1 for training dataset
import stage1DT as stage1Train
# Run stage 2 for training dataset
import stage2Classification as stage2Train


print("\n\nSTAGE 1 and STAGE 2 OF TESTING IS EXECUTING ... ")
print("__________________________________________________")
# Run stage 1 for test dataset
import stage1DTTest as stage1Test
# Run stage 2 for test dataset
import stage2ClassificationTest as stage2Test
