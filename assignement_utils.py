import numpy as np
# Convert 'M' and 'F' categories into 0 and 1 respectively
# Assumes no other values than 'M' and 'F' are fed
def string2int(cat):
    if cat == 'F':
        return 1
    else:
        return 0

# Given an audio file MFCC features, and a model, split the features 
# in sequences of split_size, classify the sequences in male or female,
# average the sequences classification and returns 'M' or 'F'.
def predictGender(model, mfcc, split_size):
    features = []
    nb_batch = int((len(mfcc)/split_size))
    for i in range(nb_batch): 
        features.append(mfcc[i*split_size:(i+1)*split_size])
    
    features = np.array(features)
    preds = model.predict(features, batch_size = 1)
    pred = np.sum(preds, axis = 0)
    if(pred[0] > pred[1]):
        return 'M'
    else:
        return 'F'