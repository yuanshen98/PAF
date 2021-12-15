import numpy as np
import os
import sys
import time
import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
from keras.models import load_model

WINDOW = 128

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def challenge_entry(sample_path, model):
    print ("Predicting Sample: {}".format(sample_path))
    sig, _, fs = load_data(sample_path)
    sig = sig[:, 1]
    end_points = []

    #load model
    classes = ["N", "A"]

    #Find peaks
    r_peaks = qrs_detect(sig, fs=200)


    #Sample signal centered at peaks
    current_rhythm = 0
    start_end = []
    #print (len(r_peaks))
    count = 0
    incr = int(len(r_peaks)/1000)+1
    for index in range(0, len(r_peaks), incr):

        peak = r_peaks[index]
        if (peak < ((WINDOW)/2 - 1)):
            sample_wave = sig[:WINDOW]
        elif (peak > (len(sig) - (WINDOW)/2)):
            sample_wave = sig[-WINDOW:]
        else:
            sample_wave = sig[int(peak-(WINDOW/2)+1):int(peak+(WINDOW/2)+1)]
        
        #print (sample_wave.shape)
        sample_wave = np.reshape(sample_wave, (1,128))
        #sample_wave = np.expand_dims(sample_wave, axis=2)
        #print (sample_wave.shape)
        #predict sample
        prob = model.predict(sample_wave)
        ann = np.argmax(prob)
        sample_ann = classes[ann]
        #if sample_ann == "A":
            #print (peak)
            #print (sample_ann)
        if current_rhythm==0 :
            if sample_ann == "A":
                start_end.extend([peak, peak])
                current_rhythm = 1
        else:
            if sample_ann == "A":
                start_end[1] = peak
            else:
                start_end[1] = r_peaks[index-1]
                end_points.append(start_end)
                current_rhythm = 0
                start_end = []
        
        #print (count)
        count = count +1
    pred_dict = {'predict_endpoints': end_points}

    return pred_dict

if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    test_set = open(os.path.join(DATA_PATH, 'RECORDS_II'), 'r').read().splitlines()
    print ("Loading model...")
    model = load_model("weights-best_k2_r0.hdf5")
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        start_time = time.time()
        pred_dict = challenge_entry(sample_path, model)

        print (time.time()-start_time)
        save_dict(os.path.join(RESULT_PATH, sample +'.json'), pred_dict)
        #if (i>5):
        #break
