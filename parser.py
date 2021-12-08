import sys
import scipy.io
import numpy as np
import os
import wfdb

WINDOW = 128
FS = 200

def qrs_detect(ECG, fs):
	winsize = 5 * fs * 60 # 5min 滑窗
	#winsize = 10 * fs # 10s 滑窗
	NB_SAMP = len(ECG)
	peaks = []
	if NB_SAMP < winsize:
		peaks.extend(p_t_qrs(ECG, fs))
		peaks = np.array(peaks)
		peaks = np.delete(peaks, np.where(peaks >= NB_SAMP-2*fs)[0]) # 删除最后2sR波位置
	else:
		# 5分钟滑窗检测，重叠5s数据
		count = NB_SAMP // winsize
		for j in range(count+1):
			if j == 0:
				ecg_data = ECG[j*winsize: (j+1)*winsize]

				peak = p_t_qrs(ecg_data, fs)
				peak = np.array(peak)
				peak = np.delete(peak, np.where(peak >= winsize-2*fs)[0]).tolist() # 删除5分钟窗口最后2sR波位置

				peaks.extend(map(lambda n: n+j*winsize, peak))
			elif j == count:
				ecg_data = ECG[j*winsize-5*fs: ]
				if len(ecg_data) == 0:
					pass
				else:
					peak = p_t_qrs(ecg_data, fs)
					peak = np.array(peak)
					peak = np.delete(peak, np.where(peak <= 2*fs)[0]).tolist() # 删除最后多余窗口前2sR波位置

					peaks.extend(map(lambda n: n+j*winsize-5*fs, peak))
			else:
				ecg_data = ECG[j*winsize-5*fs: (j+1)*winsize]
				peak = p_t_qrs(ecg_data, fs)
				peak = np.array(peak)
				peak = np.delete(peak, np.where((peak <= 2*fs) | (peak >= winsize-2*fs))[0]).tolist() # 删除中间片段5分钟窗口前2s和最后2sR波位置

				peaks.extend(map(lambda n: n+j*winsize-5*fs, peak))

	peaks = np.array(peaks)
	peaks = np.sort(peaks)
	dp = np.abs(np.diff(peaks))

	final_peaks = peaks[np.where(dp >= 0.2*fs)[0]+1]

	return final_peaks

def load_data(sample_path):
    print ("Preprocessing data at: {}".format(sample_path))
    sig, fields = wfdb.rdsamp(sample_path)
    #Only consider first lead
    sig = sig[:, 1]

    length = len(sig)
    fs = fields['fs']
    fs_comment = fields['comments']
    ann_ref = wfdb.rdann(sample_path, 'atr')
    #Find the annotated peak and labels
    ann_note = np.array(ann_ref.aux_note)
    ann_peak = np.array(ann_ref.sample)

    #print ("SIG: {}, LENGTH: {}, FS: {}, ANN: {}".format(sig, length, fs_comment, ann_note))
    #Take sample centered at r_peak with size WINDOW
    current_rhythm = 0
    trainset = np.zeros((len(ann_note),128))
    traintarget = np.zeros((len(ann_note),2))
    for i, (note, peak) in enumerate(zip(ann_note, ann_peak)):

        #Check current rhythm
        if (note == '(AFIB') or (note == '(AFL'):
            #Change current class to 1
            current_rhythm = 1
        else:
            current_rhythm = 0


        #Decenter the sample to WINDOW size if peak at start or end
        if (peak < ((WINDOW)/2 - 1)):
            sample_wave = sig[:WINDOW]
            sample_ann = current_rhythm
        elif (peak > (len(sig) - (WINDOW)/2)):
            sample_wave = sig[-WINDOW:]
            sample_ann = current_rhythm
        else:
            sample_wave = sig[int(peak-(WINDOW/2)+1):int(peak+(WINDOW/2)+1)]
            sample_ann = current_rhythm

        #print (sample_wave)
        #print (sample_ann)

        tmp = np.zeros(2,)
        np.put(tmp,sample_ann, 1)
        trainset[i] = sample_wave
        traintarget[i] = tmp

        #print (trainset)
        #print (traintarget)

    return trainset, traintarget

if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()

    #initialize trainset and traintarget
    trainset = np.zeros((1,128))
    traintarget = np.zeros((1,2))
    #two classes, ["A", "N"]

    for i, sample in enumerate(test_set):
        sample_path = os.path.join(DATA_PATH, sample)

        #load data at sample_path and save into train_set and train_target
        sample_train_set, sample_train_target = load_data(sample_path)

        trainset = np.vstack((trainset, sample_train_set))
        traintarget = np.vstack((traintarget, sample_train_target))

        trainset = trainset[1:]
        traintarget = traintarget[1:]

        #print (trainset)
        #print (traintarget)
        #break

    #Delete the first row of train data

    scipy.io.savemat('trainingset.mat', mdict={'trainset': trainset, 'traintarget': traintarget})
