#!/usr/bin/env python3

import numpy as np
import os
import sys

import wfdb
from utils import qrs_detect, comp_cosEn, save_dict

"""
Written by:  Xingyao Wang, Chengyu Liu
             School of Instrument Science and Engineering
             Southeast University, China
             chengyu@seu.edu.cn

Save answers to '.json' files, the format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]]}.
"""


def load_data(sample_path):
    print (wfdb.rdsamp(sample_path))
    print ("*******************************")
    print (wfdb.rdann(sample_path, 'atr'))
    print ("*******************************")
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']
    fs_comment = fields['comments']
    ann_ref = wfdb.rdann(sample_path, 'atr')
    ann_note = np.array(ann_ref.aux_note)
    ann_peak = np.array(ann_ref.sample)
    print ("SIG: {}, LENGTH: {}, FS: {}, ANN: {}".format(sig, length, fs_comment, ann_note))

    print ("ref peak: {}".format(ann_peak))
    #print (sig.shape)
    print ("*************************")
    print (ann_note.shape)
    print (ann_note)
    print (ann_peak.shape)
    print (ann_peak)
    return sig, length, fs


def ngrams_rr(data, length):
    grams = []
    for i in range(0, length - 12, 12):
        grams.append(data[i: i + 12])
    return grams


def challenge_entry(sample_path):
    """
    This is a baseline method.
    """

    sig, _, fs = load_data(sample_path)
    sig = sig[:, 1]
    end_points = []

    r_peaks = qrs_detect(sig, fs=200)
    #print("RPEAKS: {}".format(r_peaks))
    print (r_peaks.shape)
    rr_seq = np.diff(r_peaks) / fs
    len_rr = len(rr_seq)

    rr_seq_slice = ngrams_rr(rr_seq, len_rr)
    # print ("RR_SEQ_SLICE: {}".format(rr_seq_slice))
    '''
    is_af = []
    for rr_period in rr_seq_slice:
        cos_en, _ = comp_cosEn(rr_period)
        if cos_en <= -1.4:
            is_af.append(0)
        else:
            is_af.append(1)
    is_af = np.array([[j] * 12 for j in is_af]).flatten()
    rr_seq_last = rr_seq[-12: ]
    cos_en, _ = comp_cosEn(rr_seq_last)
    if cos_en <= -1.4:
        is_af_last = 0
    else:
        is_af_last = 1

    len_rr_remain = len_rr - int(12*len(rr_seq_slice))
    is_af = np.concatenate((is_af, np.array([is_af_last] * len_rr_remain).flatten()), axis=0)

    if np.sum(is_af) == len(is_af):
        end_points.append([0, len(sig)-1])
    elif np.sum(is_af) != 0:
        state_diff = np.diff(is_af)
        start_r = np.where(state_diff==1)[0] + 1
        end_r = np.where(state_diff==-1)[0] + 1

        if is_af[0] == 1:
            start_r = np.insert(start_r, 0, 0)
        if is_af[-1] == 1:
            end_r = np.insert(end_r, len(end_r), len(is_af)-1)
        start_r = np.expand_dims(start_r, -1)
        end_r = np.expand_dims(end_r, -1)
        start_end = np.concatenate((r_peaks[start_r], r_peaks[end_r]), axis=-1).tolist()
        end_points.extend(start_end)

    pred_dcit = {'predict_endpoints': end_points}

    return pred_dcit
    '''


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)

        # load_data(sample_path)

        # save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

        if (i>5):
            break
