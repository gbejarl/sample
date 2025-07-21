import obspy.signal.filter
import os
import sys
import obspy
import sklearn
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy import signal
from obspy import UTCDateTime
from time import gmtime, strftime
from scipy.ndimage import binary_opening, binary_closing
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib as mpl

# %% samples_from_stream


def samples_from_stream(processed_stream, features, window_length, window_overlap):
    """
    Extracts features from a processed seismic stream in a sliding window manner.
    Parameters:
    processed_stream (obspy.Stream): The processed seismic stream containing the trace data.
    features (list of int): List of indices corresponding to the features to be extracted.
    window_length (int): Length of the sliding window in minutes.
    window_overlap (float): Fraction of overlap between consecutive windows (0 <= window_overlap < 1).
    Returns:
    pandas.DataFrame: A DataFrame containing the extracted features and their corresponding timestamps.
    Features:
    00_Envelope_Unfiltered: Mean envelope of the unfiltered signal.
    01_Envelope_5Hz: Mean envelope of the signal filtered with a highpass filter at 5Hz.
    02_Envelope_5_10Hz: Mean envelope of the signal filtered with a bandpass filter between 5Hz and 10Hz.
    03_Envelope_5_20Hz: Mean envelope of the signal filtered with a bandpass filter between 5Hz and 20Hz.
    04_Envelope_10Hz: Mean envelope of the signal filtered with a highpass filter at 10Hz.
    05_Freq_Max_Unfiltered: Maximum frequency of the unfiltered signal.
    06_Freq_25th: 25th percentile frequency of the unfiltered signal.
    07_Freq_50th: 50th percentile frequency of the unfiltered signal.
    08_Freq_75th: 75th percentile frequency of the unfiltered signal.
    09_Kurtosis_Signal_Unfiltered: Kurtosis of the unfiltered signal.
    10_Kurtosis_Signal_5Hz: Kurtosis of the signal filtered with a highpass filter at 5Hz.
    11_Kurtosis_Signal_5_10Hz: Kurtosis of the signal filtered with a bandpass filter between 5Hz and 10Hz.
    12_Kurtosis_Signal_5_20Hz: Kurtosis of the signal filtered with a bandpass filter between 5Hz and 20Hz.
    13_Kurtosis_Signal_10Hz: Kurtosis of the signal filtered with a highpass filter at 10Hz.
    14_Kurtosis_Envelope_Unfiltered: Kurtosis of the envelope of the unfiltered signal.
    15_Kurtosis_Envelope_5Hz: Kurtosis of the envelope of the signal filtered with a highpass filter at 5Hz.
    16_Kurtosis_Envelope_5_10Hz: Kurtosis of the envelope of the signal filtered with a bandpass filter between 5Hz and 10Hz.
    17_Kurtosis_Envelope_5_20Hz: Kurtosis of the envelope of the signal filtered with a bandpass filter between 5Hz and 20Hz.
    18_Kurtosis_Envelope_10Hz: Kurtosis of the envelope of the signal filtered with a highpass filter at 10Hz.
    19_Kurtosis_Frequency_Unfiltered: Kurtosis of the frequency domain of the unfiltered signal.
    20_Kurtosis_Frequency_5Hz: Kurtosis of the frequency domain of the signal filtered with a highpass filter at 5Hz.
    21_Kurtosis_Frequency_5_10Hz: Kurtosis of the frequency domain of the signal filtered with a bandpass filter between 5Hz and 10Hz.
    22_Kurtosis_Frequency_5_20Hz: Kurtosis of the frequency domain of the signal filtered with a bandpass filter between 5Hz and 20Hz.
    23_Kurtosis_Frequency_10Hz: Kurtosis of the frequency domain of the signal filtered with a highpass filter at 10Hz.
    24_Skewness_Signal_Unfiltered: Skewness of the unfiltered signal.
    25_Skewness_Signal_5Hz: Skewness of the signal filtered with a highpass filter at 5Hz.
    26_Skewness_Signal_5_10Hz: Skewness of the signal filtered with a bandpass filter between 5Hz and 10Hz.
    27_Skewness_Signal_5_20Hz: Skewness of the signal filtered with a bandpass filter between 5Hz and 20Hz.
    28_Skewness_Signal_10Hz: Skewness of the signal filtered with a highpass filter at 10Hz.
    29_Skewness_Env_Unfiltered: Skewness of the envelope of the unfiltered signal.
    30_Skewness_Env_5Hz: Skewness of the envelope of the signal filtered with a highpass filter at 5Hz.
    31_Skewness_Env_5_10Hz: Skewness of the envelope of the signal filtered with a bandpass filter between 5Hz and 10Hz.
    32_Skewness_Env_5_20Hz: Skewness of the envelope of the signal filtered with a bandpass filter between 5Hz and 20Hz.
    33_Skewness_Env_10Hz: Skewness of the envelope of the signal filtered with a highpass filter at 10Hz.
    34_Skewness_Frequency_Unfiltered: Skewness of the frequency domain of the unfiltered signal.
    35_Skewness_Frequency_5Hz: Skewness of the frequency domain of the signal filtered with a highpass filter at 5Hz.
    36_Skewness_Frequency_5_10Hz: Skewness of the frequency domain of the signal filtered with a bandpass filter between 5Hz and 10Hz.
    37_Skewness_Frequency_5_20Hz: Skewness of the frequency domain of the signal filtered with a bandpass filter between 5Hz and 20Hz.
    38_Skewness_Frequency_10Hz: Skewness of the frequency domain of the signal filtered with a highpass filter at 10Hz.
    39_Spectral_Entropy_Unfiltered: Spectral entropy of the unfiltered signal.
    40_Spectral_Entropy_5Hz: Spectral entropy of the signal filtered with a highpass filter at 5Hz.
    41_Spectral_Entropy_5_10Hz: Spectral entropy of the signal filtered with a bandpass filter between 5Hz and 10Hz.
    42_Spectral_Entropy_5_20Hz: Spectral entropy of the signal filtered with a bandpass filter between 5Hz and 20Hz.
    43_Spectral_Entropy_10Hz: Spectral entropy of the signal filtered with a highpass filter at 10Hz.
    44_Ratio_Unfiltered_5Hz_10Hz: Ratio of the mean power spectral density below 5Hz to the mean power spectral density between 5Hz and 10Hz.
    """

    # Extract trace from stream and add to data arrays
    tr = processed_stream[0]
    tr_x = tr.data
    tr_t = tr.times(type='timestamp')

    # Define window length and overlap
    ww = (1/(tr.stats.delta))*60*window_length  # window width in samples
    ll = ww*(1-window_overlap)  # step length in samples

    feature_labels = ['00_Envelope_Unfiltered',
                      '01_Envelope_5Hz',
                      '02_Envelope_5_10Hz',
                      '03_Envelope_5_20Hz',
                      '04_Envelope_10Hz',
                      '05_Freq_Max_Unfiltered',
                      '06_Freq_25th',
                      '07_Freq_50th',
                      '08_Freq_75th',
                      '09_Kurtosis_Signal_Unfiltered',
                      '10_Kurtosis_Signal_5Hz',
                      '11_Kurtosis_Signal_5_10Hz',
                      '12_Kurtosis_Signal_5_20Hz',
                      '13_Kurtosis_Signal_10Hz',
                      '14_Kurtosis_Envelope_Unfiltered',
                      '15_Kurtosis_Envelope_5Hz',
                      '16_Kurtosis_Envelope_5_10Hz',
                      '17_Kurtosis_Envelope_5_20Hz',
                      '18_Kurtosis_Envelope_10Hz',
                      '19_Kurtosis_Frequency_Unfiltered',
                      '20_Kurtosis_Frequency_5Hz',
                      '21_Kurtosis_Frequency_5_10Hz',
                      '22_Kurtosis_Frequency_5_20Hz',
                      '23_Kurtosis_Frequency_10Hz',
                      '24_Skewness_Signal_Unfiltered',
                      '25_Skewness_Signal_5Hz',
                      '26_Skewness_Signal_5_10Hz',
                      '27_Skewness_Signal_5_20Hz',
                      '28_Skewness_Signal_10Hz',
                      '29_Skewness_Env_Unfiltered',
                      '30_Skewness_Env_5Hz',
                      '31_Skewness_Env_5_10Hz',
                      '32_Skewness_Env_5_20Hz',
                      '33_Skewness_Env_10Hz',
                      '34_Skewness_Frequency_Unfiltered',
                      '35_Skewness_Frequency_5Hz',
                      '36_Skewness_Frequency_5_10Hz',
                      '37_Skewness_Frequency_5_20Hz',
                      '38_Skewness_Frequency_10Hz',
                      '39_Spectral_Entropy_Unfiltered',
                      '40_Spectral_Entropy_5Hz',
                      '41_Spectral_Entropy_5_10Hz',
                      '42_Spectral_Entropy_5_20Hz',
                      '43_Spectral_Entropy_10Hz',
                      '44_Ratio_Unfiltered_5Hz_10Hz']

    # Define empty dictionary to store features
    selected_features = [feature_labels[i] for i in features]
    feature_dict = {i: [] for i in selected_features}

    # Define paths for computations
    group_time_unfiltered = True in (
        i in [0, 5, 9, 14, 19, 24, 29, 34, 39, 44] for i in features)
    group_time_5Hz = True in (
        i in [1, 10, 15, 20, 25, 30, 35, 40] for i in features)
    group_time_5_10Hz = True in (
        i in [2, 11, 16, 21, 26, 31, 36, 41] for i in features)
    group_time_5_20Hz = True in (
        i in [3, 12, 17, 22, 27, 32, 37, 42] for i in features)
    group_time_10Hz = True in (i in [4, 13, 18, 23, 28, 33, 38, 43]
                               for i in features)

    group_freq_unfiltered = True in (i in [5, 6, 7, 8, 44] for i in features)
    group_freq_5Hz = True in (i in [20, 25, 30, 35] for i in features)
    group_freq_5_10Hz = True in (i in [21, 26, 31, 36] for i in features)
    group_freq_5_20Hz = True in (i in [22, 27, 32, 37] for i in features)
    group_freq_10Hz = True in (i in [23, 28, 33, 38] for i in features)

    # Define filtered traces for time-domain features
    if group_time_unfiltered or group_freq_unfiltered:
        tr_unfiltered = tr.copy()
    if group_time_5Hz or group_freq_5Hz:
        tr_5Hz = tr.copy()
        tr_5Hz.filter('highpass', freq=5)
    if group_time_5_10Hz or group_freq_5_10Hz:
        tr_5_10Hz = tr.copy()
        tr_5_10Hz.filter('bandpass', freqmin=5, freqmax=10)
    if group_time_5_20Hz or group_freq_5_20Hz:
        tr_5_20Hz = tr.copy()
        tr_5_20Hz.filter('bandpass', freqmin=5, freqmax=20)
    if group_time_10Hz or group_freq_10Hz:
        tr_10Hz = tr.copy()
        tr_10Hz.filter('highpass', freq=10)

    # Define empty array to store timestamp
    timestamp = np.array([])

    # Iterate over every window in the trace
    for i in np.arange(0, (len(tr_x))-ww, ll):
        timestamp = np.append(timestamp, tr_t[int(i+(ww/2))])

        # Compute PSD for respective frequency bands (if prompted)
        if group_freq_unfiltered:
            segleng = 20*(1/tr_unfiltered.stats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_unfiltered, pp_unfiltered = signal.welch(tr_unfiltered.data[int(i):int(i+ww)],
                                                        fs=(1/tr_unfiltered.stats.delta),
                                                        window='hann',
                                                        nperseg=nperseg,
                                                        noverlap=nperseg/2,
                                                        nfft=nfft)
            csd_unfiltered = np.cumsum(pp_unfiltered)
            csd_unfiltered = csd_unfiltered-np.min(csd_unfiltered[1:])
            csd_unfiltered = csd_unfiltered/csd_unfiltered.max()
        if group_freq_5Hz:
            segleng = 20*(1/tr_5Hz.stats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_5Hz, pp_5Hz = signal.welch(tr_5Hz.data[int(i):int(i+ww)],
                                          fs=(1/tr_5Hz.stats.delta),
                                          window='hann',
                                          nperseg=nperseg,
                                          noverlap=nperseg/2,
                                          nfft=nfft)
            csd_5Hz = np.cumsum(pp_5Hz)
            csd_5Hz = csd_5Hz-np.min(csd_5Hz[1:])
            csd_5Hz = csd_5Hz/csd_5Hz.max()
        if group_freq_5_10Hz:
            segleng = 20*(1/tr_5_10Hz.stats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_5_10Hz, pp_5_10Hz = signal.welch(tr_5_10Hz.data[int(i):int(i+ww)],
                                                fs=(1/tr_5_10Hz.stats.delta),
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=nperseg/2,
                                                nfft=nfft)
            csd_5_10Hz = np.cumsum(pp_5_10Hz)
            csd_5_10Hz = csd_5_10Hz-np.min(csd_5_10Hz[1:])
            csd_5_10Hz = csd_5_10Hz/csd_5_10Hz.max()
        if group_freq_5_20Hz:
            segleng = 20*(1/tr_5_20Hz.stats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_5_20Hz, pp_5_20Hz = signal.welch(tr_5_20Hz.data[int(i):int(i+ww)],
                                                fs=(1/tr_5_20Hz.stats.delta),
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=nperseg/2,
                                                nfft=nfft)
            csd_5_20Hz = np.cumsum(pp_5_20Hz)
            csd_5_20Hz = csd_5_20Hz-np.min(csd_5_20Hz[1:])
            csd_5_20Hz = csd_5_20Hz/csd_5_20Hz.max()
        if group_freq_10Hz:
            segleng = 20*(1/tr_10Hz.stats.delta)
            nperseg = 2**np.ceil(np.log2(segleng))
            nfft = 4*nperseg
            ff_10Hz, pp_10Hz = signal.welch(tr_10Hz.data[int(i):int(i+ww)],
                                            fs=(1/tr_10Hz.stats.delta),
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=nperseg/2,
                                            nfft=nfft)
            csd_10Hz = np.cumsum(pp_10Hz)
            csd_10Hz = csd_10Hz-np.min(csd_10Hz[1:])
            csd_10Hz = csd_10Hz/csd_10Hz.max()

        # Compute features (if prompted)
        if 0 in features:
            feature_dict['00_Envelope_Unfiltered'].append(np.mean(
                obspy.signal.filter.envelope(tr_unfiltered.data[int(i):int(i+ww)])))
        if 1 in features:
            feature_dict['01_Envelope_5Hz'].append(np.mean(
                obspy.signal.filter.envelope(tr_5Hz.data[int(i):int(i+ww)])))
        if 2 in features:
            feature_dict['02_Envelope_5_10Hz'].append(np.mean(
                obspy.signal.filter.envelope(tr_5_10Hz.data[int(i):int(i+ww)])))
        if 3 in features:
            feature_dict['03_Envelope_5_20Hz'].append(np.mean(
                obspy.signal.filter.envelope(tr_5_20Hz.data[int(i):int(i+ww)])))
        if 4 in features:
            feature_dict['04_Envelope_10Hz'].append(np.mean(
                obspy.signal.filter.envelope(tr_10Hz.data[int(i):int(i+ww)])))
        if 5 in features:
            feature_dict['05_Freq_Max_Unfiltered'].append(
                ff_unfiltered[np.argmax(pp_unfiltered)])
        if 6 in features:
            feature_dict['06_Freq_25th'].append(
                ff_unfiltered[np.where(csd_unfiltered <= 0.25)[0][-1]])
        if 7 in features:
            feature_dict['07_Freq_50th'].append(
                ff_unfiltered[np.where(csd_unfiltered <= 0.50)[0][-1]])
        if 8 in features:
            feature_dict['08_Freq_75th'].append(
                ff_unfiltered[np.where(csd_unfiltered <= 0.75)[0][-1]])
        if 9 in features:
            feature_dict['09_Kurtosis_Signal_Unfiltered'].append(
                stats.kurtosis(tr_unfiltered.data[int(i):int(i+ww)]))
        if 10 in features:
            feature_dict['10_Kurtosis_Signal_5Hz'].append(
                stats.kurtosis(tr_5Hz.data[int(i):int(i+ww)]))
        if 11 in features:
            feature_dict['11_Kurtosis_Signal_5_10Hz'].append(
                stats.kurtosis(tr_5_10Hz.data[int(i):int(i+ww)]))
        if 12 in features:
            feature_dict['12_Kurtosis_Signal_5_20Hz'].append(
                stats.kurtosis(tr_5_20Hz.data[int(i):int(i+ww)]))
        if 13 in features:
            feature_dict['13_Kurtosis_Signal_10Hz'].append(
                stats.kurtosis(tr_10Hz.data[int(i):int(i+ww)]))
        if 14 in features:
            feature_dict['14_Kurtosis_Envelope_Unfiltered'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_unfiltered.data[int(i):int(i+ww)])))
        if 15 in features:
            feature_dict['15_Kurtosis_Envelope_5Hz'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_5Hz.data[int(i):int(i+ww)])))
        if 16 in features:
            feature_dict['16_Kurtosis_Envelope_5_10Hz'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_5_10Hz.data[int(i):int(i+ww)])))
        if 17 in features:
            feature_dict['17_Kurtosis_Envelope_5_20Hz'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_5_20Hz.data[int(i):int(i+ww)])))
        if 18 in features:
            feature_dict['18_Kurtosis_Envelope_10Hz'].append(
                stats.kurtosis(obspy.signal.filter.envelope(
                    tr_10Hz.data[int(i):int(i+ww)])))
        if 19 in features:
            feature_dict['19_Kurtosis_Frequency_Unfiltered'].append(
                stats.kurtosis(pp_unfiltered))
        if 20 in features:
            feature_dict['20_Kurtosis_Frequency_5Hz'].append(
                stats.kurtosis(pp_5Hz))
        if 21 in features:
            feature_dict['21_Kurtosis_Frequency_5_10Hz'].append(
                stats.kurtosis(pp_5_10Hz))
        if 22 in features:
            feature_dict['22_Kurtosis_Frequency_5_20Hz'].append(
                stats.kurtosis(pp_5_20Hz))
        if 23 in features:
            feature_dict['23_Kurtosis_Frequency_10Hz'].append(
                stats.kurtosis(pp_10Hz))
        if 24 in features:
            feature_dict['24_Skewness_Signal_Unfiltered'].append(
                stats.skew(tr_unfiltered.data[int(i):int(i+ww)]))
        if 25 in features:
            feature_dict['25_Skewness_Signal_5Hz'].append(
                stats.skew(tr_5Hz.data[int(i):int(i+ww)]))
        if 26 in features:
            feature_dict['26_Skewness_Signal_5_10Hz'].append(
                stats.skew(tr_5_10Hz.data[int(i):int(i+ww)]))
        if 27 in features:
            feature_dict['27_Skewness_Signal_5_20Hz'].append(
                stats.skew(tr_5_20Hz.data[int(i):int(i+ww)]))
        if 28 in features:
            feature_dict['28_Skewness_Signal_10Hz'].append(
                stats.skew(tr_10Hz.data[int(i):int(i+ww)]))
        if 29 in features:
            feature_dict['29_Skewness_Env_Unfiltered'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_unfiltered.data[int(i):int(i+ww)])))
        if 30 in features:
            feature_dict['30_Skewness_Env_5Hz'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_5Hz.data[int(i):int(i+ww)])))
        if 31 in features:
            feature_dict['31_Skewness_Env_5_10Hz'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_5_10Hz.data[int(i):int(i+ww)])))
        if 32 in features:
            feature_dict['32_Skewness_Env_5_20Hz'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_5_20Hz.data[int(i):int(i+ww)])))
        if 33 in features:
            feature_dict['33_Skewness_Env_10Hz'].append(
                stats.skew(obspy.signal.filter.envelope(
                    tr_10Hz.data[int(i):int(i+ww)])))
        if 34 in features:
            feature_dict['34_Skewness_Frequency_Unfiltered'].append(
                stats.skew(pp_unfiltered))
        if 35 in features:
            feature_dict['35_Skewness_Frequency_5Hz'].append(
                stats.skew(pp_5Hz))
        if 36 in features:
            feature_dict['36_Skewness_Frequency_5_10Hz'].append(
                stats.skew(pp_5_10Hz))
        if 37 in features:
            feature_dict['37_Skewness_Frequency_5_20Hz'].append(
                stats.skew(pp_5_20Hz))
        if 38 in features:
            feature_dict['38_Skewness_Frequency_10Hz'].append(
                stats.skew(pp_10Hz))
        if 39 in features:
            normalized_psd = pp_unfiltered/np.sum(pp_unfiltered)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['39_Spectral_Entropy_Unfiltered'].append(
                spectral_entropy)
        if 40 in features:
            normalized_psd = pp_5Hz/np.sum(pp_5Hz)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['40_Spectral_Entropy_5Hz'].append(spectral_entropy)
        if 41 in features:
            normalized_psd = pp_5_10Hz/np.sum(pp_5_10Hz)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['41_Spectral_Entropy_5_10Hz'].append(spectral_entropy)
        if 42 in features:
            normalized_psd = pp_5_20Hz/np.sum(pp_5_20Hz)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['42_Spectral_Entropy_5_20Hz'].append(spectral_entropy)
        if 43 in features:
            normalized_psd = pp_10Hz/np.sum(pp_10Hz)
            spectral_entropy = - \
                np.sum(normalized_psd * np.log2(normalized_psd))
            feature_dict['43_Spectral_Entropy_10Hz'].append(spectral_entropy)
        if 44 in features:
            feature_dict['44_Ratio_Unfiltered_5Hz_10Hz'].append(
                np.mean(pp_unfiltered[np.where(ff_unfiltered < 5)]) /
                np.mean(pp_unfiltered[np.where((ff_unfiltered >= 5) & (ff_unfiltered <= 10))]))

    feature_dict.update(
        {'Times': timestamp})

    df = pd.DataFrame.from_dict(feature_dict)

    return df

# %% clean_detections


def clean_detections(df, min_detection_length=15, max_gap_length=15):
    """
    Cleans the detection predictions in a DataFrame by applying binary opening, binary closing, 
    and enforcing a minimum detection length.
    Parameters:
    data_frame (pd.DataFrame): The input DataFrame containing 'Prediction' and 'Times' columns.
    min_detection_length (int, optional): The minimum length of a detection in minutes. Defaults to 15.
    max_gap_length (int, optional): The maximum length of a gap in minutes that can be filled. Defaults to 15.
    Returns:
    pd.DataFrame: The DataFrame with an additional 'Detection' column containing the cleaned detection predictions.
    """

    # Convert predictions to numpy array
    predictions = np.array(df['Prediction'])

    # Calculate time step
    tst = df['Times'].iloc[1]-df['Times'].iloc[0]

    # Calculate minimum detection and maximum gap length in points
    min_detection_points = (min_detection_length)*60/tst
    max_gap_points = (max_gap_length)*60/tst

    # Step 1: Remove isolated ones (binary opening)
    cleaned_array = binary_opening(
        predictions, structure=np.ones(int(max_gap_points))).astype(int)
    cleaned_array2 = cleaned_array

    # # Step 2: Remove isolated zeros (binary closing)
    cleaned_array = binary_closing(
        cleaned_array, structure=np.ones(int(max_gap_points))).astype(int)

    # # Step 3: Enforce minimum detection length
    labeled_regions = np.split(cleaned_array, np.where(
        np.diff(cleaned_array) != 0)[0] + 1)
    if (sum(labeled_regions[-1]) <= len(labeled_regions[-1])/2) & (sum(cleaned_array2[-len(labeled_regions[-1]):]) >= len(labeled_regions[-1])/2):
        labeled_regions[-1] = np.ones(len(labeled_regions[-1]))
    cleaned_array = np.concatenate([block if len(
        block) >= int(min_detection_points) or block[0] == 0 or i == len(labeled_regions) - 1 else np.zeros(len(block)) for i, block in enumerate(labeled_regions)])

    df['Detection'] = cleaned_array

    return df

# %% retrieve_dates


def retrieve_dates(df, target='Detection'):
    """
    Retrieve start and end times of detection events from a DataFrame.
    This function identifies the indices in the DataFrame where the target column
    has a value of 1, indicating a detection event. It then calculates the start
    and end times of these events and returns them as lists of UTCDateTime objects.
    Additionally, it determines if the detection events start at the beginning or
    end at the end of the DataFrame.
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    target (str): The column name in the DataFrame to check for detection events. Default is 'Detection'.
    Returns:
    tuple: A tuple containing:
        - starttimes (list of UTCDateTime): List of start times of detection events.
        - endtimes (list of UTCDateTime): List of end times of detection events.
        - open_start (bool): True if the first detection event starts at the beginning of the DataFrame, False otherwise.
        - open_end (bool): True if the last detection event ends at the end of the DataFrame, False otherwise.
    """

    # Convert DataFrame to numpy array
    det = np.array(df.index[df[target] == 1].tolist())

    # Check if there are any detections
    if np.any(det):

        # Find start and end times of detection events
        det1 = det[:-1]
        det2 = det[1:]
        det3 = det2-det1

        # Find indices where the difference is greater than 1
        det_i1 = (np.where(abs(det3) > 1)[0])
        det_i2 = [det[i] for i in det_i1]
        det_i3 = [det[i+1] for i in det_i1]

        # Extract start and end times of detection events
        if np.any(det_i2):
            det_0 = det[0]
            det_1 = det_i2
            det_0 = np.append(det_0, det_i3)
            det_1 = np.append(det_1, det[-1])
            starttimes = [UTCDateTime(df['Times'].iloc[i])
                          for i in det_0]
            endtimes = [UTCDateTime(df['Times'].iloc[i])
                        for i in det_1]
        else:
            det_0 = det[0]
            det_1 = det[-1]
            starttimes = [UTCDateTime(df['Times'].iloc[det_0])]
            endtimes = [UTCDateTime(df['Times'].iloc[det_1])]

        # Check if detection events start at the beginning or end at the end
        if det[0] == 0:
            open_start = True
        else:
            open_start = False

        # Check if detection events end at the end
        if det[-1] == len(df[target]):
            open_end = True
        else:
            open_end = False

    #
    else:
        starttimes = None
        endtimes = None
        open_start = False
        open_end = False

    return starttimes, endtimes, open_start, open_end

# %% plot_detections


def plot_detections(data_frame, st, target='Detection', vmin=None, vmax=None, save=False, save_path='', show=True, vertical_lim=None, sensitivity=None):

    dt = retrieve_dates(data_frame, target=target)

    t0 = [x.matplotlib_date for x in dt[0]]
    t1 = [x.matplotlib_date for x in dt[1]]

    st_x = st.data
    if sensitivity:
        st_y = st_x/sensitivity
    else:
        st_y = st_x
    st_t = st.times(type='matplotlib')

    isamp = int(1/(st.stats.delta))
    inseg = isamp*10
    infft = 2048

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 7))
    ax[0].plot(st_t, st_y, lw=0.5, c='k')
    ax[0].set_xlim(min(st_t), max(st_t))
    if vertical_lim:
        ax[0].set_ylim(-vertical_lim, vertical_lim)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[0].xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax[0].xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
    ax[0].set_title(st.stats.station)
    for i in range(len(t0)):
        ax[0].axvspan(t0[i], t1[i], color='red', alpha=0.25)
    xspf, xspt, xsps = signal.spectrogram(
        st_x, fs=isamp, nperseg=inseg, nfft=infft, detrend=False)
    xspd = 20*np.log10(abs(xsps))
    ax[1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(24)))
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[1].xaxis.set_minor_locator(
        mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    im = ax[1].imshow(xspd, origin='lower', interpolation='nearest', aspect='auto',
                      extent=[st_t[0], st_t[-1], xspf[0], xspf[-1]],
                      cmap='jet', vmin=vmin, vmax=vmax)
    for i in range(len(t0)):
        ax[1].axvspan(t0[i], t1[i], color='black', alpha=0.5)
    ax[1].set_ylabel('Hz')
    fig.suptitle('Detection between\n'+st.stats.starttime.strftime('%Y-%m-%d''T''%H:%M:%S') +
                 ' to '+st.stats.endtime.strftime('%Y-%m-%d''T''%H:%M:%S'))
    plt.tight_layout

    if show:
        plt.show()

    if save:
        if save_path == '':
            save_path = os.getcwd()
        plt.savefig(save_path+'/'+st.stats.station+'_' +
                    st.stats.starttime.strftime('%Y%m%d%H%M%S')+'.png')
        plt.close()

# %%


def preprocess_stream(stream,
                      starttime=None,
                      endtime=None,
                      decimate=None,
                      resample=None,
                      detrend=True,
                      detrend_type='linear',
                      taper=True,
                      taper_fraction=0.001,
                      freqmin=None,
                      freqmax=None,
                      merge=True,
                      merge_fill_value='interpolate',
                      merge_method=0,
                      merge_interpolation_samples=0,
                      sensitivity=None,
                      select_channel=None):
    """
    Preprocesses a stream of seismic data by applying various filters and corrections.

    Args:
        stream (obspy.core.stream.Stream): The stream of seismic data to preprocess.
        starttime (obspy.UTCDateTime, optional): The start time to trim the stream to. Defaults to None.
        endtime (obspy.UTCDateTime, optional): The end time to trim the stream to. Defaults to None.
        decimate (int, optional): The factor to decimate the stream by. Defaults to None.
        resample (float, optional): The sampling rate to resample the stream to. Defaults to None.
        detrend (bool, optional): Whether or not to detrend the stream. Defaults to True.
        detrend_type (str, optional): The type of detrending to apply. Defaults to 'linear'.
        taper (bool, optional): Whether or not to taper the stream. Defaults to True.
        taper_fraction (float, optional): The fraction of the stream to taper. Defaults to 0.05.
        freqmin (float, optional): The minimum frequency to filter the stream to. Defaults to None.
        freqmax (float, optional): The maximum frequency to filter the stream to. Defaults to None.
        merge (bool, optional): Whether or not to merge the stream. Defaults to True.
        merge_fill_value (str, optional): The fill value to use when merging the stream. Defaults to 'interpolate'.
        merge_method (int, optional): The method to use when merging the stream. Defaults to 0.
        merge_interpolation_samples (int, optional): The number of samples to interpolate when merging the stream. Defaults to 0.
        calibration (float, optional): The calibration factor to apply to the stream. Defaults to None.

    Returns:
        obspy.core.stream.Stream: The preprocessed stream of seismic data.
    """

    if select_channel:
        stream = stream.select(channel=select_channel)

    if merge:
        stream.merge(fill_value=merge_fill_value,
                     method=merge_method,
                     interpolation_samples=merge_interpolation_samples)

    if starttime or endtime:
        stream.trim(starttime=starttime, endtime=endtime)

    if decimate:
        stream.decimate(factor=decimate)

    if resample:
        stream.resample(sampling_rate=resample)

    if detrend:
        stream.detrend(type=detrend_type)

    if taper:
        stream.taper(max_percentage=taper_fraction)

    if freqmin:
        stream.filter('highpass', freq=freqmin)

    if freqmax:
        stream.filter('lowpass', freq=freqmax)

    if sensitivity:
        for traces in stream:
            traces.data /= sensitivity

    return stream
