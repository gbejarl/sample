##############################
# %% 1 Import packages
# 1 Import packages
##############################

import obspy.signal.filter
import os
import sys
import json
import obspy
import sklearn
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy import signal
from obspy import UTCDateTime
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib as mpl
import joblib

from laharml import samples_from_stream, preprocess_stream, clean_detections, retrieve_dates


def run_detection(
    stream_filepath,
    network,
    station,
    location,
    channel,
    sensitivity,
    code_label,
    minimum_frequency,
    minimum_duration_to_alert
):

    if not stream_filepath:
        raise ValueError("stream_filepath is required and cannot be empty.")
    if not code_label:
        raise ValueError("code_label is required and cannot be empty.")
    if minimum_duration_to_alert is None:
        raise ValueError(
            "minimum_duration_to_alert is required and cannot be None.")

    ##############################
    # 2 Initialize parameters
    ##############################

    stream = obspy.read(stream_filepath)

    # Select the vertical (Z) component from the stream
    stream = stream.select(component='Z')
    if len(stream) == 0:
        raise ValueError("No vertical (Z) component found in the stream.")

    if not network:
        network = stream[0].stats.network
    if not station:
        station = stream[0].stats.station
    if not location:
        location = stream[0].stats.location
    if not channel:
        channel = stream[0].stats.channel

    # Define station code
    station_code = f'{network}.{station}.{location}.{channel}'

    # Load sensitivity from JSON file
    with open('sensitivities.json', 'r') as f:
        sensitivities = json.load(f)
    if not sensitivity:
        sensitivity = sensitivities.get(station_code)

    # Model codes
    model_code = f'XX.XXXX.XX.XXX_{code_label}'

    # Extract frequency, window length, and overlap from model_code
    frequency = int(model_code.split('_')[1].split('Hz')[0])
    window_length = int(model_code.split(f'{frequency}Hz')[1].split('min')[0])
    window_overlap = int(model_code.split(f'{window_length:02d}min')[1]) / 100

    resample_frequency = 50

    # Set model and scaler paths (including kranges and kscores arrays)
    model_path = os.path.join('Models',
                              f'{model_code}_smodel.pkl')
    scaler_path = os.path.join('Models',
                               f'{model_code}_sscaler.pkl')
    krange_path = os.path.join('Models',
                               f'{model_code}_krange.npy')
    kscore_path = os.path.join('Models',
                               f'{model_code}_kscore.npy')

    # Load model, scaler, krange, and kscore
    krange = np.load(krange_path)
    kscore = np.load(kscore_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Define features from training file
    features = model.feature_names_in_
    features = [int(i.split('_')[0]) for i in features]

    # Find the minimum in kscore and get the corresponding value from krange
    min_index = np.argmin(kscore)
    optimal_k = krange[min_index]

    while True:

        detections_all = pd.DataFrame(columns=['identifier',
                                               'start',
                                               'end',
                                               'status',
                                               'notified'])
        detections_all.set_index('identifier', inplace=True)

        # Set parameters for post-processing
        x1 = np.array([])  # Start time
        x2 = np.array([])  # End time
        x3 = np.array([])  # Average power 0-5Hz
        x4 = np.array([])  # Average power 5-10Hz2
        x5 = np.array([])  # Ratio of x3/x4

        starttime = stream[0].stats.starttime
        endtime = stream[-1].stats.endtime

        # Set instrument response
        sensitivity = sensitivities.get(station_code)

        # Preprocess stream
        event_stream = preprocess_stream(stream,
                                         starttime=starttime,
                                         endtime=endtime,
                                         resample=resample_frequency,
                                         freqmin=minimum_frequency,
                                         sensitivity=sensitivity)
        print(
            f"Started processing date {endtime.strftime('%Y-%m-%dT%H:%M:%S')}")

        # Extract features from stream (parametrize stream)
        unclassified_data_frame = samples_from_stream(event_stream,
                                                      features,
                                                      window_length,
                                                      window_overlap)

        # Scale data
        scaled_df = scaler.transform(
            unclassified_data_frame.drop(columns='Times'))

        # Turn scaled_df into pandas dataframe with respective column names
        scaled_df = pd.DataFrame(
            scaled_df, columns=unclassified_data_frame.drop(columns='Times').columns)

        # Generate inferences
        predictions = model.predict(scaled_df)

        # Add predictions to data frame
        unclassified_data_frame['Prediction'] = predictions

        # Clean detections
        cleaned_data_frame = clean_detections(unclassified_data_frame)

        # Retrieve dates
        lah_0, lah_1, lah_0l, lah_1l = retrieve_dates(cleaned_data_frame)

        # Count number of detections in iteration and add to arrays
        if lah_0 is not None:
            lah_count = len(lah_0)
            x1 = np.append(x1, lah_0)
            x2 = np.append(x2, lah_1)
        # If no detections, set count to 0 and move to next iteration
        else:
            lah_count = 0
            tot_count = len(x1)

        # Calculate frequency parameters when detections exist
        if lah_count:
            for i in range(-len(lah_0), 0):
                sts = event_stream.slice(x1[i], x2[i])
                st_data = sts[0].data
                ffx, ppx = signal.welch(st_data, fs=sts[0].stats.sampling_rate)
                avg_lo = np.mean(ppx[np.where(ffx < 5)])
                avg_hi = np.mean(ppx[np.where((ffx >= 5) & (ffx <= 10))])
                x3 = np.append(x3, avg_lo)
                x4 = np.append(x4, avg_hi)
                x5 = np.append(x5, avg_lo/avg_hi)
        else:
            xts = np.array([])

        # Update total number of detections over multiple iterations
        if x1.size > 0:
            tot_count = len(x1)
        else:
            tot_count = 0

        print('''\r\n
            Finished detections for the following dates:
            {date_1} to {date_2}.
            Detections found={number_1}
            Number of detections saved={number_2}
            '''.format(date_1=starttime.strftime('%Y-%m-%dT%H:%M:%S'),
                       date_2=endtime.strftime('%Y-%m-%dT%H:%M:%S'),
                       number_1=lah_count,
                       number_2=tot_count))

        # Consolidate detection subarrays into single array
        xts = np.stack(([i.strftime('%Y-%m-%dT%H:%M:%S') for i in x1],
                        [i.strftime('%Y-%m-%dT%H:%M:%S') for i in x2],
                        x3,
                        x4,
                        x5), axis=-1)

        # Automated post processing

        r1a = []  # Start time, step 1
        r1b = []  # End time, step 1
        r2a = []  # Start time, step 2
        r2b = []  # End time, step 2
        r3a = []  # Start time, step 3
        r3b = []  # End time, step 3

        if tot_count:

            # Step 1: Remove detections that are likely noise (use frequency ratios)

            for i in range(len(xts)):
                if (float(xts[i][2])/float(xts[i][3])) <= 0.75:
                    r1a.append(xts[i][0])
                    r1b.append(xts[i][1])

            # Step 2: Remove detection of less than minimum_lahar_duration minutes

            for i in range(len(r1a)):
                if (UTCDateTime(r1b[i])-UTCDateTime(r1a[i])) >= \
                        (60*minimum_duration_to_alert):
                    r2a.append(r1a[i])
                    r2b.append(r1b[i])

            r2a = np.array(r2a)
            r2b = np.array(r2b)

            # Step 3: Remove overlapping detections

            for i in range(len(r2a)):
                skipped = [*range(len(r2a))]
                skipped.remove(i)
                overlap = 0
                for j in skipped:
                    a1 = UTCDateTime(r2a[i])
                    a2 = UTCDateTime(r2b[i])
                    if (a1 <= UTCDateTime(r2b[j])) and (a2 >= UTCDateTime(r2a[j])):
                        overlap += 1
                        if ((UTCDateTime(r2b[j])-UTCDateTime(r2a[j])) <= (a2-a1)):
                            r3a.append(a1)
                            r3b.append(a2)
                        else:
                            r3a.append(UTCDateTime(r2a[j]))
                            r3b.append(UTCDateTime(r2b[j]))
                if overlap == 0:
                    r3a.append(UTCDateTime(r2a[i]))
                    r3b.append(UTCDateTime(r2b[i]))

            x1 = np.unique(
                np.array([i.strftime('%Y-%m-%dT%H:%M:%S') for i in r3a]))
            x2 = np.unique(
                np.array([i.strftime('%Y-%m-%dT%H:%M:%S') for i in r3b]))

        else:

            # Final list of detections (empty, no detections)

            x1 = np.array([])
            x2 = np.array([])

        detections_dict = {}

        if len(x1) >= 1:
            for a in range(len(x1)):
                if pd.to_datetime(x2[a]) >= pd.to_datetime(event_stream[0].times('timestamp'), unit='s')[-1] - pd.Timedelta(minutes=2):
                    detections_dict[f'{station_code.replace(".", "")+x1[a].replace("-", "").replace("T", "").replace(":", "")}'] = {
                        'station': station_code,
                        'start': x1[a],
                        'end': x2[a],
                        'status': 'active',
                        'notified': 'no',
                    }
                else:
                    detections_dict[f'{station_code.replace(".", "")+x1[a].replace("-", "").replace("T", "").replace(":", "")}'] = {
                        'station': station_code,
                        'start': x1[a],
                        'end': x2[a],
                        'status': 'inactive',
                        'notified': 'no',
                    }

        else:
            detections_dict = {}

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 7), dpi=200)
        ax[0].set_title(
            f'Waveform')
        ax[0].plot(pd.to_datetime(event_stream[0].times('timestamp'), unit='s'),
                   event_stream[0].data, 'k', lw=2)
        ax[0].set_xlim(pd.to_datetime(event_stream[0].times('timestamp'), unit='s')[0],
                       pd.to_datetime(event_stream[0].times('timestamp'), unit='s')[-1])
        ax[0].yaxis.set_major_formatter(
            mpl.ticker.ScalarFormatter(useMathText=True))
        ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax[0].set_ylabel('Amplitude in m/s')
        if 1 in np.unique(cleaned_data_frame['Prediction']):
            sns.scatterplot(x=pd.to_datetime(cleaned_data_frame['Times'], unit='s'), y=cleaned_data_frame.columns[0], data=cleaned_data_frame,
                            ax=ax[1], hue='Prediction', palette=['black', 'tab:red'], edgecolor='None', legend=False)
        else:
            sns.scatterplot(x=pd.to_datetime(cleaned_data_frame['Times'], unit='s'), y=cleaned_data_frame.columns[0], data=cleaned_data_frame,
                            ax=ax[1], hue='Prediction', palette=['black'], edgecolor='None', legend=False)
        ax[1].set_ylabel(f'{cleaned_data_frame.columns[0].replace("_", " ")}')
        ax[1].set_xlim(pd.to_datetime(event_stream[0].times('timestamp'), unit='s')[0],
                       pd.to_datetime(event_stream[0].times('timestamp'), unit='s')[-1])
        ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax[1].set_title(
            f'Raw LaharML detections')
        ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if 1 in np.unique(cleaned_data_frame['Detection']):
            sns.scatterplot(x=pd.to_datetime(cleaned_data_frame['Times'], unit='s'), y=cleaned_data_frame.columns[0], data=cleaned_data_frame,
                            ax=ax[2], hue='Detection', palette=['black', 'tab:red'], edgecolor='None', legend=False)
        else:
            sns.scatterplot(x=pd.to_datetime(cleaned_data_frame['Times'], unit='s'), y=cleaned_data_frame.columns[0], data=cleaned_data_frame,
                            ax=ax[2], hue='Detection', palette=['black'], edgecolor='None', legend=False)
        ax[2].set_xlabel('')
        ax[2].set_ylabel(f'{cleaned_data_frame.columns[0].replace("_", " ")}')
        ax[2].yaxis.set_major_formatter(
            mpl.ticker.ScalarFormatter(useMathText=True))
        ax[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax[2].set_xlim(pd.to_datetime(event_stream[0].times('timestamp'), unit='s')[0],
                       pd.to_datetime(event_stream[0].times('timestamp'), unit='s')[-1])
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax[2].xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax[2].xaxis.set_minor_locator(mdates.MinuteLocator([0, 30]))
        ax[2].set_title(f'Cleaned detections and extracted intervals')
        if x1.size > 0:
            for i in range(len(lah_0)):
                ax[2].axvspan(lah_0[i], lah_1[i], color='tab:red', alpha=0.5)
        plt.suptitle(
            f'LaharML at {station_code}\n{starttime.strftime("%Y-%m-%dT%H:%M:%S")} and {endtime.strftime("%Y-%m-%dT%H:%M:%S")}')
        plt.tight_layout()
        plt.show()

        # Convert detections_dict to dataframe
        if detections_dict:
            detections_df = pd.DataFrame.from_dict(
                detections_dict, orient='index')
            detections_df.index.name = 'identifier'

        if len(detections_all) == 0:
            detections_all = pd.concat([detections_all, detections_df])
        else:
            for i in range(len(detections_df)):
                det_min = detections_df['start'][i]
                det_max = detections_df['end'][i]
                det_id = detections_df.index[i]

                overlapping_detections = detections_all[
                    (detections_all['start'] <= det_max) & (
                        detections_all['end'] >= det_min)
                ]

                if overlapping_detections.empty:
                    detections_all = pd.concat([detections_all, detections_df])
                else:
                    for index, row in overlapping_detections.iterrows():
                        if row['start'] < det_min:
                            det_min = row['start']
                        if row['end'] > det_max:
                            det_max = row['end']
                        notified = row['notified']
                        detections_all.drop(index, inplace=True)
                    detections_all = pd.concat(detections_all, pd.DataFrame({
                        'start': [det_min],
                        'end': [det_max],
                        'status': ['active'],
                        'notified': notified
                    }, index=[det_id]))

        # Save detections to CSV
        detections_all.to_csv(f'detections_{station_code}.csv')

        detections_all

        break


# Guard for running as script with default/example parameters
if __name__ == "__main__":
    run_detection(
        stream_filepath=None,  # default example
        network=None,
        station=None,
        location=None,
        channel=None,
        sensitivity=None,
        code_label=None,
        minimum_frequency=None,
        minimum_duration_to_alert=None
    )
