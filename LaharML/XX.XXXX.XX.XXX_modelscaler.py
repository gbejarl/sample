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
from obspy.clients.fdsn import Client
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

from laharml import preprocess_stream, samples_from_stream

##############################
# %% 2 Initialize parameters
# 2 Initialize parameters
##############################

# Add directory to pull data from (either for files that are locally saved
# or for files that are pulled from the server).
# Not required if using FDSN client.
# If both a local directory and a FDSN client are used, the local directory
# will be used to pull data from.
data_directory = ''  # Path to local data directory or None

# FDSNWS Archive and data download parameters
# Not required if using local files
# If both a local directory and a FDSN client are used, the local directory
# will be used to pull data from.
service = ''  # FDSNWS Archive URL or None

# DEFINE FILEPATHS FOR EVENT AND NOISE TIMES
event_filepath = ''
noise_filepath = ''

# Define list of station codes (not all need to be used)
# Station codes are in the format XX.XXXX.XX.XXX
sensitivities = {'6Q.FEC1..HHZ': 2.9947e8,
                 '6Q.FEC2..HHZ': 2.9947e8,
                 '6Q.FEC4..HHZ': 2.9947e8,
                 '6Q.FGLR..HHZ': 1.25841e9,
                 'GI.FG12.00.BHZ': 4.81e8,
                 'GI.FG14.00.SHZ': 2.14079e8,
                 'GI.FG14.01.BHZ': 2.9947e8,
                 'GI.FG16.00.BHZ': 2.9947e8}

# Define station code to train a specific model
# Leave XXXX if you want to train a model for all stations
# defined in the sensitivities dictionary above
network = 'GI'
station = 'FG12'
location = '00'
channel = 'BHZ'

# Define training data characteristics
# Ideally keep everything the same but feel free to change the
# window length and evaluate performance. Best lahar model for Fuego
# have best performance at 1 minute window length and 25% overlap for
# general models.
# Resample data to this frequency (in Hz), if None, no resampling
resample_frequency = 50
scale_data = True  # If True, scale data
window_length = 1  # Moving window length in minutes
window_overlap = 0.25  # Fraction of overlap between windows
minimum_frequency = 0.1  # Minimum frequency to filter data

# Evaluates all features when building the model
features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44]

# Choose a maximum number of best features the random forest method
# will select. Keep in mind the final number of features might be less
# than this number due to the dropping of features with high correlation.
number_of_best_features = 6

##############################
# %% 3 Feature Extraction
# 3 Feature Extraction
##############################

feature_groups = ['A'] * 5 + ['B'] * 4 + ['C'] * 5 + ['D'] * 5 + \
    ['E'] * 5 + ['F'] * 5 + ['G'] * 5 + ['H'] * 5 + ['I'] * 5 + ['J']
feature_groups = [feature_groups[i] for i in features]

# Full channel code (for labels and file names)
channel_code = f'{network}.{station}.{location}.{channel}'

# Load event and noise dates
event_dates = pd.read_csv(event_filepath, sep=',', header=None)
noise_dates = pd.read_csv(noise_filepath, sep=',', header=None)

# Filter station
if f'{network}.{station}.{location}.{channel}' != 'XX.XXXX.XX.XXX':
    event_dates = event_dates[event_dates[0] == channel_code]
    noise_dates = noise_dates[noise_dates[0] == channel_code]

# Initialize total duration variable at zero
total_lahar_time = 0

# Define empty lists to store event and noise data
traintest = pd.DataFrame()

# Iterate over every row in the event_dates dataframe
for index, row in event_dates.iterrows():

    # Extract event start and end times and compute total record duration
    event_t0 = UTCDateTime(row[1])
    event_t1 = UTCDateTime(row[2])
    total_lahar_time += event_t1 - event_t0

    if data_directory:

        # Build filepaths to event data
        event_filepath1 = os.path.join(data_directory,
                                       row[0][3:7],
                                       f'{row[0]}.{event_t0.year}.{event_t0.julday:03d}.mseed')
        event_filepath2 = os.path.join(data_directory,
                                       row[0][3:7],
                                       f'{row[0]}.{event_t1.year}.{event_t1.julday:03d}.mseed')

        # Load event data
        stream = obspy.read(event_filepath1, starttime=event_t0, endtime=event_t1) +\
            obspy.read(event_filepath2, starttime=event_t0, endtime=event_t1)

    elif service:

        # Build client request
        client = Client(service)
        stream = client.get_waveforms(
            network, station, location, channel, event_t0, event_t1)

    else:

        # If no data directory or FDSN client is provided, exit the script
        sys.exit("No data directory or FDSN client provided. Exiting.")

    print(f"Processing lahar event {index+1} of {len(event_dates)}")

    # Resample data if propted and if sampling rate is greater than 50 Hz
    sampling_rate = stream[0].stats.sampling_rate
    if (sampling_rate > resample_frequency):
        new_sampling_rate = 50
    else:
        print(f"Resample frequency is less than sampling rate. No resampling.")
        new_sampling_rate = None

    # Retrieve sensitivity for station
    sensitivity = sensitivities.get(row[0])

    # Preprocess stream
    event_stream = preprocess_stream(stream,
                                     resample=new_sampling_rate,
                                     freqmin=minimum_frequency,
                                     sensitivity=sensitivity)

    # Extract features from stream (parametrize stream)
    extracted_df = samples_from_stream(event_stream,
                                       window_length,
                                       window_overlap)

    # Assign class label to extracted data
    extracted_df['Class'] = 1

    # Append extracted data to traintest dataframe
    traintest = pd.concat([traintest, extracted_df])

# Compute individual noise time
individual_noise_time = total_lahar_time/len(noise_dates)

# Iterate over every row in the noise_dates dataframe
for index, row in noise_dates.iterrows():

    # Extract event start and end times and compute total record duration
    noise_t0 = UTCDateTime(row[1])-(individual_noise_time/2)
    noise_t1 = UTCDateTime(row[1])+(individual_noise_time/2)

    # Build filepaths to noise data
    noise_filepath1 = os.path.join(data_directory,
                                   row[0][3:7],
                                   f'{row[0]}.{noise_t0.year}.{noise_t0.julday:03d}.mseed')
    noise_filepath2 = os.path.join(data_directory,
                                   row[0][3:7],
                                   f'{row[0]}.{noise_t1.year}.{noise_t1.julday:03d}.mseed')

    # Load event data
    stream = obspy.read(noise_filepath1, starttime=noise_t0, endtime=noise_t1) +\
        obspy.read(noise_filepath2, starttime=noise_t0, endtime=noise_t1)

    print(f"Processing noise event {index+1} of {len(noise_dates)}")

    # Resample data if propted and if sampling rate is greater than 50 Hz
    sampling_rate = stream[0].stats.sampling_rate
    if (sampling_rate > resample_frequency):
        new_sampling_rate = 50
    else:
        print(f"Resample frequency is less than sampling rate. No resampling.")
        new_sampling_rate = None

    # Retrieve sensitivity for station
    sensitivity = sensitivities.get(row[0])

    # Preprocess stream
    noise_stream = preprocess_stream(stream,
                                     resample=new_sampling_rate,
                                     freqmin=minimum_frequency,
                                     sensitivity=sensitivity)

    # Extract features from stream (parametrize stream)
    extracted_df = samples_from_stream(noise_stream,
                                       window_length,
                                       window_overlap)

    # Assign class label to extracted data
    extracted_df['Class'] = 0

    # Append extracted data to traintest dataframe
    traintest = pd.concat([traintest, extracted_df]).reset_index(drop=True)

# Model code (for labels and file names)
if new_sampling_rate:
    model_code = f'{str(int(new_sampling_rate))}Hz{window_length:02d}min{int(window_overlap*100):02d}'
else:
    original_sampling_rate = stream[0].stats.sampling_rate
    model_code = f'{str(int(original_sampling_rate))}Hz{window_length:02d}min{int(window_overlap*100):02d}'

# Check if folder exist, if it doesn't, create it
Path(f'{channel_code}_{model_code}').mkdir(parents=True, exist_ok=True)

##############################
# %% 4 Data Scaling
# 4 Data Scaling
##############################

# Fit and transform using a scaler
traintest_params = traintest.drop(['Times', 'Class'], axis=1)
scaler = PowerTransformer(method='yeo-johnson')
traintest_params = scaler.fit_transform(traintest_params)

# Create a dataframe with scaled data
scaled = pd.DataFrame(
    traintest_params, columns=traintest.columns[:-2]).reset_index(drop=True)
scaled['Class'] = traintest['Class']

##############################
# %% 5 Feature Importance
# 5 Feature Importance
##############################

X_train, X_test, y_train, y_test = train_test_split(
    scaled.drop(['Class'], axis=1), scaled['Class'], test_size=0.5, stratify=scaled['Class'])

# Train random forest classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Extract and rank feature importances
importances = rf.feature_importances_

# Build dictionary with feature names and their importances
feature_importance_dict = {feature: (importance, group) for feature, importance, group in zip(
    traintest.drop(['Times', 'Class'], axis=1).columns, importances, feature_groups)}

# Create a pandas dataframe with feature importances and groups
feature_importance_df = pd.DataFrame.from_dict(
    feature_importance_dict, orient='index', columns=['Importance', 'Group'])

# Sort the dataframe by importance
feature_importance_df = feature_importance_df.sort_values(
    by='Importance', ascending=False)

# Select the best six features by choosing from the top n of features of each group
bestfeatures = []
for group in feature_importance_df['Group'].unique():
    bestfeatures += feature_importance_df[feature_importance_df['Group']
                                          == group].index[:1].tolist()
bestfeatures = bestfeatures[:number_of_best_features]

# Plot the feature importances
plt.figure(figsize=(12, 8))
sns.pairplot(scaled[bestfeatures + ['Class']], hue='Class')
plt.suptitle(f'{channel_code}_{model_code}')
plt.savefig(
    f'{channel_code}_{model_code}/{channel_code}_{model_code}_pairplot.png', bbox_inches='tight')

# Check correlation of all columns in scaled[bestfeatures]
correlation_matrix = scaled[bestfeatures].corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
mask = np.zeros_like(correlation_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask,
            cbar_kws={"shrink": .8}, annot_kws={"size": 10, "color": "white"},
            linewidths=.5, linecolor='black')
# Highlight cells with correlation greater than 0.9
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='black', lw=0))
plt.suptitle(f'{channel_code}_{model_code}')
plt.savefig(
    f'{channel_code}_{model_code}/{channel_code}_{model_code}_correlation.png', bbox_inches='tight')
plt.show()

# Find pairs of attributes with correlation 0.9 or higher and remove the latter one
correlated_features = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            colname = correlation_matrix.columns[i]
            correlated_features.append(colname)

if correlated_features:
    bestfeatures = [
        feature for feature in bestfeatures if feature not in correlated_features]

# Sort list alphabetically
bestfeatures.sort()

##############################
# %% 6 General Model
# 6 General Model
##############################

# Define the new scaled dataframe with the best features
bestfeatures_params = traintest[bestfeatures]
scaler_function = PowerTransformer(method='yeo-johnson')
bestfeatures_scaled_params = scaler_function.fit_transform(
    bestfeatures_params)

# Create a dataframe with scaled data
bestfeatures_scaled = pd.DataFrame(
    bestfeatures_scaled_params, columns=bestfeatures).reset_index(drop=True)
bestfeatures_scaled['Class'] = traintest['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bestfeatures_scaled.drop(
    ['Class'], axis=1), bestfeatures_scaled['Class'], test_size=0.25, stratify=bestfeatures_scaled['Class'])
X_train = pd.DataFrame(
    X_train, columns=bestfeatures_scaled.columns[:-1], index=y_train.index)
X_test = pd.DataFrame(
    X_test, columns=bestfeatures_scaled.columns[:-1], index=y_test.index)
train = X_train.copy()
train['Class'] = y_train.values

krange = range(1, 200, 2)
kscore = []
kf = StratifiedKFold(n_splits=10, shuffle=True)

for k in krange:
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    scores = cross_val_score(knn, train.drop(
        ['Class'], axis=1), train['Class'], cv=kf)
    kscore.append(1-scores.mean())

minpos = kscore.index(min(kscore))
n = krange[minpos]

minpos = kscore.index(min(kscore))
n = krange[minpos]
neighbors = n

model = KNeighborsClassifier(n_neighbors=int(
    neighbors), weights='uniform')
model.fit(X_train, y_train)
pred = model.predict(X_test)
report = classification_report(y_test, pred)
conmat = confusion_matrix(y_test, pred)
print(report)
print(n)

fig, ax = plt.subplots(figsize=(6, 4))
plt.plot(krange, kscore)
plt.ylabel('Cross-Validated Error')
plt.suptitle(
    f'{channel_code}_{model_code}\nKNN Classifier with {neighbors} Neighbors')
# Add table with classification report
report_dict = classification_report(y_test, pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
table = plt.table(cellText=report_df.values.round(2),
                  colLabels=report_df.columns,
                  rowLabels=report_df.index,
                  cellLoc='center',
                  loc='bottom',
                  bbox=[0, -0.65, 1, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)
# Add table with confusion matrix
conmat_df = pd.DataFrame(
    conmat, index=['True 0', 'True 1'], columns=['Pred 0', 'Pred 1'])
table2 = plt.table(cellText=conmat_df.values,
                   colLabels=conmat_df.columns,
                   rowLabels=conmat_df.index,
                   cellLoc='center',
                   loc='bottom',
                   bbox=[0, -1.0, 1, 0.2])
table2.auto_set_font_size(False)
table2.set_fontsize(8)
table2.scale(1, 1.5)
plt.subplots_adjust(left=0.2, bottom=0.3)
fig.savefig(
    f'{channel_code}_{model_code}/{channel_code}_{model_code}_knn.png', bbox_inches='tight')
plt.show()

##############################
# %% 7 Serialize and Save Model
# 7 Serialize and Save Model
##############################

# Save array of k_range and k_scores
krange = np.array(krange)
krange_filename = f'{channel_code}_{model_code}/{channel_code}_{model_code}_krange.npy'
np.save(krange_filename, krange)
print(f"Range of Ks saved to {krange_filename}")
kscore = np.array(kscore)
kscore_filename = f'{channel_code}_{model_code}/{channel_code}_{model_code}_kscore.npy'
np.save(kscore_filename, kscore)
print(f"Range of Ks saved to {kscore_filename}")

# Serialize and save the model
model_filename = f'{channel_code}_{model_code}/{channel_code}_{model_code}_smodel.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# Serialize and save the scaler
scaler_filename = f'{channel_code}_{model_code}/{channel_code}_{model_code}_sscaler.pkl'
joblib.dump(scaler_function, scaler_filename)
print(f"Scaler saved to {scaler_filename}")

# %%
