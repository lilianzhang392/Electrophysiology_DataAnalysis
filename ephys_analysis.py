import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re

def clean_DF(DF_path):
    """
    cleans DF to add data on Rounds and overall time
    
    DF_path: str, local path to V_m .txt file
    
    returns rec, pandas DataFrame, dataframe edited to include continuous time 
    
    """
    rec = pd.read_csv(DF_path, delimiter = "\t")

    # drop column with I_m if included with data
    if rec.columns.size>2:
        rec = rec.drop(rec.columns[1], axis=1)

    rec.loc[-1] = [float(s) for s in rec.columns.to_list()]  # adding a row
    rec.index = rec.index + 1  # shifting index
    rec = rec.sort_index()
    
    rec.columns=['time', 'V_m']
    
    rounds = []
    i=0
    for num in range(len(rec['time'])):
        time_0 = rec['time'][num]
        if time_0==0 and num!=0:
            i+=1
        rounds.append(i)
        
        
    rec['Round'] = rounds
    
    rec['Time']=rec['time']+rec['Round']

    return rec

def extract_signals(rec_path, min_height=60, peak_height_multiplier=3, signal_duration_before=0.1, signal_duration_after=0.1):
    """
    extract signals with peak heights above 60mV and signal duration ±10ms
    
    :param rec_path: str, location of recordings
    :param peak_height_multiplier: int, threshold for peaks
    :param signal_duration: float, ±[signal duration] around the peak time
    
    returns all_extracted_sigs: list of lists, list of rounds, with each round as a list of events
    """
    
    # clean dataframe to include the round and the total time to create a continuous timeline 
    rec = clean_DF(rec_path) 
    rec.index = rec.index + 1  # shifting index
    rec = rec.sort_index() # ensure for continuous index (not reset)

    all_extracted_sigs = []
    # for each round in recording, extract signals
    for _round in rec['Round'].unique():
        _round = rec[rec['Round']==_round] # select round
        
        # filter recording signal (Butterworth filter)
        w = 60 / (1000 / 2) # Normalize the frequency
        b, a = sp.butter(5, w, 'low')
        _round['Filtered V_m'] = sp.filtfilt(b, a, _round['V_m']) # add as a column in DF
        filtered_Vm = _round['Filtered V_m'] # save variable as series for later use
        
        stdev = filtered_Vm.std()
        mean = filtered_Vm.mean()
        
        # find peaks with scipy of any with height>3*stdev + mean
        peaks, _ = sp.find_peaks(filtered_Vm, height=peak_height_multiplier*stdev + mean, distance=150)

        # extract peak signals within ±10ms of the peak found (peak must be >60mV)
        if peaks.size>0 and filtered_Vm.max()>=min_height:
            _round = _round.reset_index()    
            time_pts = [float(_round[_round.index==peak]['Time']) for peak in peaks] # list of time points for peaks
            extracted_signals = [_round[(_round.Time <= time_pt+signal_duration_after)&(_round.Time>=time_pt-signal_duration_before)] 
                                for time_pt in time_pts] # chunks of dataframe where signals are found
            all_extracted_sigs.append(extracted_signals)
        
    return all_extracted_sigs 

def extract_longest_event(extracted_signals_lst):

    """
    extract longest event, in order to align peak dataframe
    :param extracted_signals_lst: list of lists of lists, organized by recording then event in recording
    returns: longest event (series, event longest in length) and max_size_signal (length of longest event)
    """
    # set as first recording initially
    max_size_signal = len(extracted_signals_lst[0][0][0])
    longest_event = extracted_signals_lst[0][0][0]
    
    for recording in extracted_signals_lst:
        num_events = len(recording)
        for _round in recording:
            for event in _round:
                if len(event)>max_size_signal: # update for each longer event
                    max_size_signal=len(event)
                    longest_event = event
                
    return longest_event, max_size_signal # return the event that is longest and length of longest event

def align_peaks(extracted_signals_lst):
    """
    create dataframe with all peaks aligned to the peak of the max length event
    
    :param extracted_signals_lst: list (recordings) of list (rounds) of lists (events in round)
    
    returns ---: pandas DF, dataframe with peaks aligned
    """
    
    # begin dataframe with the longest recording
    longest_event, max_size_signal = extract_longest_event(extracted_signals_lst)
    aligned_df = pd.DataFrame(longest_event['Filtered V_m'].reset_index(drop=True)
                              ).T.reset_index().drop('index', axis=1)

    maxval = aligned_df.max(axis=1) # value V_m of the peak in the longest recording
    peakcol = aligned_df.idxmax(axis=1)[0] # peak column that all recordings will be aligned to

    row_num = 1 
    for recording in extracted_signals_lst:
        for _round in recording: 
            for event in _round:
                event = event['Filtered V_m'].fillna(0).reset_index(drop=True)
                event_peakcol = event.idxmax(axis=1)
                shift_diff = peakcol - event_peakcol # difference to adjust columns by

                aligned_df = aligned_df.append(event, ignore_index=True)
                aligned_df.loc[row_num] = aligned_df.loc[row_num].shift(shift_diff)

                row_num+=1
    
    aligned_df = aligned_df.iloc[1:, :].fillna(0) # drops first row to ensure no duplicates
    
    return aligned_df

def extract_features(aligned_df, peak_height_multiplier):
    
    """
    select shape_features to define an event
    
    :param aligned_df: pandas DataFrame, contains all filtered events, with peaks aligned to same column
    :param peak_height_multiplier: int, multiplied to SD to extract number of peaks in the event
    
    returns feature_df: pandas DataFrame, contains relevant shape features 
    """
    
    # set up feature df
    features = ['pos_peak_amp', 'neg_trough_before_amp', 
                'neg_trough_after_amp', 'spike_width_peak_to_troughbefore', 
                'spike_width_troughafter_to_peak', 'num_peaks']
    feature_df = pd.DataFrame(columns=features)

    # extract amplitude of max peak
    max_col = aligned_df.idxmax(axis=1).mode()[0] # finds column with the max V_m
    max_peak = aligned_df[max_col]
    feature_df['pos_peak_amp']=max_peak

    # extract all troughs
    trough_col = aligned_df.apply(lambda vm: sp.argrelextrema(vm.to_numpy(), np.less), axis=1)

    # extract amplitude of negative trough before the peak
    trough_before_col = trough_col.apply(lambda lst:[i for i in lst[0] if i<max_col])
    trough_before_col = trough_before_col.apply(lambda lst: lst[-1] if len(lst)>0 else 0)
    trough_before_amp = aligned_df.lookup(trough_before_col.index, trough_before_col)
    feature_df['neg_trough_before_amp'] = trough_before_amp

    # extract amplitude of negative trough after the peak
    trough_after_col = trough_col.apply(lambda lst:[i for i in lst[0] if i>max_col])
    trough_after_col = trough_after_col.apply(lambda lst: lst[0] if len(lst)>0 else 0)
    trough_after_amp = aligned_df.lookup(trough_after_col.index, trough_after_col)
    feature_df['neg_trough_after_amp'] = trough_after_amp

    # extract idx diff (time) between trough before and peak 
    width_peak_to_troughbefore = aligned_df.idxmax(axis=1) - trough_before_col
    feature_df['spike_width_peak_to_troughbefore'] = width_peak_to_troughbefore

    # extract idx diff (time) between peak and trough after
    width_troughafter_to_peak = trough_after_col - aligned_df.idxmax(axis=1)
    feature_df['spike_width_troughafter_to_peak'] = width_troughafter_to_peak

    # extract the number of peaks in a 10ms signal
    peaks = aligned_df.apply(lambda vm: sp.find_peaks(vm, height=(peak_height_multiplier*np.std(vm))+np.mean(vm), distance=150), axis=1)
    num_peaks = peaks.apply(lambda row: len(row[0])) 
    feature_df['num_peaks'] = num_peaks

    feature_df = feature_df[feature_df['num_peaks']!=0]

    return feature_df

def k_plots_inertia(X, k_list, jellyname, condition, path):
    """Generates plots for determining the appropriate number of clusters for
    k-means clustering through inertia (sum of squared errors from cluster center)
    """
    # Intra-Cluster Sum of Squared Errors
    X = X
    WSS = list()
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=100).fit(X)
        WSS.append(kmeans.inertia_)
   
    # plotting
    elbow_df = pd.DataFrame(zip(k_list, WSS), columns=['k', 'WSS'])

    plt.rcParams.update({'font.size': 22})
    plt.tight_layout()

    elbow_df.plot(x='k', xticks=k_list, grid=True, figsize=(100, 20),
                  title='Within Cluster Sum of Squared Errors for various k (Inertia)')
    
    # plt.savefig('%s/%s_%s_kmeans.png' % (path, jellyname, condition), bbox_inches='tight', facecolor="white")
    
    plt.show()

#def pca_plot(X, y, colors, savepath, savefig=False):