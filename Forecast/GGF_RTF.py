# -*- coding: utf-8 -*-
"""
Program name: 
 GGF_RTF.py

Program objectives:
 This program reads in regression model coefficients produced by program 
 GGF_Training_Model.py, and real time solar wind data from a Met Office API, 
 and produces a an approximately one-hour forecast of the ground external 
 (i.e. excluding the internal field contribution) geomagnetic field (GGF) 
 at the three UK magnetometer stations operated by the British Geological 
 Survey: Eskdalemuir, Hartland and Lerwick. The forecast is given as the max, 
 min and mean of a 100-model ensemble at each time-step, where the 100 models 
 are obtained by training on 100 random samples of 80% of a database of 
 geomagnetic storm periods.

Input data requirements: 
 - GGF_Training_Model_Dataset_of_Stored_Model_Coefficients.pkl
    - Description: regression model coefficients relating the solar wind 
      epsilon parameter to the ground geomagnetic variation. These are used by 
      the ground geomagnetic prediction model in this program.
    - Source: produced by program 'GGF_Training_Model.py'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: a Python list containing a 5-dimensional numpy array of model 
      coefficients. The dimensions are: [stations, local time bins, magnetic 
      components, model parameters, randomised training data instances]. The 
      details of each dimension is as follows.
       - There are three stations (i.e. observatories) in the order: 
         Eskdalemuir, Hartland, Lerwick.
       - There are 24 local time bins, which are spaced at 1 hour intervals 
         with a 3-hour width for each bin. The first bin has a centroid at 
         00:30 local time, and spans the three hours from 23:00 to 02:00 local 
         time.
       - The three magnetic components have the order: x, y, z.
       - There are 6 model parameters which are used to solve for an estimate 
         of the ground geomagnetic variation (y) for a given magnetic component
         at a given station at a given local time. The equation being solved is 
         y = a + b*(sin(DOY)) + c*(cos(DOY)) + d*epsilon + e*(sin(DOY) * epsilon) + f*(cos(DOY) * epsilon)
         where a is the intercept coefficient, and the other coefficients 
         describe the effect of the following parameters: b for sine of day of 
         year (DOY), c for cosine of DOY, d for epsilon, e for 
         (sine of DOY)*epsilon, and f for (cosine of DOY)*epsilon.
       - There are 100 randomised training data instances, each of which 
         relates to a different random selection of 80% of the full set of 
         geomagnetic storm intervals used to train the model.
 - n4-key.txt
    - Description: the access key for the Met Office API used to access real 
      time solar wind data.
    - Source: Met Office, pers. comm.
    - Is it included here: yes.
    - Format: ascii text file contianing the API key as a single string.
 - substorm_model_solar_wind_weights_mean_std.json
    - Description: a file of the mean and standard deviations of the OMNI solar
      wind data from 1996-01-01 to 2014-12-31, for the solar wind parameters 
      used to train the substorm onset likelihood forecast model. The same mean
      and standard deviation information is used in this program to normalise 
      the real-time solar wind dtaa obtained from the API, before it is used to 
      produce the substorm onset likelihood prediction.
    - Source: this is a JSON-formatted copy of the values in the file 
      'omn_mean_std.npy', which is produced by program 
      'Substorm_Training_Model.py' and is stored in the directory 
      'Substorm/data'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: JSON format, with 10 data rows of alternating mean and standard 
      deviation values for the following parameters in order, where all 
      components relate to the GSM frame: IMF Bx, IMF By, IMF Bz, solar wind 
      x-component velocity, proton density. The magnetic components have units 
      of nanoTesla, the velocity has units of metres per second, and the proton
      density has units of n/cc.
 - substorm_model_coefficient_weights.epoch_200.val_loss_0.58.val_accuracy_0.70.hdf5
    - Description: the model coefficients used to forecast substorm onset from 
      solar wind data, trained by a convolutional neural network model.
    - Source: produced by program 'Substorm_Training_Model.py'. This is a 
      renamed copy of the coefficients file in directory 
      'Substorm/trained_model'.
    - Is it included here: yes. These data do not need to be reproduced 
      because the file is provided in the same directory as this program.
    - Format: Hierarchical Data Format HDF5.
 - Access to the Met Office API at 
   'https://gateway.api-management.metoffice.cloud/swx_swimmr_n4/1.0/', to 
   import real time solar wind observations from the L1 Lagrange point and 
   ground geomagnetic observations from the UK's three geomagnetic 
   observatories operated by the British Geological Survey.

Outputs: these are each in directory 'Forecast/Temp_storage_for_output_GGF_forecast':
 - real_time_magnetic_field_forecast_from_program_GGF_RTF_version_BRTFv2p3.dat
    - Description: an ascii-formatted text file of the forecast 
      values from the regression model ensemble (for the ground geomagnetic 
      perturbation) and the convolutional neural network model (for the 
      substorm onset likelihood). The forecast values extend about one hour
      into the future (the exact forecast span is dynamic and depends on the 
      solar wind speed).
    - Source: made by program GGF_RTF.py.
    - Format: the column format is as follows: [Day-Month-Year,  Hour:Minute,  
      Model Type Flag,  Substorm onset probability, Esk x mean,  Esk y mean,  
      Esk z mean,  Had x mean,  Had y mean,  Had z mean,  Ler x mean,  
      Ler y mean,  Ler z mean,  Esk x min,   Esk y min,   Esk z min,  
      Had x min,   Had y min,   Had z min,   Ler x min,   Ler y min,  
      Ler z min,   Esk x max,   Esk y max,   Esk z max,   Had x max,   
      Had y max,   Had z max,   Ler x max,   Ler y max,   Ler z max], where
      'Esk' is the observatory Eskdalemuir, 'Har' is Hartland, 'Ler' is 
      Lerwick, 'x','y','z' are the ground geomagnetic field perturbation 
      components, 'mean' is the mean value of the 100 model ensemble at a given 
      epoch, and likewise for 'max' and 'min'. The Model Type Flag has the 
      following meanings. 0: no problems: model forecast is based on the 
      epsilon coupling function. 1: some parts of the solar wind data were not 
      available for this epoch of the real-time solar wind stream, so the 
      forecast is based on interpolated values. 2: there were no solar wind 
      data at this epoch, so the output is nan. The substorm onset probability 
      defines the onset likelihood (from 0 to 1, with 1 the most likely), for 
      the hour after the model run time. Hence, the onset forecast is only 
      valid for that hour: the output file has nan values for onset 
      probabilities at forecast epochs greater than [current time + 60 mins].

Instructions for running this program:
 This program can be run on any machine with internet access. This is required
 for downloading the real-time API data. Optionally, the program can also be 
 run in a virtual environment using Docker (instructions below) -- in this 
 case, internet access is required to import the code files from a github 
 repository. Before running, the user should set the 'in_docker' flag to True 
 or False, dependent on whether the program is being run in Docker. If running 
 locally, the user should manually set the variable 'WORKDIR', dependent on 
 where the code is stored. If running on a BAS machine, the code is in this 
 directory: /data/psdcomplexity/eimf/SAGE_Model_Handover/Forecast.

Instructions for running in Docker:
 - Install Docker on the local machine.
 - Set the 'in_docker' flag in the code below to True.
 - The Dockerfile can then be run using the following commands:
    - docker builder prune -a (to remove previous cloned github repositories and make room for newer ones)
    - docker build -f [on the local machine, the directory and filename of the Dockerfile stored here in Forecast/Dockerfile] -t ggf_image . (to build the image)
    - docker run --name ggf_container ggf_image GGF_RTF.py (to run the Python script 'GGF_RTF.py' from the image in a container)
    - docker cp ggf_container:/root/run_environment/Temp_storage_for_output_GGF_forecast/ [on the local machine, the directory where you wish the output of the Python script to be copied to] (this copies the output forecast from a location in the Docker container to somewhere else on the local machine)

@author: robore@bas.ac.uk. Robert Shore, ORCID: orcid.org/0000-0002-8386-1425.
For author's reference: this program was based on program 
 BGS_RealTime_Forecast_v2p3.py, shorthand 'BRTFv2p3'.
"""
output_version_identifier = 'BRTFv2p3'#Version string, for author's reference.

#%% Load packages.

import os
# Import these packages.
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#suppresses tensorflow warnings.
import json
import numpy as np
import datetime
import time
import math
import pickle
import datetime as dt
from tensorflow import keras
import requests
import pandas as pd

#Set geometric constants.
#Degrees to radians
rad = np.pi / 180 #scalar
#Radians to degrees
deg = 180 / np.pi #scalar

#Define flag to switch between developing locally and running in Docker virtual environment.
in_docker = True

#Set directory variables dependent on run-environment.
if in_docker:
    WORKDIR = os.path.join(os.sep,'root','run_environment')
else:
    WORKDIR = os.path.join('C:' + os.sep,'Users','robore','BAS_Files','Research','Code','SAGE','SAGE_Model_Handover','Forecast')
    #Dear user: if running on a BAS machine, please edit the above string to be
    # WORKDIR = '/data/psdcomplexity/eimf/SAGE_Model_Handover/Forecast'.
#End conditional: tell the program whether to run in Docker or locally.

#%% Define LT bins used by the model, at 180 mins width and 60 mins cadence.

#Define LT bin width and LT bin cadence.
LT_bin_width = 180
LT_bin_cadence = 60#units of minutes.

#Define the centroids of the LT bins, in the form of datetimes throughout a 
# given day.
LT_bin_centroids_datetime = []#will be list of datetime objects of size [24].
#I choose the bin edge to be at 00:00 for a bin width which matches the bin 
# cadence (this will make it equal to earlier runs of 18-min contiguous bins).
# Hence the centroid of the first bin is at 00:09 for an 18-min cadence, and 
# 00:30 for a 60-min cadence.
date_x = datetime.datetime(2000,1,1) + datetime.timedelta(minutes=(LT_bin_cadence/2))#arbitrary scalar datetime object for starting date, at hour 0 on some given day, plus half the bin cadence.
LT_bin_centroids_datetime.append(date_x)
while (date_x + datetime.timedelta(minutes=LT_bin_cadence)) < datetime.datetime(2000,1,2):
    date_x += datetime.timedelta(minutes=LT_bin_cadence)
    LT_bin_centroids_datetime.append(date_x)#appends time in datetime format.
#End iteration over LT bin definitions within a sample day.

#Now define the edges of the bins by stepping out (both forward and backward 
# in time) from each bin's centroid. No seconds variable, because the spans 
# are at the 0th second. Convert to ordinal day fraction.
LT_bin_starts_day_fraction = np.empty([np.shape(LT_bin_centroids_datetime)[0],1])#np array of floats, size [LT bins by 1].
LT_bin_ends_day_fraction = np.empty([np.shape(LT_bin_centroids_datetime)[0],1])#np array of floats, size [LT bins by 1].
LT_bin_centroids_day_fraction = np.empty([np.shape(LT_bin_centroids_datetime)[0],1])#np array of floats, size [LT bins by 1].
LT_bin_starts_datetime = np.empty([np.shape(LT_bin_centroids_datetime)[0],1], dtype='datetime64[s]')#np array of datetime64[s], size [LT bins by 1].
LT_bin_ends_datetime = np.empty([np.shape(LT_bin_centroids_datetime)[0],1], dtype='datetime64[s]')#np array of datetime64[s], size [LT bins by 1].
seconds_per_day = 24*60*60# hours * mins * secs
for i_LT in range(len(LT_bin_centroids_datetime)):
    #Define temporary datetime objects for the starts and ends of this bin, 
    # for later conversion to decimal day format (within this cell).
    LT_bin_starts_datetime[i_LT] = LT_bin_centroids_datetime[i_LT] - datetime.timedelta(minutes=(LT_bin_width/2))
    LT_bin_ends_datetime[i_LT] =   LT_bin_centroids_datetime[i_LT] + datetime.timedelta(minutes=(LT_bin_width/2))
    
    #Create decimal day versions of the LT bin limits and centroids.
    LT_bin_starts_day_fraction[i_LT,0] = datetime.timedelta(hours=LT_bin_starts_datetime[i_LT][0].astype(datetime.datetime).time().hour, minutes=LT_bin_starts_datetime[i_LT][0].astype(datetime.datetime).time().minute, seconds=0).total_seconds()/seconds_per_day
    LT_bin_ends_day_fraction[i_LT,0] = datetime.timedelta(hours=LT_bin_ends_datetime[i_LT][0].astype(datetime.datetime).time().hour, minutes=LT_bin_ends_datetime[i_LT][0].astype(datetime.datetime).time().minute, seconds=0).total_seconds()/seconds_per_day
    LT_bin_centroids_day_fraction[i_LT,0] = datetime.timedelta(hours=LT_bin_centroids_datetime[i_LT].time().hour, minutes=LT_bin_centroids_datetime[i_LT].time().minute, seconds=0).total_seconds()/seconds_per_day
#End loop over LT bins.

#Manually alter the last element of the LT bin end hour to be 1, rather than 0.
# This occurs because the ending date of the LT_bin_edges variable is the start 
# (i.e. hour zero) of the next day, rather than hour 24 on the same day.
if LT_bin_ends_day_fraction[len(LT_bin_ends_day_fraction) - 1,0] == 0:
    LT_bin_ends_day_fraction[len(LT_bin_ends_day_fraction)-1,0] = 1.0
#End conditional: alter the last element of the LT bin ends if you need to.

#%% Load in the regression model coefficients for the 100-model ensemble.
#The chosen model is similar to BIRv5p5, in that it is a DOY-dependent, auroral 
# boundary location independent regression model driven by epsilon. The 
# differences to BIRv5p5 (as it is used in 'BRTFv1p4_') are that the LT bins are 
# 3 hours wide instead of 18 mins wide, and the trained model is from an 
# ensemble of 100 randomised instances of the training data, with the training 
# performed by program 'RTRv3p0'.

#Load the model coefficients.
with open(os.path.join(WORKDIR,'Storage_for_model_coefficients','GGF_Training_Model_Dataset_of_Stored_Model_Coefficients.pkl'),'rb') as f:  # Python 3: open(..., 'rb')
    all_randomised_storms_sets_model_coeffs = pickle.load(f)
#End indenting for this load command.

#Rename, and extract from list.
model_coefficients_all_stations_all_components_all_ensemble = all_randomised_storms_sets_model_coeffs[0]#size [3 stations by 24 LT bins by 3 components by 6 model parameters by 100 trained model ensemble instances].

#%% Define Met Office API key.

#Define API key filename location.
MO_API_key_filename = os.path.join(WORKDIR,'Storage_for_model_coefficients','n4-key.txt')

#Load API key to local variable.
with open(MO_API_key_filename, 'r') as file:
    MO_API_key = file.read().replace('\n', '')
#End indenting for file-open command.

#%% Define the current time, accounting for time zone.

#Create a variable for the current time, and convert to UTC if daylight saving
# is in effect.
if time.localtime().tm_isdst > 0:
    current_time = np.datetime64(dt.datetime.now()) - np.timedelta64(1,'h')#scalar datetime64 object.
else:
    current_time = np.datetime64(dt.datetime.now())#scalar datetime64 object.
#End conditional: check if daylight savings time applies.

#%% Load in Met Office API data.

# ----------------------- Import Met Office API mag data.

#Define Met Office API common url path.
MO_API_url_base_path = 'https://gateway.api-management.metoffice.cloud/swx_swimmr_n4/1.0/'

#How many minutes into the past to extract data for. T
#In the GGF model, the span extracted here must cover at least the maximum 
# propagation time from L1 to the bow shock nose, plus the model lags, which 
# are presently 20 mins max.
#In the substorm model, the extracted data 120 mins into the past from whatever
# (future) time point results from the maximum propagation time from L1 to the 
# bow shock nose, plus 10 mins. This extraction is based on regridded data 
# (post-propagation), so there may be issues if we don't have a bit of a 
# time-buffer.
#If we extract 180 mins into the past from the current time, it should always work.
minutes_ago = 180

#Convert the start-time of the extraction period to a formatted string.
start_time_string = (current_time.astype(datetime.datetime) - dt.timedelta(minutes=minutes_ago)).strftime('%Y-%m-%dT%H:%M:%S')

#Define specific Met Office API url path.
MO_API_mag_url_path = MO_API_url_base_path + 'v1/data/rtsw_magnetometer' + '?from=' + start_time_string

#Access the API.
MO_API_mag_response = requests.get(MO_API_mag_url_path, headers={"apikey": MO_API_key, "accept": "application/json"})

#Determine if the API is being accessed properly, and exit the program if not.
if MO_API_mag_response.status_code != 200:
    print("exec_query: error in query submitted via requests.get()")
    print("            error code returned: ", MO_API_mag_response.status_code)
    sys.exit(1)
#End conditional: check if API is accessed OK.

#Convert to list of json-format data, extract only 'data', ignoring pagination.
MO_API_mag_data_json = MO_API_mag_response.json()['data']

#Extract data.
MO_API_mag_data_df = pd.DataFrame([{
    'timestamp': x['timestamp'],
    'active': x['active'],
    'source': x['source'],
    'bx_gsm': x['bx_gsm'],
    'by_gsm': x['by_gsm'],
    'bz_gsm': x['bz_gsm']
} for x in MO_API_mag_data_json])#size [minutes in past day (ish) by 6 columns].
#End indenting.

# ----------------------- Import Met Office API plasma data.

#Now, import the plasma data for the same timespan -- do not expect perfect temporal correspondence.
#Define specific Met Office API url path.
MO_API_plasma_url_path = MO_API_url_base_path + 'v1/data/rtsw_wind' + '?from=' + start_time_string

#Access the API.
MO_API_plasma_response = requests.get(MO_API_plasma_url_path, headers={"apikey": MO_API_key, "accept": "application/json"})

#Convert to list of json-format data, extract only 'data', ignoring pagination.
MO_API_plasma_data_json = MO_API_plasma_response.json()['data']

#Extract data.
MO_API_plasma_data_df = pd.DataFrame([{
    'timestamp': x['timestamp'],
    'active': x['active'],
    'source': x['source'],
    'proton_speed': x['proton_speed'],
    'proton_density': x['proton_density']
} for x in MO_API_plasma_data_json])#size [minutes in past day (ish) by 5 columns].
#End indenting.

# ----------------------- Post-processing, for all Met Office API dataframes.

#Remove non-active satellite measurements from each dataframe to obtain a continuous data record.
MO_API_mag_data_df = MO_API_mag_data_df.drop(MO_API_mag_data_df[MO_API_mag_data_df['active'] == False].index)
MO_API_plasma_data_df = MO_API_plasma_data_df.drop(MO_API_plasma_data_df[MO_API_plasma_data_df['active'] == False].index)

#Replace MO API dataframe 'None' values with 'nan' values.
MO_API_mag_data_df.fillna(value=np.nan, inplace=True)
MO_API_plasma_data_df.fillna(value=np.nan, inplace=True)

#Define variables of Met Office API timestamps.
MO_API_mag_times = np.array(MO_API_mag_data_df['timestamp']).astype('datetime64[s]')#size [minutes in past day (ish) by 0].
MO_API_plasma_times = np.array(MO_API_plasma_data_df['timestamp']).astype('datetime64[s]')#size [minutes in past day (ish) by 0].

#Convert the Met Office API solar wind variables that will need additional 
# processing to numpy arrays.
MO_API_IMF_Bx = np.array(MO_API_mag_data_df['bx_gsm'])#size [minutes in past day (ish) by 0].
MO_API_IMF_By = np.array(MO_API_mag_data_df['by_gsm'])#size [minutes in past day (ish) by 0].
MO_API_IMF_Bz = np.array(MO_API_mag_data_df['bz_gsm'])#size [minutes in past day (ish) by 0].
MO_API_sw_speed = np.array(MO_API_plasma_data_df['proton_speed'])#size [minutes in past day (ish) by 0].
MO_API_sw_density = np.array(MO_API_plasma_data_df['proton_density'])#size [minutes in past day (ish) by 0].


#Remove duplicate instances in the solar wind API data where multiple 
# satellites are tagged as 'active' for the same timestamp. At this point in 
# the code, there may be differences between the mag and plasma API dataframe 
# lengths, but they should not contain duplicate temporal entries unless there
# is more than one active satellite at the same time. So, we check for temporal 
# duplicates.
# ---- For the plasma API data.
index_of_duplicate_MO_API_plasma_times = []
for i_t in range(len(MO_API_plasma_times)):
    #Define an index of where this time exists in the full set of times.
    index_same_time = np.nonzero(MO_API_plasma_times == MO_API_plasma_times[i_t])[0]#np array of ints, size [duplicate dates by 0]
    
    #If there's more than two duplicate elements, then the code below will not 
    # work, so flag that occurrence.
    if len(index_same_time) > 2:
        print('Multiple duplicate entries in MO_API_plasma_times.')
    #End conditional: data check.
    
    #If there's more than one of this time, flag the non-DSCOVR time(s) for later removal.
    if len(index_same_time) > 1:
        index_DSCOVR_element_mask = np.flatnonzero(MO_API_plasma_data_df['source'].iloc[index_same_time] != 'DSCOVR')
        
        #If there's no non-DSCOVR satellite, then it's a duplication of the 
        # temporal record, rather than an instance of both satellites being 
        # 'active', so we just take the first of the duplicates in that case.
        if len(index_DSCOVR_element_mask) == 0:
            index_DSCOVR_element_mask = 0
        #End conditional: data check.
        
        index_of_duplicate_MO_API_plasma_times = np.append(index_of_duplicate_MO_API_plasma_times,index_same_time[index_DSCOVR_element_mask])
    #End conditional.
#End loop over times.

# ---- For the mag API data.
index_of_duplicate_MO_API_mag_times = []
for i_t in range(len(MO_API_mag_times)):
    #Define an index of where this time exists in the full set of times.
    index_same_time = np.nonzero(MO_API_mag_times == MO_API_mag_times[i_t])[0]#np array of ints, size [duplicate dates by 0].
    
    #If there's more than two duplicate elements, then the code below will not 
    # work, so flag that occurrence.
    if len(index_same_time) > 2:
        print('Multiple duplicate entries in MO_API_mag_times.')
    #End conditional: data check.
    
    #If there's more than one of this time, flag the non-DSCOVR time(s) for later removal.
    if len(index_same_time) > 1:
        index_DSCOVR_element_mask = np.flatnonzero(MO_API_mag_data_df['source'].iloc[index_same_time] != 'DSCOVR')
        
        #If there's no non-DSCOVR satellite, then it's a duplication of the 
        # temporal record, rather than an instance of both satellites being 
        # 'active', so we just take the first of the duplicates in that case.
        if len(index_DSCOVR_element_mask) == 0:
            index_DSCOVR_element_mask = 0
        #End conditional: data check.
        
        index_of_duplicate_MO_API_mag_times = np.append(index_of_duplicate_MO_API_mag_times,index_same_time[index_DSCOVR_element_mask])
    #End conditional.
#End loop over times.

#Retrieve just the unique indices of the duplicate epochs. This is required 
# since the progression through the entire set of times will, by definition, 
# count each duplicate twice.
index_of_MO_API_plasma_times_to_remove = np.transpose(np.unique(index_of_duplicate_MO_API_plasma_times)[np.newaxis].astype(np.int64))[:,0]
index_of_MO_API_mag_times_to_remove = np.transpose(np.unique(index_of_duplicate_MO_API_mag_times)[np.newaxis].astype(np.int64))[:,0]

#Remove the flagged duplicates.
MO_API_mag_data_df.drop(MO_API_mag_data_df.index[index_of_MO_API_mag_times_to_remove],inplace=True)
MO_API_mag_times = np.delete(MO_API_mag_times,index_of_MO_API_mag_times_to_remove)
MO_API_IMF_Bx = np.delete(MO_API_IMF_Bx,index_of_MO_API_mag_times_to_remove)
MO_API_IMF_By = np.delete(MO_API_IMF_By,index_of_MO_API_mag_times_to_remove)
MO_API_IMF_Bz = np.delete(MO_API_IMF_Bz,index_of_MO_API_mag_times_to_remove)
MO_API_plasma_data_df.drop(MO_API_plasma_data_df.index[index_of_MO_API_plasma_times_to_remove],inplace=True)
MO_API_plasma_times = np.delete(MO_API_plasma_times,index_of_MO_API_plasma_times_to_remove)
MO_API_sw_speed = np.delete(MO_API_sw_speed,index_of_MO_API_plasma_times_to_remove)
MO_API_sw_density = np.delete(MO_API_sw_density,index_of_MO_API_plasma_times_to_remove)

#%% Force temporal equivalence in Met Office RTSW API data.

#We are using the Met Office plasma API temporal record as our ballistic 
# propagation basis, so any mag records which do not match the non-propagated 
# plasma temporal record must be removed.
#Here we list, then remove, the Met Office mag API records which do not have a 
# match in the Met Office plasma API records.
index_MO_API_mag_data_to_remove = []#pertains to rows of MO_API_plasma_data_df and MO_API_plasma_times.
for i_t in range(len(MO_API_mag_times)):
    if np.all(MO_API_plasma_times != MO_API_mag_times[i_t]):
        index_MO_API_mag_data_to_remove.append(i_t)
    #End conditional: if this MO mag API time has no match in the entire MO plasma API temporal dataset, then flag it for removal.
#End loop over Met Office temporal data.
MO_API_mag_data_df.drop(MO_API_mag_data_df.index[index_MO_API_mag_data_to_remove],inplace=True)
MO_API_mag_times = np.delete(MO_API_mag_times,index_MO_API_mag_data_to_remove)
MO_API_IMF_Bx = np.delete(MO_API_IMF_Bx,index_MO_API_mag_data_to_remove)
MO_API_IMF_By = np.delete(MO_API_IMF_By,index_MO_API_mag_data_to_remove)
MO_API_IMF_Bz = np.delete(MO_API_IMF_Bz,index_MO_API_mag_data_to_remove)

#Vice-versa, list and remove any plasma records which do not match the new 
# downsized mag record.
index_MO_API_plasma_data_to_remove = []#pertains to rows of MO_API_plasma_data_df and MO_API_plasma_times.
for i_t in range(len(MO_API_plasma_times)):
    if np.all(MO_API_mag_times != MO_API_plasma_times[i_t]):
        index_MO_API_plasma_data_to_remove.append(i_t)
    #End conditional: if this MO plasma API time has no match in the entire MO mag API temporal dataset, then flag it for removal.
#End loop over Met Office temporal data.
MO_API_plasma_data_df.drop(MO_API_plasma_data_df.index[index_MO_API_plasma_data_to_remove],inplace=True)
MO_API_plasma_times = np.delete(MO_API_plasma_times,index_MO_API_plasma_data_to_remove)
MO_API_sw_speed = np.delete(MO_API_sw_speed,index_MO_API_plasma_data_to_remove)
MO_API_sw_density = np.delete(MO_API_sw_density,index_MO_API_plasma_data_to_remove)

#Double-check equivalence, since we will use the plasma temporal records as a 
# temporal reference for the mag data henceforth.
if np.any(MO_API_plasma_times != MO_API_mag_times):
    print('Warning: temporal errors in Met Office RSTW API.')
#End conditional: data check.

#%% Temporal propagation of solar wind data.
#The GGF model requires data which are propagated to the bow shock nose, plus 
# additional time lags for the time taken for the solar wind to propagate to 
# the magnetopause, and for the ionosphere to reconfigure to the solar wind driving.
#
#The substorm model requires data which are propagated to the bow shock nose, plus 
# a fixed 10-min time lag to account for the propagation delay to the magnetopause. 
# The neural network model accounts for the rest of the reconfiguration timescale.
#
#I will use the variable 'MO_API_plasma_times_bow_shock_nose' to indicate temporal 
# values propagated to the bow shock nose. Variable 'MO_API_plasma_times_bow_shock_nose_lagged'
# refers to times which are referenced to the specific lags required by the GGF model.
# Variable 'MO_API_plasma_times_bow_shock_nose_plus10min' refers to temporal 
# values used by the substorm forecast model. These variables indicate when the 
# L1 measurements will arrive at a certain downstream point. The L1 data will 
# need temporal re-sorting based on these propagated timestamps before they are 
# used to drive the model(s).

#Interpolate over the (nan) gaps in the RTSW speed data, 
# using the command '.astype(float)' to convert the datetime64 values to 
# ordinal seconds.
MO_API_sw_speed_interpolated = np.interp(
    MO_API_plasma_times.astype(float),
    MO_API_plasma_times[np.nonzero(~np.isnan(MO_API_sw_speed))[0]].astype(float),
    MO_API_sw_speed[np.nonzero(~np.isnan(MO_API_sw_speed))[0]])#np array of floats, size [(minutes in day) by 0]
#End indenting for variable definition.

#Preallocate storage for propagated times.
MO_API_plasma_times_bow_shock_nose = np.empty([len(MO_API_plasma_times),1],dtype='datetime64[s]')#np array of datetime64[s], size [minutes in day by 1]
MO_API_plasma_times_bow_shock_nose_lagged = np.empty([len(MO_API_plasma_times),1],dtype='datetime64[s]')#np array of datetime64[s], size [minutes in day by 1]
MO_API_plasma_times_bow_shock_nose_plus10min = np.empty([len(MO_API_plasma_times),1],dtype='datetime64[s]')#np array of datetime64[s], size [minutes in day by 1]

#Define constants.
Earth_radius = 6371009.0
metres_to_km = 1e-3

#Loop over data records, and propagate each epoch to the bow shock nose.
for i_t in range(len(MO_API_plasma_times)):
    #To temporally assign the data to the bow shock nose, we ballistically propagate 
    # the Met Office API data from its measurement location (Lagrange 1, at 
    # ~220 Earth radii) to the bow-shock of the magnetosphere (12 Earth radii) 
    # using solar wind speed. Projection is simple linear time = distance/speed.
    #The RTSW speed data is in units of km/s, so we want the standoff 
    # propagation distance to be in km too, such that the propagation 
    # time will be in seconds.
    ballistic_propagation_in_seconds = \
        ((220 - 12) * (Earth_radius * metres_to_km)) / MO_API_sw_speed_interpolated[i_t]#scalar float, units of seconds.
    #End indenting for variable definition.
    
    #Compute and store the propagated times.
    MO_API_plasma_times_bow_shock_nose[i_t] = MO_API_plasma_times[i_t] + np.timedelta64(np.int64(np.round(ballistic_propagation_in_seconds)),'s')#scalar datetime64[s] object.
    MO_API_plasma_times_bow_shock_nose_lagged[i_t] = MO_API_plasma_times[i_t] + np.timedelta64(np.int64(np.round(ballistic_propagation_in_seconds)),'s') + np.timedelta64(20,'m')#scalar datetime64[s] object.
    MO_API_plasma_times_bow_shock_nose_plus10min[i_t] = MO_API_plasma_times[i_t] + np.timedelta64(np.int64(np.round(ballistic_propagation_in_seconds)),'s') + np.timedelta64(10,'m')#scalar datetime64[s] object.
    
    #Check for 'NaT' values, caused by NaN values in the solar wind speed.
    if pd.isnull(MO_API_plasma_times_bow_shock_nose[i_t]):
        print('Warning: NaT values encountered in temporal propagation to bow shock nose.')
    #End conditional: data check.
#End loop over each RTSW storm days epoch.

#%% Account for missing values in the solar wind speed data.
#The solar wind plasma data are more susceptible to missing values than the IMF 
# data are. We are OK with using the interpolated solar wind speed data to 
# drive the model because we have already had to use it to temporally propagate
# the IMF data, and any additional error is assumed to be acceptable.

#We need to flag where the missing data were, so that we know when the model 
# forecast is based on interpolated data. The transition from an irregular 
# temporal basis to a regular temporal grid does not help here. This is a 
# work-around. We set a series to equal 1 when NaN and 0 otherwise, interpolate 
# that, and threshold it at 0.5 to indicate NaN in the interpolated series.
nan_indicator_series_original = np.zeros(len(MO_API_sw_speed))#size [minutes in past day (ish) by 0].
nan_indicator_series_original[np.nonzero(np.isnan(MO_API_IMF_Bx))] = 1
nan_indicator_series_original[np.nonzero(np.isnan(MO_API_sw_speed))] = 1

#Define a neatened version of the pandas 'forward-fill' function, which 
# replaces missing values with the last recorded value. This is preferable 
def pandas_fill(arr):
    df = pd.DataFrame(arr)
    df.fillna(method='ffill', axis=1, inplace=True)
    out = df.values
    return out
#End indentation for function definition.

#Here we are using an interpolation scheme based on the last recorded value, 
# which allows for extrapolation in the 1-hour-ish forecast span.
MO_API_sw_speed_forward_filled = pandas_fill(MO_API_sw_speed)#size [minutes in past day (ish) by 1].

#%% Pre-process GGF model driver data: Account for solar wind arrival-time overlaps.
#That is, we're using a pre-propagated solar wind data set, so there exists the
# possibility that some temporal elements can 'arrive' at Earth in the reverse 
# order to that of their measurement order.  Here, we sort the propagated times
# so that this does not happen.

#Define an index which will sort the propagated solar wind dates in ascending
# order, starting from the lowest value.
indices_of_sorted_lagged_sw_data = np.argsort(MO_API_plasma_times_bow_shock_nose_lagged,axis=0)#size [minutes in past day (ish) by 1].

#Apply the index to the solar wind data that you use to drive the GGF model.
# The GGF_times are propagated, lagged, and sorted.
GGF_IMF_Bx = MO_API_IMF_Bx[indices_of_sorted_lagged_sw_data[:,0]]#size [minutes in past day (ish) by 0].
GGF_IMF_By = MO_API_IMF_By[indices_of_sorted_lagged_sw_data[:,0]]#size [minutes in past day (ish) by 0].
GGF_IMF_Bz = MO_API_IMF_Bz[indices_of_sorted_lagged_sw_data[:,0]]#size [minutes in past day (ish) by 0].
GGF_sw_speed = MO_API_sw_speed_forward_filled[indices_of_sorted_lagged_sw_data[:,0],:]#size [minutes in past day (ish) by 1].
nan_indicator_series_original = nan_indicator_series_original[indices_of_sorted_lagged_sw_data[:,0]]#size [minutes in past day (ish) by 0].
GGF_times = MO_API_plasma_times_bow_shock_nose_lagged[indices_of_sorted_lagged_sw_data[:,0]]#size [minutes in past day (ish) by 1].

#At this point in the code, a solar wind measurement associated with a time-shifted 
# tag which has a value of the current time (i.e. contemporaneous, i.e. now) is 
# able to provide a nowcast of the ionospheric state, and all following 
# measurements in the real-time series represent forecast capability.

#%% Temporally re-grid the propagated, sorted solar wind epochs to a regular 1-min cadence, for a recent subset of times.
#Specifically, the subset spans from the now (which for the solar wind data, 
# means the closest-to-contemporaneous shifted time element), up until the most
# recent measurement (which, after propagation, gives us the forecast span).

#Make a starting time for the temporal re-gridding based on the current time, 
# reduced to 1-min precision.
start_time = np.datetime64(current_time,'m')#scalar datetime64 object, at minute precision.
#For instance: current_time.astype(datetime.datetime).second > start_time.astype(datetime.datetime).second.

#Make an end time for the temporal re-gridding based on the last available 
# propagated measurement epoch, also reduced to 1-min precision. This applies 
# a 'floor' operation, ensuring that the range of the regular gridded series 
# is within the range of the existing data.
end_time = np.datetime64(GGF_times[-1][0],'m')#scalar datetime64 object, at minute precision.

#Synthesise a 1-min cadence time series using the predefined start and end 
# times. Note that this is set to microsecond ('us') precision, because it 
# needs to match the precision of the 'shifted_times' variable for when I 
# convert both of them to ordinal floats (which will have units of 
# microseconds) for the interpolation.
GGF_times_regular_grid = np.arange(start_time,end_time+np.timedelta64(1,'m'),np.timedelta64(1,'m'), dtype='datetime64[s]')#size [minutes in past day (ish) by 0].

#Interpolate the various solar wind measurement values to the regular temporal 
# grid, using the command '.astype(float)' to convert the datetime64 values to 
# ordinal microseconds.
GGF_IMF_Bx_regular_grid = np.interp(GGF_times_regular_grid.astype(float), GGF_times[~np.isnan(GGF_IMF_Bx),0].astype(float), GGF_IMF_Bx[~np.isnan(GGF_IMF_Bx)])#size [minutes in past day (ish) by 0].
GGF_IMF_By_regular_grid = np.interp(GGF_times_regular_grid.astype(float), GGF_times[~np.isnan(GGF_IMF_By),0].astype(float), GGF_IMF_By[~np.isnan(GGF_IMF_By)])#size [minutes in past day (ish) by 0].
GGF_IMF_Bz_regular_grid = np.interp(GGF_times_regular_grid.astype(float), GGF_times[~np.isnan(GGF_IMF_Bz),0].astype(float), GGF_IMF_Bz[~np.isnan(GGF_IMF_Bz)])#size [minutes in past day (ish) by 0].
GGF_sw_speed_regular_grid = np.interp(GGF_times_regular_grid.astype(float), GGF_times[~np.isnan(GGF_sw_speed[:,0]),0].astype(float), GGF_sw_speed[~np.isnan(GGF_sw_speed[:,0]),0])#size [minutes in past day (ish) by 0].
nan_indicator_series_regular_grid = np.interp(GGF_times_regular_grid.astype(float), GGF_times[:,0].astype(float), nan_indicator_series_original)#size [minutes in past day (ish) by 0].

#%% Compute local times at each BGS station, from the timestamps of the most-recent solar wind measurements.

#Define the three BGS station longitudes, taken from the INTERMAGNET daily file 
# metadata. In alphabetic order: Eskdalemuir, Hartland, Lerwick.
BGS_station_longitudes = [(356.8 - 360),(355.5 - 360),(358.8 - 360)]#list of size [3] -- geodetic longitude. Converted to east longitude on a +-180 scale (for later local time calculations), so it's negative because it's slightly west.

#Preallocate storage for BGS station local times.
BGS_stations_shifted_times_regular_grid_local_time_day_fraction = np.empty([np.shape(GGF_times_regular_grid)[0],3])#np array of floats, size [minutes in past day (ish) by 3].

#Loop over BGS stations.
for i_station in range(3):
    #Extract the longitude for this station.
    BGS_station_longitude = BGS_station_longitudes[i_station]#scalar float.
    
    #Compute the local time of this BGS station, rounded to 1-second precision, 
    # because numpy doesn't accept float inputs to timedelta64.
    BGS_single_station_shifted_times_regular_grid_with_local_time_alteration = GGF_times_regular_grid + np.timedelta64(round((BGS_station_longitude / 15) * 60 * 60),'s')#np array of datetime64[us], size [minutes in past day (ish) by 1] -- converts longitude to hours, then minutes, then seconds.
    
    #Loop over each time element, and convert the BGS station local time to 
    # ordinal fraction of the day.
    for i_time, time_element in enumerate(BGS_single_station_shifted_times_regular_grid_with_local_time_alteration):
        BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_time,i_station] = datetime.timedelta(
            hours=time_element.astype(object).hour,
            minutes=time_element.astype(object).minute,
            seconds=time_element.astype(object).second).total_seconds()/seconds_per_day
    #End loop over each time element of the recent solar wind data subset.
#End loop over the three BGS stations.

#%% Forecast the different GGF regression models for each epoch of the most-recent solar wind measurements.

#Preallocate storage for predictions.
model_predictions_all_stations_all_cmpnts_all_ensemble = np.empty([np.shape(GGF_IMF_Bx_regular_grid)[0],3,3,100])#np array of floats, size [minutes in past day (ish) by BGS stations by BGS components by model ensemble instances].
max_model_predictions_all_stations_all_cmpnts = np.empty([np.shape(GGF_IMF_Bx_regular_grid)[0],3,3])#np array of floats, size [minutes in past day (ish) by BGS stations by BGS components].
min_model_predictions_all_stations_all_cmpnts = np.empty([np.shape(GGF_IMF_Bx_regular_grid)[0],3,3])#np array of floats, size [minutes in past day (ish) by BGS stations by BGS components].
mean_model_predictions_all_stations_all_cmpnts = np.empty([np.shape(GGF_IMF_Bx_regular_grid)[0],3,3])#np array of floats, size [minutes in past day (ish) by BGS stations by BGS components].
model_prediction_flags = np.empty([np.shape(GGF_IMF_Bx_regular_grid)[0],1])#np array of floats, size [minutes in past day (ish) by 1].
#Loop over each temporal element in the prediction timespan.
for i_t in range(np.shape(GGF_times_regular_grid)[0]):
    #Choose which model you will use to predict this temporal element of the 
    # forecast span, dependent on what solar wind measurements are available.
    if np.isnan(GGF_IMF_Bx_regular_grid[i_t]) or np.isnan(GGF_sw_speed_regular_grid[i_t]):
        #If the magnetic field measurements do not exist for this epoch, we
        # forecast a null value, and assign a flag value indicating this.
        model_predictions_all_stations_all_cmpnts_all_ensemble[i_t,:,:,:] = np.nan
        model_prediction_flags[i_t] = 2
        
        #Skip the rest of this loop's iteration.
        continue#pertains to loop over i_t.
        
    elif np.invert(np.isnan(GGF_IMF_Bx_regular_grid[i_t])) & np.invert(np.isnan(GGF_sw_speed_regular_grid[i_t])):
        #If we have both magnetic field and plasma velocity values, then 
        # we can use the epsilon-driven model for the forecast.
        
        #Here, we flag if the model values came from an interpolated span of 
        # data, or if there were NaNs nearby in the pre-interpolation data.
        if nan_indicator_series_regular_grid[i_t] > 0.5:
            model_prediction_flags[i_t] = 1
        else:
            model_prediction_flags[i_t] = 0
        #End conditional: check whether this data point was interpolated over a NaN instance in the non-propagated series.
    #End conditional: choose forecast model based on what data are there to drive it.
    
    #Compute epsilon solar wind coupling function.
    #Tips from here on angles and quadrants
    # 'http://www.mathworks.co.uk/matlabcentral/newsreader/view_thread/167016'
    # 'atan' only returns angles in the range -pi/2 and +pi/2.  'atan2'
    # implicitly divides the two inputs and will produce angles between -pi and
    # +pi.  If you want to convert these to angles between 0 and 2*pi, run:
    # mod(angle,2*pi). Strictly speaking, the part inside the brackets
    # should each be multipled by rad, but the ratio is used so it would
    # have no effect.
    IMF_clock_angle = math.atan2(GGF_IMF_By_regular_grid[i_t], GGF_IMF_Bz_regular_grid[i_t]) * deg
    
    #Calculate magnitude of full IMF vector.
    IMF_B_magnitude = math.sqrt((GGF_IMF_Bx_regular_grid[i_t] ** 2) + (GGF_IMF_By_regular_grid[i_t] ** 2) + (GGF_IMF_Bz_regular_grid[i_t] ** 2))
    
    #Convert from nT to Tesla.
    IMF_B_magnitude_in_Tesla = IMF_B_magnitude * 1e-9  #Converted to T, from nT. #np array of floats, size [minutes in past day (ish) by 0].
    
    #Convert SWvx to SI units.
    speed_in_m_per_s = GGF_sw_speed_regular_grid[i_t] * 1e3  #Converted to m/s from km/s. #np array of floats, size [minutes in past day (ish) by 0].
    
    #Assign a length-scale factor of 7Re.
    l_nought = 7 * (6371.2 * 1000) #scalar float.
    
    #Assign a factor for the amplitude of the permeability of free space, once 
    # 4 pi has been divided by mew_nought.
    four_pi_over_mew_nought = 1e7 #Note: 1/1e-7 = 1e7. #scalar float.
    
    #Form the epsilon coupling function.
    sw_coupling_function = four_pi_over_mew_nought * speed_in_m_per_s * (IMF_B_magnitude_in_Tesla ** 2) * (math.sin((IMF_clock_angle / 2) * rad) ** 4) * (l_nought ** 2)
    #Units: Watts.
    
    #Compute the model prediction for each component of each station.
    
    #Loop over BGS stations.
    for i_station in range(3):
        #Based on the decimal day-fraction of this BGS station's local time,
        # find the fiducials of the two LT bins which flank this value. 
        # Note that we are basing the model coefficient interpolation on 
        # the LT bin centroids: it dose not matter which LT bin the station
        # is actually 'within'.
        if BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station] < LT_bin_centroids_day_fraction[0]:
            #Here, the BGS station is between midnight and the centroid LT
            # of the first bin. That is, the BGS LT is near zero.
            
            #The LT bin which sits behind the station in LT (i.e. the 
            # 'past' LT bin) needs to have one day subtracted from its 
            # decimal day LT value, such that the effective distance between 
            # it and the LT bin which sits ahead of the station in LT (i.e. 
            # the 'future' LT bin) is accurate.
            past_LT_bin_centroid = LT_bin_centroids_day_fraction[-1] - 1
            future_LT_bin_centroid = LT_bin_centroids_day_fraction[0]
            
            #Define indices for these LT bins: these can be specified 
            # manually due to the conditions required for these indented 
            # lines of code to be activated.
            index_past_LT_bin = 23
            index_future_LT_bin = 0
        elif BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station] > LT_bin_centroids_day_fraction[-1]:
            #Here, the BGS station is between the centroid LT of the last 
            # bin, and midnight. That is, the BGS LT is near 1.
            
            #The LT bin which sits ahead of the station in LT (i.e. the 
            # 'future' LT bin) needs to have one day added to its 
            # decimal day LT value, such that the effective distance between 
            # it and the LT bin which sits behind of the station in LT (i.e. 
            # the 'past' LT bin) is accurate.
            past_LT_bin_centroid = LT_bin_centroids_day_fraction[-1]
            future_LT_bin_centroid = LT_bin_centroids_day_fraction[0] + 1
            
            #Define indices for these LT bins: these can be specified 
            # manually due to the conditions required for these indented 
            # lines of code to be activated.
            index_past_LT_bin = 23
            index_future_LT_bin = 0
        else:
            #Here, the BGS station lies within some other pair of LT bins,
            # or directly upon one of them (we deal with that condition 
            # later).
            
            #Define indices for the flanking LT bins.
            index_past_LT_bin = np.nonzero(LT_bin_centroids_day_fraction <= BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station])[0][-1]
            index_future_LT_bin = np.nonzero(LT_bin_centroids_day_fraction >= BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station])[0][0]
            
            #Find the flanking LT bins centroid decimal day values.
            past_LT_bin_centroid = LT_bin_centroids_day_fraction[index_past_LT_bin]
            future_LT_bin_centroid = LT_bin_centroids_day_fraction[index_future_LT_bin]
        #End conditional: find the two LT bins which flank the station's precise location in LT.
            
        #Loop over BGS data components.
        for i_component in range(3):
            #Loop over model ensemble instances.
            for i_ensemble_instance in range(100):
                #Based on the station, the component, and the ensemble instance, 
                # extract model parameter vectors for each of the two 
                # interpolant LT bins, from the RTRv3p0 model coefficients.
                past_LT_bin_model_parameter_vector =   model_coefficients_all_stations_all_components_all_ensemble[i_station,index_past_LT_bin,i_component,:,i_ensemble_instance]#size [6 by 0]
                future_LT_bin_model_parameter_vector = model_coefficients_all_stations_all_components_all_ensemble[i_station,index_future_LT_bin,i_component,:,i_ensemble_instance]#size [6 by 0]
                
                #For each model coefficient, interpolate it to the central (ish) 
                # LT of the BGS station.
                interpolated_model_parameter_vector = np.empty([np.shape(model_coefficients_all_stations_all_components_all_ensemble)[3],1])#size [6 (model parameters) by 1].
                for i_parameter in range(np.shape(model_coefficients_all_stations_all_components_all_ensemble)[3]):
                    interpolated_model_parameter_vector[i_parameter,0] = np.interp(
                        BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station],
                        np.array([past_LT_bin_centroid[0],future_LT_bin_centroid[0]]),
                        np.array([past_LT_bin_model_parameter_vector[i_parameter],future_LT_bin_model_parameter_vector[i_parameter]]))
                    #End indenting for this variable.
                #End loop over model parameters.
                #Note that the above interpolator code will function properly even 
                # if past bin LT = future bin LT = current BGS LT.
                
                #Define the fractional day of year of this date, specific to this station.
                BGS_date_DOY = GGF_times_regular_grid[i_t].astype(datetime.datetime).timetuple().tm_yday \
                    + BGS_stations_shifted_times_regular_grid_local_time_day_fraction[i_t,i_station]#scalar float.
                #End indenting for this variable.
                
                #Create model data kernel vector.
                data_kernel = np.array([
                    1,
                    np.sin(((BGS_date_DOY - 79) * (2 * np.pi))/365.25),
                    np.cos(((BGS_date_DOY - 79) * (2 * np.pi))/365.25),
                    sw_coupling_function,
                    np.sin(((BGS_date_DOY - 79) * (2 * np.pi))/365.25) * sw_coupling_function,
                    np.cos(((BGS_date_DOY - 79) * (2 * np.pi))/365.25) * sw_coupling_function,
                ])#size [6 parameters by 0]
                #End indentation for this variable.
                
                #Make and store data prediction.
                model_predictions_all_stations_all_cmpnts_all_ensemble[i_t,i_station,i_component,i_ensemble_instance] = np.matmul(np.transpose(data_kernel[:,np.newaxis]),interpolated_model_parameter_vector)[0][0]#scalar float result.
            #End loop over model ensemble instances.
            
            #For this station and component, find the max, min and mean of the
            # ensemble predictions at this epoch.
            max_model_predictions_all_stations_all_cmpnts[i_t,i_station,i_component] = np.nanmax(model_predictions_all_stations_all_cmpnts_all_ensemble[i_t,i_station,i_component,:])
            min_model_predictions_all_stations_all_cmpnts[i_t,i_station,i_component] = np.nanmin(model_predictions_all_stations_all_cmpnts_all_ensemble[i_t,i_station,i_component,:])
            mean_model_predictions_all_stations_all_cmpnts[i_t,i_station,i_component] = np.nanmean(model_predictions_all_stations_all_cmpnts_all_ensemble[i_t,i_station,i_component,:])
        #End loop over BGS components.
    #End loop over BGS stations.
#End loop over each temporal element of the recent solar wind data subset.

#%% Forecast substorm onset likelihood in the next hour.

#Initiate some model options.
omn_pred_hist=120
omn_train_params=["Bx", "By", "Bz", "Vx", "Np"]

#Create a dataframe of the substorm input data.
substorm_forecast_data_df_part1 = pd.DataFrame(MO_API_plasma_times_bow_shock_nose_plus10min[:,0][:,np.newaxis],columns=['propagated_time_tag'])
substorm_forecast_data_df_part2 = pd.DataFrame(np.concatenate((MO_API_IMF_Bx[:,np.newaxis],MO_API_IMF_By[:,np.newaxis],
                                                               MO_API_IMF_Bz[:,np.newaxis],MO_API_sw_speed[:,np.newaxis],MO_API_sw_density[:,np.newaxis]),
                                                              axis=1),columns=['bx','by','bz','speed','density'])
substorm_forecast_data_df = pd.concat([substorm_forecast_data_df_part1, substorm_forecast_data_df_part2], axis=1)#size [minutes in past day (ish) by 7 columns].

#Convert timestamps to datetime.
substorm_forecast_data_df["propagated_time_tag"] = pd.to_datetime(substorm_forecast_data_df["propagated_time_tag"])

#Sort the data by the propagated time tags, since some propagated records will 
# appear to arrive at the bow shock nose in the wrong time order.
substorm_forecast_data_df.sort_values(by=["propagated_time_tag"], inplace=True)

#Interpolate the data to a regular cadence, using median averages to set 
# the propagated time tags to a 1-min step.
substorm_forecast_data_df.set_index("propagated_time_tag", inplace=True)
substorm_forecast_data_df = substorm_forecast_data_df.resample('1min').median()

#Linearly interpolate the solar wind data to the new (propagated) 1-min cadence.
substorm_forecast_data_df.interpolate(method='linear', axis=0, inplace=True)

#Load mean and std values of input features from a json file.
inp_mean_std_file_path = os.path.join(WORKDIR,'Storage_for_model_coefficients','substorm_model_solar_wind_weights_mean_std.json')

#Load the mean and std values into memory.
with open(inp_mean_std_file_path) as jf:
    params_mean_std_dct = json.load(jf)#dict object.
#End indenting for this load-in.

#Normalise the solar wind data to the mean and std values of input features 
# from the training data. Note that Bharat's code originally used the 'speed' 
# variable in the NOAA SWPC data stream, reversed its sign, and called it 'Vx',
#  so the code below is at least consistent with the trained model from Maimati et al.
substorm_forecast_data_df["Vx"] = ((-substorm_forecast_data_df["speed"]) - params_mean_std_dct["Vx_mean"]) / params_mean_std_dct["Vx_std"]
substorm_forecast_data_df["Np"] = (substorm_forecast_data_df["density"] - params_mean_std_dct["Np_mean"]) / params_mean_std_dct["Np_std"]
substorm_forecast_data_df["Bz"] = (substorm_forecast_data_df["bz"] - params_mean_std_dct["Bz_mean"]) / params_mean_std_dct["Bz_std"]
substorm_forecast_data_df["By"] = (substorm_forecast_data_df["by"] - params_mean_std_dct["By_mean"]) / params_mean_std_dct["By_std"]
substorm_forecast_data_df["Bx"] = (substorm_forecast_data_df["bx"] - params_mean_std_dct["Bx_mean"]) / params_mean_std_dct["Bx_std"]

#Define substorm forecast model filename.
model_name = os.path.join(WORKDIR,'Storage_for_model_coefficients','substorm_model_coefficient_weights.epoch_200.val_loss_0.58.val_accuracy_0.70.hdf5')

#Load the substorm forecast model.
substorm_onset_model = keras.models.load_model(model_name)#tensorflow.python.keras.engine.functional.Functional object.

#Define the temporal limits of the data to be ingested into the forward model. 
# These are the end epoch of the propagated timestamps, and 2 hours prior to 
# that epoch. the forecast ingestion span is based on propagated timestamps, 
# and the forecast pertains to the (non-propagated) contemporaneous epoch (i.e. now).
omn_end_time = substorm_forecast_data_df.index.max()
omn_begin_time = (omn_end_time - datetime.timedelta(minutes=omn_pred_hist)).strftime("%Y-%m-%d %H:%M:%S")

#Compute the substorm onset probability model prediction at this epoch.
inp_omn_vals = substorm_forecast_data_df.loc[omn_begin_time : omn_end_time][omn_train_params].values
inp_omn_vals = inp_omn_vals.reshape(1,inp_omn_vals.shape[0],inp_omn_vals.shape[1])
sson_pred_enc = substorm_onset_model.predict(inp_omn_vals, batch_size=1)
sson_pred_enc = sson_pred_enc[0].round(2)

#Extract the substorm onset probability from the model prediction.
substorm_onset_probability = sson_pred_enc[1]

#Make a series of this forecast value for the next hour, and NaN thereafter. 
# Start by making a substorm probability series of the length of the forecast 
# span.
substorm_onset_probability_series = np.ones(np.shape(GGF_times_regular_grid)) * substorm_onset_probability

#Determine the length of the forecast span: if it's more than an hour, we need 
# to replace the values over 60 min with NaNs. Otherwise, the substorm 
# probability value can fill the GGF forecast span (as defined above).
if np.timedelta64(GGF_times_regular_grid[-1] - GGF_times_regular_grid[0], 'm') / np.timedelta64(1, 'm') > 60:
    #In this case, the forecast span is more than one hour.
    
    #Find the index of the forecast span which is just over the one-hour-long 
    # mark. It occurs to me that this should return 61, or something's wrong.
    index_outside_probability_forecast_span = np.nonzero((GGF_times_regular_grid - GGF_times_regular_grid[0]) / np.timedelta64(1,'m') == 61)[0][0]
    
    #Set the values of the substorm onset probability forecast to be NaN 
    # outside of the valid forecast span of one hour.
    substorm_onset_probability_series[index_outside_probability_forecast_span:] = np.nan
#End conditional: different synthesis of substorm probability series, dependent on prediction span.

#%% Format and save out the forecast data file.

#Define filename for real-time geomagnetic forecast output.
real_time_solar_wind_data_output_filename = os.path.join(WORKDIR,'Temp_storage_for_output_GGF_forecast','real_time_magnetic_field_forecast_from_program_GGF_RTF_version_' + output_version_identifier + '.dat')

#Delete any existing data file of solar wind-based geomagnetic forecast outputs.
if os.path.isfile(real_time_solar_wind_data_output_filename):
    os.remove(real_time_solar_wind_data_output_filename)
#End:conditional: delete any already-existing output files, to avoid appending data to them.

#Open file of real time solar wind data.
file_id = open(real_time_solar_wind_data_output_filename, 'wb')

#Loop over each line of output.
for i_t in range(np.shape(mean_model_predictions_all_stations_all_cmpnts)[0]):
    #Format the date string.
    date_string = '{i_t_day:02d}-{i_t_month:02d}-{i_t_year}  {i_t_hour:02d}:{i_t_min:02d}'\
        .format(i_t_day = GGF_times_regular_grid[i_t].astype(datetime.datetime).day,
                i_t_month = GGF_times_regular_grid[i_t].astype(datetime.datetime).month,
                i_t_year = GGF_times_regular_grid[i_t].astype(datetime.datetime).year,
                i_t_hour = GGF_times_regular_grid[i_t].astype(datetime.datetime).hour,
                i_t_min = GGF_times_regular_grid[i_t].astype(datetime.datetime).minute)
    #End indenting for this string formatting.
    
    #Format the prediction string.
    data_string = '       {data_flag:1d}       {onset_probability:.2f}       ' + \
        '{esk_x:.2f}       {esk_y:.2f}       {esk_z:.2f}       ' + \
        '{had_x:.2f}       {had_y:.2f}       {had_z:.2f}       ' + \
        '{ler_x:.2f}       {ler_y:.2f}       {ler_z:.2f}       ' + \
        '{esk_x_min:.2f}       {esk_y_min:.2f}       {esk_z_min:.2f}       ' + \
        '{had_x_min:.2f}       {had_y_min:.2f}       {had_z_min:.2f}       ' + \
        '{ler_x_min:.2f}       {ler_y_min:.2f}       {ler_z_min:.2f}       ' + \
        '{esk_x_max:.2f}       {esk_y_max:.2f}       {esk_z_max:.2f}       ' + \
        '{had_x_max:.2f}       {had_y_max:.2f}       {had_z_max:.2f}       ' + \
        '{ler_x_max:.2f}       {ler_y_max:.2f}       {ler_z_max:.2f}'
    data_string = data_string.format(data_flag = model_prediction_flags[i_t,0].astype(int),
                                     onset_probability = substorm_onset_probability_series[i_t],
                                     esk_x = mean_model_predictions_all_stations_all_cmpnts[i_t,0,0],
                                     esk_y = mean_model_predictions_all_stations_all_cmpnts[i_t,0,1],
                                     esk_z = mean_model_predictions_all_stations_all_cmpnts[i_t,0,2],
                                     had_x = mean_model_predictions_all_stations_all_cmpnts[i_t,1,0],
                                     had_y = mean_model_predictions_all_stations_all_cmpnts[i_t,1,1],
                                     had_z = mean_model_predictions_all_stations_all_cmpnts[i_t,1,2],
                                     ler_x = mean_model_predictions_all_stations_all_cmpnts[i_t,2,0],
                                     ler_y = mean_model_predictions_all_stations_all_cmpnts[i_t,2,1],
                                     ler_z = mean_model_predictions_all_stations_all_cmpnts[i_t,2,2],
                                     esk_x_min = min_model_predictions_all_stations_all_cmpnts[i_t,0,0],
                                     esk_y_min = min_model_predictions_all_stations_all_cmpnts[i_t,0,1],
                                     esk_z_min = min_model_predictions_all_stations_all_cmpnts[i_t,0,2],
                                     had_x_min = min_model_predictions_all_stations_all_cmpnts[i_t,1,0],
                                     had_y_min = min_model_predictions_all_stations_all_cmpnts[i_t,1,1],
                                     had_z_min = min_model_predictions_all_stations_all_cmpnts[i_t,1,2],
                                     ler_x_min = min_model_predictions_all_stations_all_cmpnts[i_t,2,0],
                                     ler_y_min = min_model_predictions_all_stations_all_cmpnts[i_t,2,1],
                                     ler_z_min = min_model_predictions_all_stations_all_cmpnts[i_t,2,2],
                                     esk_x_max = max_model_predictions_all_stations_all_cmpnts[i_t,0,0],
                                     esk_y_max = max_model_predictions_all_stations_all_cmpnts[i_t,0,1],
                                     esk_z_max = max_model_predictions_all_stations_all_cmpnts[i_t,0,2],
                                     had_x_max = max_model_predictions_all_stations_all_cmpnts[i_t,1,0],
                                     had_y_max = max_model_predictions_all_stations_all_cmpnts[i_t,1,1],
                                     had_z_max = max_model_predictions_all_stations_all_cmpnts[i_t,1,2],
                                     ler_x_max = max_model_predictions_all_stations_all_cmpnts[i_t,2,0],
                                     ler_y_max = max_model_predictions_all_stations_all_cmpnts[i_t,2,1],
                                     ler_z_max = max_model_predictions_all_stations_all_cmpnts[i_t,2,2])
    #End indenting for this string formatting.

    #Collate formatted strings.
    line = date_string + data_string
    
    #Alter the string so that it has a consistent number of spaces between data elements.
    line = ' '.join(('%*s' % (10, i) for i in line.split()))
    #Solution from https://stackoverflow.com/questions/3685195/line-up-columns-of-numbers-print-output-in-table-format.
    
    #Append a carriage-return, and encode as ascii.
    ascii_output = (line + '\n').encode('ascii')
    
    #Write out this line.
    file_id.write(ascii_output)
#End loop over each time element of the forecasts.

#Close the output ascii file.
file_id.close()

#%% State program status.

print('GGF_RTF.py run successfully completed.')
