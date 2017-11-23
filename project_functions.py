import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
#import glob


def Create_Feature_Matrix(directory, 
                          match_data_file_location = './all_match_outcomes.csv',
                          years = [2014,2010,2006,2002],
                          features_to_consider = ["Matches Played",
                                                  'Yellow_Per_Game_Avg', 
                                                  'YellowRed_Per_Game_Avg',
                                                  'Red_Per_Game_Avg', 
                                                  'Goal_Per_Game_Avg', 
                                                  'Goal_Against_Per_Game_Avg',
                                                  'PenGoal_Per_Game_Avg',
                                                  'FIFA rank',
                                                  'seed',
                                                  'host',
                                                  'stars',
                                                  'cohesion', 
                                                  'dist', 
                                                  'cohesion sans 1']):
    
    feature_matrix = pd.DataFrame()
    match_data =  pd.DataFrame.from_csv(match_data_file_location)
    for y in range(len(years)-1):
        current_year = years[y]
        last_year = years[y+1]

        #Grab Last World Cup's Performance Metrics of Country 1
        perf_data = pd.DataFrame.from_csv(directory+"%s.csv"%last_year)
        perf_data = perf_data[features_to_consider]
        perf_data.columns = [x+"_C1" for x in perf_data.columns]
        perf_data = perf_data.reset_index()
        perf_data = perf_data.rename(columns = {'index':'Country1'})

        features_of_this_year = match_data[match_data['Year'] == current_year]

        #print(len(features_of_this_year))

        features_of_this_year = features_of_this_year.merge(perf_data, how='left', on="Country1")

        #Grab Last World Cup's Performance Metrics of Country 2

        perf_data = pd.DataFrame.from_csv(directory+"%s.csv"%last_year)
        perf_data = perf_data[features_to_consider]
        perf_data.columns = [x+"_C2" for x in perf_data.columns]
        perf_data = perf_data.reset_index()
        perf_data = perf_data.rename(columns = {'index':'Country2'})

        features_of_this_year = features_of_this_year.merge(perf_data, how='left', on="Country2")

        #Combine to Feature Matrix
        feature_matrix = pd.concat([feature_matrix,features_of_this_year], axis=0, ignore_index=True)
        
    return feature_matrix

