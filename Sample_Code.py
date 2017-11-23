from project_functions import Create_Feature_Matrix

directory = "/Users/anjian/Dropbox (MIT)/Courses/Fall 2017/6.439/Class Project/Project Data/"

match_data_file_location = "/Users/anjian/Dropbox (MIT)/Courses/Fall 2017/6.439/Class Project/Project Data/all_match_outcomes.csv" 
Create_Feature_Matrix(directory, 
                      match_data_file_location = match_data_file_location,
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
                                              'cohesion sans 1'])