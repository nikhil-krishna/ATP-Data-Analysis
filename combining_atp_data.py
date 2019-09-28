import pandas as pd
import numpy as np 

# function to read the data from the folder
# input - pathToFolder, start year, end year
# output - dataframe with all years' matches
def read_atp_data(pathToFolder, start=2000, end=2017):

	# go through years and add dataframes to list
	df_list = []
	for i in range(start, end+1):
		df_list.append(pd.read_csv(pathToFolder + "/atp_matches_" + str(i) + ".csv"))

	# concatenate and return
	all_matches_df = pd.concat(df_list)

	return all_matches_df


# read in data and export data to csv
read_atp_data("/Users/nikhil/Desktop/atp-matches-dataset").to_csv("atp_matches.csv")




