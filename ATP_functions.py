import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

 

# function to normalize a dataframe's columns with a scaler
# input - a dataframe, columns needing scaling, a scaler
# output - a copy of the dataframe with certain columns normalized
def normalize_data(dataframe, columns, scaler=StandardScaler()):

	# scale a copy of the columns needed
	df_to_norm = dataframe[columns]
	df = dataframe.copy()
	df[columns] = scaler.fit_transform(df_to_norm)
	return df

# function to plot a dictionary with a bar graph
# input - dictionary, x tick labels, xlabel, ylabel, title
# output - a bar graph of the dictionary
def plot_bar_dictionary(dictToPlot, xticks, xlabel, ylabel, title):

	# create a df from the dictionary and plot
	df = pd.DataFrame(dictToPlot)
	df.plot(kind="bar")

	# add features to plot
	plt.xticks(np.arange(len(dictToPlot)), xticks, rotation=0)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

# function that returns matches where name is playing
# input - dataframe to search, name, win (is name the winner)
# output - dataframe with the person's matches
def matches_with_person(dataframe, name, win=True):

	# Check if winner else get loser
	if win:
		df = dataframe[(dataframe.winner_name == name)] 
	else:
		df = dataframe[(dataframe.loser_name == name)]

	return df

# function that shows a heat map of certain stats and people given data
# input - list of people and stats to plot and data in a 2-d array
# output - a plot described above
def plot_heat_stats(people, stats, data):

	# get attributes of plot and change them
	fig, ax  = plt.subplots()

	plt.imshow(data)
	ax.set_xticks(np.arange(len(stats)))
	ax.set_xticklabels(stats, rotation=0, fontsize=8)
	ax.set_yticks(np.arange(len(people)))
	ax.set_yticklabels(people, rotation=0)

	# put labels on each square
	for i in range(len(people)):
		for j in range(len(stats)):
			text = ax.text(j, i, round(data[i][j], 4), 
				ha="center", va="center")

	plt.colorbar()
	plt.show()







