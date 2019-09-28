import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ATP_functions import *


# read csv of all matches and store into dataframe
all_matches_df = pd.read_csv("atp_matches.csv", low_memory=False, index_col=0)

# initialize some useful variables
discarded_columns = ["draw_size", "winner_id", "winner_entry", "loser_id", "loser_entry", "tourney_id"]
some_statistics = ["minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpFaced", "w_bpSaved",
 "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpFaced", "l_bpSaved"]
change_dtype = ["l_bpFaced", "w_bpFaced", "w_bpSaved", "l_bpSaved"]
list_of_relevant_stats = ["w_ace", "w_df", "w_SvGms", 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 
"Total_Aces", "Total_Break_Points", "Total_bpConverted", "w_1stsv_pts_won_rate", "l_1stsv_pts_won_rate", "w_2ndsv_pts_won_rate", "l_2ndsv_pts_won_rate"]
win_statistics_we_care_about = ["w_ace", "w_df", "w_bpConverted", "wl_diff_bpConverted", "w_1stsv_pts_won_rate", "w_2ndsv_pts_won_rate"]
lose_statistics_we_care_about = ["l_ace", "l_df", "l_bpConverted", "wl_diff_bpConverted", "l_1stsv_pts_won_rate", "l_2ndsv_pts_won_rate"]
statistics_we_care_about = list_of_relevant_stats

### Clean up data ###

# format the tourney date and clean up name
all_matches_df["tourney_date"] = [str(date)[0:4] + "-" + str(date)[4:6] + "-" + str(date)[6:8] for date in all_matches_df["tourney_date"]]
all_matches_df["tourney_name"] = all_matches_df["tourney_name"].replace(["Us Open"], "US Open")

# replace surface names with numbers for classification reasons 
dict_to_replace_surface = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
all_matches_df["surface"].replace(dict_to_replace_surface, inplace=True)

# drop data we don't need/want
all_matches_df.drop(discarded_columns, axis=1, inplace=True)
all_matches_df = all_matches_df.dropna(subset=some_statistics)

# Change data type of certain columns
for column in change_dtype:
	all_matches_df[column] = pd.to_numeric(all_matches_df[column], errors="coerce")


# Create some useful features
all_matches_df["w_bpConverted"] = (all_matches_df["l_bpFaced"] - all_matches_df["l_bpSaved"]) / all_matches_df["l_bpFaced"]
all_matches_df["w_bpConverted"].fillna(0, inplace=True) # Loser didn't face a break point
all_matches_df["l_bpConverted"] = (all_matches_df["w_bpFaced"] - all_matches_df["w_bpSaved"]) / all_matches_df["w_bpFaced"]
all_matches_df["l_bpConverted"].fillna(0, inplace=True) # Winner didn't face a break point
all_matches_df["wl_diff_bpConverted"] = all_matches_df["w_bpConverted"] - all_matches_df["l_bpConverted"]
all_matches_df["Total_Aces"] = all_matches_df["w_ace"] + all_matches_df["l_ace"]
all_matches_df["Total_Break_Points"] = all_matches_df["l_bpFaced"] + all_matches_df["w_bpFaced"]
all_matches_df["Total_bpConverted"] = all_matches_df["w_bpConverted"] + all_matches_df["l_bpConverted"]
all_matches_df["w_1stsv_pts_won_rate"] = all_matches_df["w_1stWon"] / all_matches_df["w_1stIn"]
all_matches_df["l_1stsv_pts_won_rate"] = all_matches_df["l_1stWon"] / all_matches_df["l_1stIn"]
all_matches_df["w_2ndsv_pts_won_rate"] = all_matches_df["w_2ndWon"] / (all_matches_df["w_svpt"] - all_matches_df["w_1stIn"])
all_matches_df["l_2ndsv_pts_won_rate"] = all_matches_df["l_2ndWon"] / (all_matches_df["l_svpt"] - all_matches_df["l_1stIn"]) 
all_matches_df["Latitude"] = np.nan
all_matches_df["Longitude"] = np.nan

# normalize the relevant data
# COMMENT THIS TO VIEW UNNORMED DATA ANALYSIS
all_matches_df = normalize_data(all_matches_df, list_of_relevant_stats)


print(all_matches_df)


## Get various statistics by surface and create dataframe of it
surface_aces_mean = all_matches_df.groupby("surface")["Total_Aces"].mean()
surface_bp_mean = all_matches_df.groupby("surface")["Total_Break_Points"].mean()
surface_bpConverted_mean = all_matches_df.groupby("surface")["Total_bpConverted"].mean()

# Separate by bo3 and bo5 matches
best_of_3_matches = all_matches_df[all_matches_df["best_of"] == 3]
best_of_5_matches = all_matches_df[all_matches_df["best_of"] == 5]
surface_minutes_3_mean = best_of_3_matches.groupby("surface")["minutes"].mean() 
surface_minutes_5_mean = best_of_5_matches.groupby("surface")["minutes"].mean()

stats_by_surface_dict = {"Aces": surface_aces_mean, "Break Points": surface_bp_mean, 
	"Hours (Best of 3)": surface_minutes_3_mean/60, "Hours (Best of 5)": surface_minutes_5_mean/60}

stats_by_surface_df = pd.DataFrame({"Aces": surface_aces_mean, "Break Points": surface_bp_mean, 
	"Hours (Best of 3)": surface_minutes_3_mean/60, "Hours (Best of 5)": surface_minutes_5_mean/60})

plot_bar_dictionary(stats_by_surface_dict, ("Hard", "Clay", "Grass", "Carpet"), "Surface", "Count", "Match Stats by Surface")


# GOAT Stuff

# dataframe for big 3 wins/loses and non big 3 wins/loses for comparisons
all_matches_df_goat_win  = all_matches_df[(all_matches_df.winner_name == "Roger Federer") | (all_matches_df.winner_name == "Rafael Nadal") | 
           (all_matches_df.winner_name == "Novak Djokovic")]	
all_matches_df_non_goat_win = pd.concat([all_matches_df, all_matches_df_goat_win, all_matches_df_goat_win]).drop_duplicates(keep=False)
all_matches_df_goat_lose = all_matches_df[(all_matches_df.loser_name == "Roger Federer") | (all_matches_df.loser_name == "Rafael Nadal") | 
           (all_matches_df.loser_name == "Novak Djokovic")]
all_matches_df_non_goat_lose = pd.concat([all_matches_df, all_matches_df_goat_lose, all_matches_df_goat_lose]).drop_duplicates(keep=False)
all_matches_df_goat = pd.concat([all_matches_df_goat_lose, all_matches_df_goat_win])
all_matches_df_non_goat = pd.concat([all_matches_df, all_matches_df_goat, all_matches_df_goat]).drop_duplicates(keep=False)


# dataframe for big 3 win and losses (separate)
all_matches_df_djokovic_won = matches_with_person(all_matches_df, "Novak Djokovic")
all_matches_df_federer_won = matches_with_person(all_matches_df, "Roger Federer")
all_matches_df_nadal_won = matches_with_person(all_matches_df, "Rafael Nadal")
all_matches_df_djokovic_lost = matches_with_person(all_matches_df, "Novak Djokovic", win=False)
all_matches_df_federer_lost = matches_with_person(all_matches_df, "Roger Federer", win=False)
all_matches_df_nadal_lost = matches_with_person(all_matches_df, "Rafael Nadal", win=False)



# comparing big 3 with others on certain statistics
djokovic_win_statistics = all_matches_df_djokovic_won[win_statistics_we_care_about].dropna()
nadal_win_statistics = all_matches_df_nadal_won[win_statistics_we_care_about].dropna()
federer_win_statistics = all_matches_df_federer_won[win_statistics_we_care_about].dropna()
all_other_win_statistics = all_matches_df_non_goat_win[win_statistics_we_care_about]
djokovic_lose_statistics = all_matches_df_djokovic_lost[lose_statistics_we_care_about].dropna()
nadal_lose_statistics = all_matches_df_nadal_lost[lose_statistics_we_care_about].dropna()
federer_lose_statistics = all_matches_df_federer_lost[lose_statistics_we_care_about].dropna()
all_other_lose_statistics = all_matches_df_non_goat_lose[lose_statistics_we_care_about]


statistics_df_big3_else_win = pd.concat([pd.DataFrame(djokovic_win_statistics.mean()).T, pd.DataFrame(federer_win_statistics.mean()).T, 
	pd.DataFrame(nadal_win_statistics.mean()).T, pd.DataFrame(all_other_win_statistics.mean()).T])
statistics_df_big3_else_win.index = ["Djokovic", "Federer", "Nadal", "All Others"]

statistics_df_big3_else_lose = pd.concat([pd.DataFrame(djokovic_lose_statistics.mean()).T, pd.DataFrame(federer_lose_statistics.mean()).T, 
	pd.DataFrame(nadal_lose_statistics.mean()).T, pd.DataFrame(all_other_lose_statistics.mean()).T])
statistics_df_big3_else_lose.index = ["Djokovic", "Federer", "Nadal", "All Others"]





list_of_statistics_big3_else_win = [list(statistics_df_big3_else_win.loc["Djokovic"]), list(statistics_df_big3_else_win.loc["Nadal"]), list(statistics_df_big3_else_win.loc["Federer"]), 
	list(statistics_df_big3_else_win.loc["All Others"])]

list_of_statistics_big3_else_lose = [list(statistics_df_big3_else_lose.loc["Djokovic"]), list(statistics_df_big3_else_lose.loc["Nadal"]), list(statistics_df_big3_else_lose.loc["Federer"]), 
	list(statistics_df_big3_else_lose.loc["All Others"])]

people_to_plot = ["Djokovic", "Nadal", "Federer", "All Others"]
stats_to_plot = ["Aces", "Double Faults", "BP Converted", "BP Converted Difference", "1st Serve Points Won (%)", "2nd Serve Points Won (%)"]

plot_heat_stats(people_to_plot, stats_to_plot, list_of_statistics_big3_else_win)
plot_heat_stats(people_to_plot, stats_to_plot, list_of_statistics_big3_else_lose)



## Analysis of Nadal on Clay for future analysis

# all_matches_df_nadal = pd.concat([all_matches_df_nadal_won, all_matches_df_nadal_lost])
# all_matches_df_federer = pd.concat([all_matches_df_federer_won, all_matches_df_federer_lost])
# all_matches_df_djokovic = pd.concat([all_matches_df_djokovic_won, all_matches_df_djokovic_lost])

# all_matches_df_nadal_clay = all_matches_df_nadal[all_matches_df_nadal["surface"] == 1]
# all_matches_df_federer_clay = all_matches_df_federer[all_matches_df_federer["surface"] == 1]
# all_matches_df_djokovic_clay = all_matches_df_djokovic[all_matches_df_djokovic["surface"] == 1]
# all_matches_df_non_goat_clay = all_matches_df_non_goat[all_matches_df_non_goat["surface"] == 1]

# all_matches_df_nadal_won_clay = all_matches_df_nadal_won[all_matches_df_nadal_won["surface"] == 1]
# all_matches_df_djokovic_won_clay = all_matches_df_djokovic_won[all_matches_df_djokovic_won["surface"] == 1]
# all_matches_df_federer_won_clay = all_matches_df_federer_won[all_matches_df_federer_won["surface"] == 1]
# all_matches_df_nadal_won_clay = all_matches_df_nadal_won[all_matches_df_nadal_won["surface"] == 1]



# # Get all grand slam rows and add latitude, longitude for future analysis
# grand_slams_df = all_matches_df[all_matches_df["tourney_level"] == "G"]
# list_of_grand_slams = grand_slams_df["tourney_name"].value_counts().index.to_list()
# statistics_loc_grand_slam_df = all_matches_df[all_matches_df["tourney_name"].isin(list_of_grand_slams)].groupby("tourney_name").mean()\
# [["Total_Aces", "Total_bpConverted", "Latitude", "Longitude"]]
# list_of_latitudes = [-37.8197, 48.8416, 51.4183, 40.7432]
# list_of_longitudes = [144.9737, 2.2509, -0.2206, -73.8410]

# for i in range(len(list_of_grand_slams)):
# 	all_matches_df.loc[all_matches_df["tourney_name"] == list_of_grand_slams[i], "Latitude"] = list_of_latitudes[i]
# 	all_matches_df.loc[all_matches_df["tourney_name"] == list_of_grand_slams[i], "Longitude"] = list_of_longitudes[i]



# statistics_with_loc = statistics_we_care_about.copy()
# statistics_with_loc.extend(["Latitude", "Longitude"])
