from os import stat
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools 
from clustering import clustering
import random

#https://github.com/ianxxiao/reinforcement_learning_project/blob/master/code/helper.py

def user_input(num_stations, num_sessions, fixed_stations, expected_balances_only, create_new_OD_matrix, num_clusters, active_cluster, stock_mode):
    episode_list = [eps for eps in range (100,(num_sessions*100)+100,100)]#(100, 2000, 100)]
    #[497, 236, 3140, 470, 513, 412, 3256, 479, 450, 409]
    #ID = 497
    cluster = clustering(n_clusters=num_clusters, new_file=create_new_OD_matrix)
    clusters = cluster.cluster(expected_balances_only = expected_balances_only)
    #print(f"CLUSTERS: {clusters}")
    active_cluster = clusters.get(active_cluster)
    #print(f"active_cluster: {active_cluster}")
    if(stock_mode == "actual"):
        station_history, IDs = process_citibike(starting_bal=20, cluster=active_cluster, num_stations=num_stations, fixed_stations = fixed_stations)
    else:
        _, IDs = process_citibike(starting_bal=20, cluster=active_cluster, num_stations=num_stations, fixed_stations = fixed_stations)
        station_history = None
    #station_history = dict(itertools.islice(station_history.items(), num_stations))
    #print(f"STATION_HISTORY: {station_history}")
    #IDs = IDs[:num_stations]
    #print(f"Length IDs: {len(IDs)}")
    #random.shuffle(IDs)
    return episode_list, IDs, station_history

#def citi_data_processing():
#    #start with 20 bikes in stock
#    station_history, IDs = process_citibike(20)
#    return station_history, IDs

def process_citibike(starting_bal, cluster, num_stations, fixed_stations):
    bike = pd.read_csv("../data/201709-citibike-tripdata.csv")
    bike['starttime'] = pd.to_datetime(bike['starttime'], infer_datetime_format= True)
    bike['stoptime'] = pd.to_datetime(bike['stoptime'], infer_datetime_format= True)
        
    bike['day'] = bike['starttime'].dt.day
    bike['start_hour'] = bike['starttime'].dt.hour
    bike['end_hour'] = bike['stoptime'].dt.hour
    bike['DOW'] = bike['starttime'].dt.dayofweek
        
    # Create a dataset with all unique station id, name, and lat/lon
        
    uni_dep_stations = bike[['start station id', 'start station name', 
                         'start station latitude', 'start station longitude']].drop_duplicates()

    uni_arv_stations = bike[['end station id', 'end station name', 
                                 'end station latitude', 'end station longitude']].drop_duplicates()
        
    uni_dep_stations.columns = ["id", "name", "lat", "lon"]
    uni_arv_stations.columns = ["id", "name", "lat", "lon"]
    uni_station = pd.concat([uni_dep_stations, uni_arv_stations], axis = 0).drop_duplicates()
    #####================FILTER=============####
    #if len(fixed_stations) > 0:
    #    uni_station = uni_station[uni_station["id"].isin(fixed_stations)]
    #else:
    #    uni_station = uni_station[uni_station["id"].isin(random.sample(cluster, num_stations))]


    # Create hourly departure count by day across the month
    # Create hourly departure count by day across the month
    print("Calculating Departure and Arrivals ...")
        
    monthDep = pd.pivot_table(bike[['start station id', 'day','start_hour', 'starttime']],
                                     index = "start station id", columns = ['day', "start_hour"], 
                                     aggfunc = np.size, fill_value= 0).reset_index()
        
    monthDep.columns = ["dep_" + str(day) + "_" + str(hour) for _, day, hour in monthDep.columns]
        
        
    # Create hourly arrival count by day across the month

    monthArv = pd.pivot_table(bike[['end station id', 'day','end_hour', 'stoptime']],
                                     index = "end station id", columns = ['day', "end_hour"], 
                                     aggfunc = np.size, fill_value= 0).reset_index()
        
    monthArv.columns = ["arv_" + str(day) + "_" + str(hour) for _, day, hour in monthArv.columns]
        
    # Create a hourly net flow count by day across the month 

    monthNet = uni_station.merge(monthDep, how = "left", left_on = "id", right_on = "dep__").\
                              merge(monthArv, how = "left", left_on = "id", right_on = "arv__").fillna(0)
        
    for day in range(1, 31):
                
        for hour in range(0, 24):
                
            try:
                net_col = "net_" + str(day) + "_" + str(hour)
                dep_col = "dep_" + str(day) + "_" + str(hour)
                arv_col = "arv_" + str(day) + "_" + str(hour)
                monthNet[net_col] = monthNet[arv_col] - monthNet[dep_col]
            except (KeyError):
                print("Missing day: {} | Missing hour: {}".format(day, hour))
                pass
        
    # Create a dataframe of bike stock amount based on starting balance
    #print(f"monthNet: {monthNet}")
    df_citibike = calHourlyBal(monthNet, starting_bal)
    station_history = defaultdict(dict)
    IDs = []
    my_sample = random.sample(cluster,num_stations)
    for (index, id) in df_citibike["id"].iteritems():
        if len(fixed_stations) > 0:
            if id in fixed_stations:
                station_history[str(id)] = list(np.array(df_citibike[df_citibike['id'] == id])[0][4:28])
                IDs.append(id)
        else:
            if id in my_sample:
                station_history[str(id)] = list(np.array(df_citibike[df_citibike['id'] == id])[0][4:28])
                IDs.append(id)
            #print(f"len(station_history): {len(station_history)}")
    return  station_history, IDs


def calHourlyBal(df, starting_bal):
        
    print("Calculating Hourly Bike Stock for Each Station ...")
    hourBal = df
        
    # Calculate hourly bike balance based on starting stock
    for day in range(1, 31):
        for hour in range(0, 24):
            try:
                    
                if day == 1 and hour == 0:
                    bal_col = "bal_1_0"
                    hourBal["bal_1_0"] = starting_bal
                        
                elif day > 1 and hour == 0:
                        
                    bal_col = "bal_" + str(day) + "_" + str(hour)
                    last_bal_col = "bal_" + str(day-1) + "_23"
                    net_col = "net_" + str(day) + "_0"
                        
                    hourBal[bal_col] = hourBal[last_bal_col] + hourBal[net_col]
                    
                else:
                        
                    bal_col = "bal_" + str(day) + "_" + str(hour)
                    last_bal_col = "bal_" + str(day) + "_" + str(hour-1)
                    net_col = "net_" + str(day) + "_" + str(hour)
                                            
                    hourBal[bal_col] = hourBal[last_bal_col] + hourBal[net_col]
                
            except (KeyError) as ex:
                # use previous balance for missing time slot
                print("Missing net flow at day {} hour {}".format(day, hour))
                    
                #hourBal[bal_col] = hourBal[last_bal_col]
                pass
        
    # Only keep balance and change columns
    bal_col = hourBal.columns[hourBal.columns.str.contains("bal_")]
    hourBal[bal_col] = hourBal[bal_col].astype('int')
    #print(f"hourBal: {hourBal}")
    final_bal = pd.concat([hourBal[["id", "name", "lat", "lon"]], hourBal[bal_col]], axis = 1) 
        
    return final_bal