import pandas as pd
import numpy as np
from collections import defaultdict
from more_itertools import unique_everseen
import json

with open('EXPECTED_BALANCES.json') as json_data:
    expected_balance = json.load(json_data)

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


def check_sum(my_list, lower_lim, upper_lim):
    res = True
    for ele in my_list:
        if ele <= lower_lim or ele >= upper_lim:
            res = False 
            break
    return res

def create_2():
    fname = "possible_station_2.csv"
    with open(fname, 'w') as f:
        i = 0
        for key in expected_balance.keys():
            print(f"i: {i}")
            i += 1
            for key2 in expected_balance.keys():
                #print("1")
                #print("2")
                #print(f"{key},{key2},{key3},{key4}")
                if (key2 != key):
                    h1 = list(np.array(df_citibike[df_citibike['id'] == int(key)])[0][4:28])
                    h2 = list(np.array(df_citibike[df_citibike['id'] == int(key2)])[0][4:28])
                    #print("4")
                    my_sum = [x + y for x, y in zip(h1, h2)]
                    #print("5")
                    if check_sum(my_sum, 0, 100):
                        #print("6")
                        my_list = []
                        my_list.append(key)
                        my_list.append(key2)
                        my_string = ""
                        #print("7")
                        for el in sorted(my_list):
                            my_string += str(el)+","
                            #print(f"my_string: {my_string}")
                        #if(key < key2):
                        f.write(f"{my_string}\n")
                        #else:
                        #    f.write(f"{key2},{key},{key3}\n")
                                
    with open('possible_station_2.csv','r') as f, open('possible_stations_2.csv','w') as out_file:
        out_file.writelines(unique_everseen(f))   

def create_3():
    fname = "possible_station_3.csv"
    with open(fname, 'w') as f:
        i = 0
        for key in expected_balance.keys():
            print(f"i: {i}")
            i += 1
            for key2 in expected_balance.keys():
                #print("1")
                for key3 in expected_balance.keys():
                    #print("2")
                    #print(f"{key},{key2},{key3},{key4}")
                    if (key2 != key) & (key3 != key) & (key3 != key2):
                        h1 = list(np.array(df_citibike[df_citibike['id'] == int(key)])[0][4:28])
                        h2 = list(np.array(df_citibike[df_citibike['id'] == int(key2)])[0][4:28])
                        h3 = list(np.array(df_citibike[df_citibike['id'] == int(key3)])[0][4:28])
                        #print("4")
                        my_sum = [x + y + z for x, y, z in zip(h1, h2, h3)]
                        #print("5")
                        if check_sum(my_sum, 0, 150):
                            #print("6")
                            my_list = []
                            my_list.append(key)
                            my_list.append(key2)
                            my_list.append(key3)
                            my_string = ""
                            #print("7")
                            for el in sorted(my_list):
                                my_string += str(el)+","
                                #print(f"my_string: {my_string}")
                            #if(key < key2):
                            f.write(f"{my_string}\n")
                            #else:
                            #    f.write(f"{key2},{key},{key3}\n")
                                
    with open('possible_station_3.csv','r') as f, open('possible_stations_3.csv','w') as out_file:
        out_file.writelines(unique_everseen(f))               






def create_5():
    fname = "possible_station_5.csv"
    with open(fname, 'w') as f:
        i = 0
        for key in expected_balance.keys():
            print(f"i: {i}")
            i += 1
            for key2 in expected_balance.keys():
                l = 0
                for key3 in expected_balance.keys():
                    k = 0
                    for key4 in expected_balance.keys():
                        j = 0
                        for key5 in expected_balance.keys():
                            #print("2")
                            #print(f"{key},{key2},{key3},{key4}")
                            if (key2 != key) & (key3 != key) & (key3 != key2) & (key4 != key) & (key4 != key2) & (key4 != key3) & (key5 != key) & (key5 != key2) & (key5 != key3) & (key5 != key4):
                                h1 = list(np.array(df_citibike[df_citibike['id'] == int(key)])[0][4:28])
                                h2 = list(np.array(df_citibike[df_citibike['id'] == int(key2)])[0][4:28])
                                h3 = list(np.array(df_citibike[df_citibike['id'] == int(key3)])[0][4:28])
                                h4 = list(np.array(df_citibike[df_citibike['id'] == int(key4)])[0][4:28])
                                h5 = list(np.array(df_citibike[df_citibike['id'] == int(key5)])[0][4:28])
                                #print("4")
                                my_sum = [x + y + z + a + b for x, y, z, a, b in zip(h1, h2, h3, h4, h5)]
                                #print("5")
                                if check_sum(my_sum, 0, 250):
                                    j += 1
                                    k += 1
                                    l += 1
                                    my_list = []
                                    my_list.append(key)
                                    my_list.append(key2)
                                    my_list.append(key3)
                                    my_list.append(key4)
                                    my_list.append(key5)
                                    my_string = ""
                                    #print("7")
                                    for el in sorted(my_list):
                                        my_string += str(el)+","
                                        #print(f"my_string: {my_string}")
                                    #if(key < key2):
                                    f.write(f"{my_string}\n")
                                    #else:
                                    #    f.write(f"{key2},{key},{key3}\n")
                                if j > 2:
                                    break
                        if k > 4:
                            break
                if l > 8:
                    break      
                
                        
                            
                                
    with open('possible_station_5.csv','r') as f, open('possible_stations_5.csv','w') as out_file:
        out_file.writelines(unique_everseen(f))  

starting_bal = 20
bike = pd.read_csv("data/201709-citibike-tripdata.csv")
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
df_citibike = calHourlyBal(monthNet, starting_bal)
station_history = defaultdict(dict)
print("starting")
# create_2()
# print("2 done")
# create_3()
# print("3 done")
create_5()
print("5 done")
             



