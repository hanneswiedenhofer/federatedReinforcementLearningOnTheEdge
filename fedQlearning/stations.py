from more_itertools import unique_everseen
with open('possible_station_3.csv','r') as f, open('possible_stations_3.csv','w') as out_file:
    out_file.writelines(unique_everseen(f))   