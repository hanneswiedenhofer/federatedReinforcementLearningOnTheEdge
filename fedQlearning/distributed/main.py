from training import trainer
import helper
import time
import sys

#https://github.com/ianxxiao/reinforcement_learning_project/blob/master/code/main.py

if __name__ == "__main__":
    ##=============PARAMETERS===================##
    num_stations = 2 #number of stations to simulate
    num_sessions = 10 #number of sessions (more sessions -> higher number of episodes)
    action_space = [1,3,5,10] #possible number of bicycles to move at a time
    upper_limit = 50 #defines the limits
    lower_limit = 0
    fixed_stations = [324,3354,509]#[[402,433,515],[351,428,523],[3161,351,406],[2006,496,509],[257,3463,379]]#[508, 459] #only use these fixed stations#[238, 388, 482] #[514,482,284]#[253,453,377] #only use these fixed stations
    #fixed_stations.reverse()
    use_expected_balances_only = True #only use stations for which there are predictions 
    create_new_OD_matrix = False #start a new clustering or use the old one
    #if int(sys.argv[1]) < 1:
    forecasting = False
    #else:
    #    forecasting = True
    #forecasting = True #take into consideration the expected balances
    forecast_hours = 1#int(sys.argv[1]) #how many predicted hours should be taken into consideration
    num_clusters = 5 #number of clusters to create
    active_cluster = 4  #which cluster to randomly choose the stations from
    remove_only = False #only remove bikes or reassign them to other stations
    stock_mode = "actual" #linear: a linear increasing bike stock with 3 additional bikes per hour
                          #random: a linear increasing bike stock with random fluctuation
                          #actual: pick traffic from the citibike stations
    assign_mode = "extremes" #random: assign the bikes from one station to a random station
                             #extremes: assign the bikes from one station to the station with the smallest stock and vice versa
                             #forecast: like extremes, but with forecasts (uses previously defined variable forecast_hours)
    ##==========================================##

    ##==================PARAMETER CHECK==========================##
    if (forecasting) & (forecast_hours < 1):
        print("forecast_hours must be greater than 0")
        sys.exit(0)
    if assign_mode == "forecast":
        if not forecasting:
            print("if assign_mode = forecast, then forecasting must be true")
            sys.exit(0)
    if (stock_mode != "actual") & (forecasting):
            print("forecasting only works for stock_mode = actual")
            sys.exit(0)

    ##==========================================================##
    
    start = time.time()
    #for i in range(5):
    print(f"Starting execution with {num_stations} stations, {num_sessions} sessions, and action space: {action_space}")
    # Get Initial Parameters    
    episode_list, IDs, station_history = helper.user_input(num_stations, num_sessions, fixed_stations, use_expected_balances_only, create_new_OD_matrix, num_clusters, active_cluster, stock_mode)
    #print(f"station_history: {station_history}")
    # Instantiate training
    trainer_QLN = trainer(station_history, forecasting, remove_only, num_stations, num_sessions, action_space, fixed_stations, use_expected_balances_only, num_clusters, active_cluster, forecast_hours, stock_mode, IDs, assign_mode, upper_limit, lower_limit)
    # Execute training
    trainer_QLN.start(episode_list, logging = True, action_space=action_space)
    end = time.time()
    print(f"Execution finished. Time elapsed: {(end-start)/60} min")
