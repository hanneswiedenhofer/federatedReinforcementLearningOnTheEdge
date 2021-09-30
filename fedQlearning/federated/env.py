import pandas as pd
import numpy as np
from collections import defaultdict
import copy
import json
import random
with open('EXPECTED_BALANCES.json') as json_data:
    expected_balance = json.load(json_data)
# https://github.com/ianxxiao/reinforcement_learning_project/blob/master/code/env.py


class env():
    def __init__(self, IDs, station_history, forecasting, remove_only, stock_mode, assign_mode, forecast_hours, upper_limit, lower_limit):
        self.seed = np.random.random_integers(0, 10)
        self.num_hours = 23
        self.current_hour = 0
        self.stock_mode = stock_mode
        #station_history = dict(reversed(list(station_history.items())))
        if self.stock_mode == "actual":     
            self.bike_stock_sim = station_history
        else:
            self.bike_stock_sim = self.generate_stock(self.stock_mode, IDs)
        self.bike_stock = copy.deepcopy(self.bike_stock_sim)#.copy() # to be reset to original copy every episode
        self.old_stock = {}
        self.new_stock = {}
        for key in self.bike_stock.keys():
            self.old_stock[key] = self.bike_stock.get(key)[0]
            self.new_stock[key] = 0
        
        self.rewards = {}
        self.bike_moved = 0
        self.IDs = []
        for el in IDs:
            self.IDs.append(str(el))
        #features of the observation: hour, old stock, new stock
        self.citibike_df = 0
        self.forecasting = forecasting
        #if self.forecasting:
        self.exp_bike_stock_sim = {}
        for key in expected_balance.keys():
            if str(key) in self.IDs:
                self.exp_bike_stock_sim[key] = expected_balance.get(key)

        self.exp_bike_stock = copy.deepcopy(self.exp_bike_stock_sim)#.copy()
        self.expected_stock = {}
        self.expected_stock_new = {}
        for key, value in self.exp_bike_stock.items():
            self.expected_stock[key] = value[0]
            self.expected_stock_new[key] = 0
        self.remove_only = remove_only
        self.dones = {}
        self.game_overs = {}
        for id in self.IDs:
            self.dones[id] = False
            self.game_overs[id] = False
        self.assign_mode = assign_mode
        self.forecast_hours = forecast_hours
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
    
    def generate_stock(self, mode, ID):
        
        # generate a list of 24 hourly bike stock based on mode
        # mode: linear, random, real based on citiBike Data
        bike_stock = {}
        
        if mode == "linear":
            for el in ID:
                bike_stock[str(el)] = [20]
                for i in range(1, 24):
                    bike_stock[str(el)].append(bike_stock[str(el)][i-1]+3)
                
        if mode == "random":
            for el in ID:
                bike_stock[str(el)] = [20]
                for i in range(1, 24):
                    bike_stock[str(el)].append(int(bike_stock[str(el)][i-1] + 3 + np.random.random_integers(-5, 5)))
                
        if mode == "actual":
            pass
            
        return bike_stock


    def my_ping(self, action, key, expected_bike_stock):
        key = str(key)
        self.rewards[key] = 0
        if self.remove_only:
            if action != 0:
                self.update_stock(action, key, expected_bike_stock)
                self.rewards[key] = -0.5*np.abs(action)
        else:
            if action[0] != 0:
                self.update_stock(action, key, expected_bike_stock)
                self.rewards[key] = -0.5*np.abs(action[0]) #Problem?

        if (self.bike_stock[key][self.current_hour] > self.upper_limit) | (self.bike_stock[key][self.current_hour] < self.lower_limit):
        #if self.bike_stock[key][self.current_hour] < 0:
            self.rewards[key] = -30
        if self.current_hour == 23:
            if (self.bike_stock[key][self.current_hour] <= self.upper_limit)&(self.bike_stock[key][self.current_hour] > self.lower_limit):
            # if self.bike_stock[key][self.current_hour] > 0:
                self.rewards[key] = 20
            else: 
                self.rewards[key] = -20
            self.dones[key] = True
            #self.new_stock = 'terminal'
            self.game_overs[key] = True
        
    
    def update_turn(self):
        if self.current_hour != 23:
            self.update_hour()  
            for key in self.IDs:
                self.old_stock[str(key)] = self.bike_stock.get(str(key))[self.current_hour-1]
                self.new_stock[str(key)] = self.bike_stock.get(str(key))[self.current_hour]
                #if self.forecasting:
                self.expected_stock[str(key)] = self.exp_bike_stock.get(str(key))[self.current_hour - 1:]           
                    #if self.current_hour < 23:
                        #self.expected_stock_new[key] = self.exp_bike_stock.get(key)[self.current_hour]
                #else:
                #    self.expected_stock = None
        return self.current_hour, self.old_stock, self.new_stock, self.rewards, self.dones, self.game_overs, self.expected_stock

    def ping(self, actions, expected_bike_stock):
        for station_key in actions.keys():
            #self.rewards[station_key] = 0
            if(actions[station_key] != 0):
                self.update_stock((actions[station_key],station_key), expected_bike_stock)
                self.rewards[station_key] += 0.5 * np.abs(actions[station_key])/len(actions)
        temp_rewards = {}  
        for station_key in self.IDs:
            temp_rewards[station_key] = 0
            if(self.bike_stock.get(station_key)[self.current_hour] > 50 | (self.bike_stock.get(station_key)[self.current_hour] < 0)):
                temp_rewards[station_key] -= 30/len(actions)
            if self.current_hour == 23:
                temp_rewards[station_key] = 0
                if (self.bike_stock.get(station_key)[self.current_hour] <= 50) & (self.bike_stock.get(station_key)[self.current_hour] > 0):
                    temp_rewards[station_key] += 20/len(actions)
                else:
                    temp_rewards[station_key] -= 20/len(actions)
                self.dones[station_key] = True
                self.game_overs[station_key] = True
        self.rewards = copy.deepcopy(temp_rewards)

        # temp_rewards = {}
        # for key in actions.keys():
        #     temp_rewards[key] = 0
        #     if actions[key][0] != 0:
        #         print("test")
        #         self.update_stock(actions[key], expected_bike_stock)
        #         temp_rewards[key] += -0.5*np.abs(actions[key][0])/len(actions)
        #         self.rewards = copy.deepcopy(temp_rewards)
        #         temp_rewards = {}
        #     for key2 in self.IDs:
        #         temp_rewards[key2] = 0
        #         if (self.bike_stock.get(key2)[self.current_hour] > 50) | (self.bike_stock.get(key2)[self.current_hour] < 0):
        #             #if self.bike_stock.get(key)[self.current_hour] > 50:#
        #             temp_rewards[key2] -= 30/len(actions)
        #             #else:
        #             #    temp_rewards[key] += 100 * self.bike_stock.get(key)[self.current_hour]#
        #         if self.current_hour == 23:
        #             temp_rewards[key2] = 0
        #             if (self.bike_stock.get(key2)[self.current_hour] <= 50)&(self.bike_stock.get(key2)[self.current_hour] > 0):
        #                 temp_rewards[key2] += 20/len(actions)
        #             else:
        #                 temp_rewards[key2] -= 20/len(actions)
        #             self.dones[key2] = True
        #             self.game_overs[key2] = True
        #     self.rewards = copy.deepcopy(temp_rewards)
            #print(f"self.reward: {self.reward}")
            # update current_hour
        if self.current_hour != 23:
            self.update_hour()
            for key in self.IDs:
                self.old_stock[str(key)] = self.bike_stock.get(str(key))[self.current_hour-1]
                self.new_stock[str(key)] = self.bike_stock.get(str(key))[self.current_hour]
                if self.forecasting:
                    self.expected_stock[str(key)] = self.exp_bike_stock.get(str(key))[self.current_hour - 1:]           
                    #if self.current_hour < 23:
                        #self.expected_stock_new[key] = self.exp_bike_stock.get(key)[self.current_hour]
                else:
                    self.expected_stock = None
        return self.current_hour, self.old_stock, self.new_stock, self.rewards, self.dones, self.game_overs, self.expected_stock #, self.expected_stock_new

    def get_old_stock(self):
        return self.old_stock

    def get_expected_bike_stock(self):
        #if self.forecasting:
        if self.current_hour < 23:
            return self.exp_bike_stock
        #    else:
        #        return None
        return None

    def update_stock(self, action, key, expected_bike_stock):
        # update bike stock based on RL Agent action at tW
        if self.current_hour != 23:
            if self.remove_only:
                if action != 0:
                    for hour in range(self.current_hour+1, len(self.bike_stock.get(list(self.bike_stock.keys())[0]))):
                        self.bike_stock[str(key)][hour] = self.bike_stock[str(key)][hour] - action
            else:
                if action[0] != 0:
                    # r = copy.deepcopy(self.bike_stock)
                    # del r[str(action[1])] # avoids assigning bikes to origin station
                    # if self.assign_mode == "random":
                    #     my_key = random.choice(list(r))
                    # else:
                    #     if self.assign_mode == "forecast":
                    #         for key in r.keys():
                    #             r[key][self.current_hour+1] = int(round((r[key][self.current_hour+1] + sum(expected_bike_stock.get(key)[:self.forecast_hours]))/self.forecast_hours+1))
                    #     if (self.assign_mode == "extremes") | (self.assign_mode == "forecast"):
                    #         # if action[0] > 0:
                    #         #     #remove from station with highest stock
                    #         #     my_key = -10000
                    #         #     for key in r.keys():
                    #         #         if (r[key][self.current_hour+1] > int(my_key)):
                    #         #             my_key = key
                                
                    #         # else:
                    #         #add to station with lowest stock
                    #         my_key = 100000
                    #         for key in r.keys():
                    #             if (r[key][self.current_hour+1] < int(my_key)):
                    #                 my_key = key
                    # # for key in r.keys():
                    # #     print(f"r[{key}][hour]= {r[key][self.current_hour]}")
                    # # print(f"self.bike_stock[{my_key}][hour]= {self.bike_stock[my_key][self.current_hour]}")
                    # # print("============================================")
                    for hour in range(self.current_hour+1, len(self.bike_stock.get(list(self.bike_stock.keys())[0]))):
                        self.bike_stock[str(key)][hour] = self.bike_stock[str(key)][hour] - action[0]
                        self.bike_stock[str(action[1])][hour] = self.bike_stock[str(action[1])][hour] + action[0]
        else:
            pass
        return
    
    def update_hour(self):
        #print("update_hour")
        self.current_hour += 1
        return

    def reset(self):
        self.num_hours = 23
        self.current_hour = 0
        self.bike_stock = copy.deepcopy(self.bike_stock_sim)#.copy()
        self.dones = {}
        for id in self.IDs:
            self.dones[id] = False
        self.rewards = {}
        self.bike_moved = 0
        self.old_stock = {}
        self.new_stock = {}
        for key in self.bike_stock.keys():
            self.old_stock[key] = self.bike_stock.get(key)[0]
            self.new_stock[key] = 0
        if self.forecasting:
            self.exp_bike_stock = copy.deepcopy(self.exp_bike_stock_sim)#.copy()
            self.expected_stock = {}
            self.expected_stock_new = {}
            for key,value in self.exp_bike_stock.items():
                self.expected_stock[key] = value[0]
                self.expected_stock_new[key] = 0
            
        
    def current_stock(self):
        ret = defaultdict(dict)
        for key, value in self.bike_stock.items():
            ret[key] = self.bike_stock.get(key)[self.current_hour]
        return ret
    
    def get_sim_stock(self):
        return self.bike_stock 