import pandas as pd
import numpy as np
from collections import defaultdict
import copy
import json
with open('EXPECTED_BALANCES.json') as json_data:
    expected_balance = json.load(json_data)
# https://github.com/ianxxiao/reinforcement_learning_project/blob/master/code/env.py


class env():
    def __init__(self, IDs, station_history, forecasting, remove_only, stock_mode, upper_limit, lower_limit):
        self.seed = np.random.random_integers(0, 10)
        self.num_hours = 23
        self.current_hour = 0
        self.stock_mode = stock_mode
        if self.stock_mode == "actual":     
            self.bike_stock_sim = station_history
        else:
            self.bike_stock_sim = self.generate_stock(self.stock_mode, IDs)
        self.bike_stock = copy.deepcopy(self.bike_stock_sim)#.copy() # to be reset to original copy every episode
        self.old_stock = {}
        self.new_stock = {}
        for key,value in self.bike_stock.items():
            self.old_stock[key] = value[0]
            self.new_stock[key] = 0

        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.IDs = []
        for el in IDs:
            self.IDs.append(str(el))
        #features of the observation: hour, old stock, new stock
        self.citibike_df = 0
        self.game_over = False
        self.forecasting = forecasting
        #print(f"self.IDs: {self.IDs}")
        if self.forecasting:
            self.exp_bike_stock_sim = {}
            for key in expected_balance.keys():
                if str(key) in self.IDs:
                    self.exp_bike_stock_sim[key] = expected_balance.get(key)
            #print(f"self.exp_bike_stock_sim: {self.exp_bike_stock_sim}")
            self.exp_bike_stock = copy.deepcopy(self.exp_bike_stock_sim)#.copy()
            self.expected_stock = {}
            self.expected_stock_new = {}
            for key, value in self.exp_bike_stock.items():
                self.expected_stock[key] = value[0]
                self.expected_stock_new[key] = 0
        self.remove_only = remove_only
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


    def my_ping(self, action):
        if self.remove_only:
            if action[1] != 0:
                self.update_stock(action)
                self.reward = -0.5*np.abs(action[1])
        else:
            if action[2] != 0:
                self.update_stock(action)
                self.reward = -0.5*np.abs(action[2])
        temp_reward = 0
        for key in self.bike_stock.keys():
            if (self.bike_stock[key][self.current_hour] > self.upper_limit) | (self.bike_stock[key][self.current_hour] < self.lower_limit):
            #if self.bike_stock[key][self.current_hour] < 0:
                temp_reward -= 30
            if self.current_hour == 23:
                if (self.bike_stock[key][self.current_hour] <= self.upper_limit)&(self.bike_stock[key][self.current_hour] > self.lower_limit):
                # if self.bike_stock[key][self.current_hour] > 0:
                    temp_reward += 20
                else: 
                    temp_reward -= 20
                self.done = True
                #self.new_stock = 'terminal'
                self.game_over = True
        self.reward = temp_reward
        if self.current_hour != 23:
            self.update_hour()  
            for key in self.bike_stock.keys():
                self.old_stock[key] = self.bike_stock.get(key)[self.current_hour-1]
                self.new_stock[key] = self.bike_stock.get(key)[self.current_hour]
                if self.forecasting:
                    self.expected_stock[key] = self.exp_bike_stock.get(key)[self.current_hour - 1:]           
                    #if self.current_hour < 23:
                    #    self.expected_stock_new[key] = self.exp_bike_stock.get(key)[self.current_hour]
                else:
                    self.expected_stock = None
        return self.current_hour, self.old_stock, self.new_stock, self.reward, self.done, self.game_over, self.expected_stock


    def ping(self, action):
        if self.remove_only:
            if action[1] != 0:
                self.update_stock(action)
                self.reward = -0.5*np.abs(action[1])
        else:
            if action[2] != 0:
                self.update_stock(action)
                self.reward = -0.5*np.abs(action[2])
        temp_reward = 0
        for key in self.bike_stock.keys():
            if (self.bike_stock.get(key)[self.current_hour] > self.upper_limit) | (self.bike_stock.get(key)[self.current_hour] < self.lower_limit):
                temp_reward -= 30
        if self.current_hour == 23:
            temp_reward = 0
            for key in self.bike_stock.keys():
                if (self.bike_stock.get(key)[self.current_hour] <= self.upper_limit)&(self.bike_stock.get(key)[self.current_hour] > self.lower_limit):
                    temp_reward += 20
                else:
                    temp_reward -= 20
            self.done = True
            self.game_over = True
        self.reward = temp_reward
        if self.current_hour != 23:
            self.update_hour()
            self.old_stock = {}
            self.new_stock = {}
            for key in self.bike_stock.keys():
                self.old_stock[key] = self.bike_stock.get(key)[self.current_hour-1]
                self.new_stock[key] = self.bike_stock.get(key)[self.current_hour]
                if self.forecasting:
                    self.expected_stock[key] = self.exp_bike_stock.get(key)[self.current_hour - 1:]           
                    #if self.current_hour < 23:
                        #self.expected_stock_new[key] = self.exp_bike_stock.get(key)[self.current_hour]
                else:
                    self.expected_stock = None
                    #self.expected_stock_new = None

        return self.current_hour, self.old_stock, self.new_stock, self.reward, self.done, self.game_over, self.expected_stock#, self.expected_stock_new

    def get_old_stock(self):
        return self.old_stock

    def get_expected_bike_stock(self):
        if self.forecasting:
            if self.current_hour < 23:
                return self.exp_bike_stock
            else:
                return None
        return None

    def update_stock(self, action):
        # update bike stock based on RL Agent action at tW
        if self.current_hour != 23:
            if self.remove_only:
                if action[1] != 0:
                    for hour in range(self.current_hour+1, len(self.bike_stock.get(list(self.bike_stock.keys())[0]))):
                        self.bike_stock[str(action[0])][hour] = self.bike_stock[str(action[0])][hour] - action[1]
            else:
                if action[2] != 0:
                    for hour in range(self.current_hour+1, len(self.bike_stock.get(list(self.bike_stock.keys())[0]))):
                        self.bike_stock[str(action[0])][hour] = self.bike_stock[str(action[0])][hour] - action[2]
                        self.bike_stock[str(action[1])][hour] = self.bike_stock[str(action[1])][hour] + action[2]
        else:
            pass
        return
    
    def update_hour(self):
        # update current_hour 
        self.current_hour += 1
        return

    def reset(self):
        self.num_hours = 23
        self.current_hour = 0
        self.bike_stock = copy.deepcopy(self.bike_stock_sim)#.copy()
        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.old_stock = {}
        self.new_stock = {}
        for key, value in self.bike_stock.items():
            self.old_stock[key] = value[0]
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