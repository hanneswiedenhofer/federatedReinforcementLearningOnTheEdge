import numpy as np
import matplotlib.pyplot as plt
from env import env
from agent import agent
import datetime
import os
import pandas as pd
import ast
import copy
from clustering import clustering
import matplotlib

class trainer():
    def __init__(self, station_history, forecasting, remove_only, num_stations, num_sessions, action_space, fixed_stations, use_expected_balances_only, num_clusters, active_cluster, forecast_hours, stock_mode, upper_limit, lower_limit):
        # Session Properties
        self.episodes = []
        self.central_server = None
        self.operator = None
        self.sim_stock = []
        self.ID = None
        self.logging = False
        self.station_history = station_history

        # Performance Metric
        self.success_ratio = 0
        self.rewards = []  # [[r from session 1], [r from session 2] ...]
        self.avg_rewards = [] #[np.mean([r from session 1]), np.mean([r from session 2])...]
        self.final_stocks = [] # [[stock from session 1], [stock from session 2] ...]
        self.episode_action_history = []
        self.episode_stock_history = []
        self.session_action_history = []
        self.session_stock_history = []
        self.q_tables = []
        self.forecasting = forecasting
        self.remove_only = remove_only
        self.num_stations = num_stations
        self.num_sessions = num_sessions
        self.action_space = action_space
        self.fixed_stations = fixed_stations
        self.use_expected_balances_only = use_expected_balances_only
        self.num_clusters = num_clusters
        self.active_cluster = active_cluster
        self.forecast_hours = forecast_hours
        self.stock_mode = stock_mode
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit


    def start(self, episodes, IDs, logging, action_space):
        self.episodes = episodes
        self.logging = logging
        self.IDs = IDs
        self.method = 'QLN'
        idx = 0
        for eps in self.episodes:
            # Initiate new evironment and RL agent
            self.central_server = env(IDs = self.IDs, station_history = self.station_history, forecasting = self.forecasting, remove_only = self.remove_only, stock_mode = self.stock_mode, upper_limit = self.upper_limit, lower_limit = self.lower_limit)
            self.sim_stock.append(self.central_server.get_sim_stock())
            self.operator = agent(epsilon = 0.9, lr = 0.01, gamma = 0.9, current_stock = self.central_server.current_stock(), IDs = IDs, action_space=action_space, expected_stock = self.central_server.get_expected_bike_stock(), forecasting = self.forecasting, remove_only = self.remove_only, forecast_hours = self.forecast_hours)#, expected_stock = self.central_server.get_expected_stock())
            rewards, final_stocks = self.train_operator(idx, len(self.episodes), eps)
            # Log the results from this training session
            self.rewards.append(rewards)
            self.avg_rewards.append(np.mean(rewards))
            self.final_stocks.append(final_stocks)
            self.q_tables.append(self.operator.get_q_table())
            self.session_action_history.append(self.episode_action_history)
            #print(f"self.session_action_history: {self.session_action_history}")
            self.session_stock_history.append(self.episode_stock_history)
            #print(f"self.session_stock_history: {self.session_stock_history}")
            self.reset_episode_history()
            
            # Destroy the environment and agent objects
            if idx != len(self.episodes)-1:
                self.central_server = None
                self.operator = None
            idx += 1
        if logging == True:
            
            self.save_session_results(self.get_timestamp(replace = True), action_space)
            pd.set_option("display.max_rows", None, "display.max_columns", None)

            
        return

    def train_operator(self, idx, num_sessions, episodes):
        rewards = 0
        reward_list = []
        final_stocks = []
        step = 0
        current_hour = 0
        for eps in range(episodes): 
            self.central_server.reset()
            while True:
                # Agent picks an action (number of bikes to move)
                # Agent sends the action to bike station environment
                # Agent gets feedback from the environment (e.g. reward of the action, new bike stock after the action, etc.)
                # Agent "learn" the feedback by updating its Q-Table (state, action, reward)
                # Repeat until end of day (23 hours)
                # Reset bike station environment to start a new day, repeat all
                action = self.operator.choose_action(self.central_server.get_old_stock(),self.central_server.get_expected_bike_stock(), current_hour)#, self.central_server.get_expected_stock())
                current_hour, old_stock, new_stock, reward, done, game_over, expected_stock = self.central_server.my_ping(action) #expected_stock,
                if done == True:
                    print("Session {} of {} | Episode: {} | Final Stock: {} | Final Reward: {:.2f}".format(idx+1, num_sessions, eps, old_stock, rewards))
                    reward_list.append(rewards)
                    final_stocks.append(old_stock)
                    rewards = 0
                    self.episode_action_history.append(self.operator.get_hourly_actions())
                    self.episode_stock_history.append(self.operator.get_hourly_stocks())
                    self.operator.reset_hourly_history()
                    break
                
                self.operator.learn(old_stock, action, reward, new_stock, game_over, expected_stock, current_hour)
                step +=1
                rewards += reward
        return reward_list, final_stocks

    def get_timestamp(self, replace):
        if replace == True:
            return str(datetime.datetime.now()).replace(" ", "").replace(":", "").replace(".", "").replace("-", "")
        else:
            return str(datetime.datetime.now())

    def reset_episode_history(self):
        self.episode_action_history = []
        self.episode_stock_history = []

    def cal_performance(self):
        successful_stocking = []
        #print(f"self.final_stocks: {self.final_stocks}")
        #print(f"len(self.final_stocks): {len(self.final_stocks)}")
        print(f"SELF.FINAL_STOCKS: {self.final_stocks}")
        for session in range(len(self.final_stocks)):
            length = len(self.final_stocks[session]) * self.num_stations
            num_overstock = 0
            num_understock = 0
            for el in self.final_stocks[session]:
                for value in el.values():
                    if value > self.upper_limit:
                        num_overstock += 1
                    if value <= self.lower_limit:
                        num_understock +=1
            ratio = (length - num_understock - num_overstock)*100 / length
            successful_stocking.append(ratio)   
        
        return successful_stocking

    def cal_my_performance(self):
        successful_stocking = []
        #print(f"self.final_stocks: {self.final_stocks}")
        #print(f"len(self.final_stocks): {len(self.final_stocks)}")
        #print(f"SELF.FINAL_STOCKS: {self.final_stocks}")
        for session in range(len(self.session_stock_history)):
            length = 0#len(self.session_stock_history[session]) * self.num_stations
            num_overstock = 0
            num_understock = 0
            #print(f"self.session_stock_history[session][-1]: {self.session_stock_history[session][-1]}")
            for el in self.session_stock_history[session][-1]:
                for value in ast.literal_eval(el).values():
                    length += 1
                    if value > self.upper_limit:
                        num_overstock += 1
                    if value <= self.lower_limit:
                        num_understock +=1
            ratio = (length - num_understock - num_overstock)*100 / length
            successful_stocking.append(ratio)   
        
        return successful_stocking

    
    def save_session_results(self, timestamp, action_space):
        print("Creating plots...")

         # --- create a session folder ---
        dir_path = "./performance_log/" + timestamp
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        successful_stocking = self.cal_my_performance()
        
        # --- Write Success Rate to File ---
        fname = dir_path + "/success_rate - " + timestamp + ".txt"

        with open(fname, 'w') as f:
            f.write("Logged at {}".format(self.get_timestamp(replace = False)))
            f.write("\n")
            f.write("This training session ran episodes: {}".format(self.episodes))
            f.write("\n")
            for session in range(len(successful_stocking)):
                f.write("Session {} | Episodes: {} | Success Rate: {:.2f}%".format(session, self.episodes[session], successful_stocking[session]))
                f.write("\n")
         # --- Plot Overall Success Rate by Episode ---
        
        title = "% of Successful Rebalancing - " + timestamp
        
        fig1 = plt.figure()
        plt.plot(self.episodes, successful_stocking)
        plt.xlabel("Episodes")
        plt.ylabel(f"% Success Rate")
        plt.title(title)
        fig1.savefig(dir_path + "/session_success_rate_" + timestamp)
        plt.close()

        # --- Plot Reward History by Training Session ---
        
        for session in range(len(self.rewards)):
            
            fig = plt.figure(figsize=(10, 8))
            
            title = "Reward History by Training Session " + str(session) + " - " + timestamp + ", action_space: " + str(action_space)
            
            x_axis = [x for x in range(self.episodes[session])]
            plt.plot(x_axis, self.rewards[session], label = "Session "+str(session))
            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(title)
            fig.savefig(dir_path + "/reward_history_session_" + str(session) + timestamp)
            plt.close()

            # --- Plot Average Reward History by Training Session ---
        figR = plt.figure(figsize=[10, 8])
        lengths = [len(r) for r in self.rewards]
        means = [np.mean(r) for r in self.rewards]
        if len(self.rewards) > 1:
            increment = (lengths[1]-lengths[0])/20
        else:
            increment = lengths[0]/20

        for reward_list in self.rewards:
            Q3 = np.percentile(reward_list, 75)
            Q1 = np.percentile(reward_list, 25)
            M = np.mean(reward_list)
            location = len(reward_list)
            plt.plot([location-increment, location+increment], [Q1, Q1], 'k-')
            plt.plot([location-increment, location+increment], [Q3, Q3], 'k-')
            plt.plot([location, location], [Q1, Q3], 'k-')
            plt.scatter(location, M, s=100, color='dodgerblue')           

        plt.xlabel('Number of Episodes in Session')
        plt.ylabel('Average Reward per Episode')
        plt.title(f"Average Reward vs. Session Size, action space: {action_space}", size=20)
        plt.xticks(lengths)

        plt.plot(lengths, means, linestyle='--')
        
        figR.savefig(dir_path + "/reward_averages")
        plt.close()
         # --- Save Q tables --- 
        
        for session in range(len(self.q_tables)):
            
            self.q_tables[session].to_csv(dir_path + "/q_table_session_" + \
                        str(session) + timestamp + ".csv")
        
        # --- Comparison Line Chart of First and Last Episode for each Session ---
        
        file_path = dir_path + "/action_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)       
        
        
        for session in range(len(self.session_action_history)):
            
            first_eps_idx = 0
            last_eps_idx = len(self.session_action_history[session])-1
            fig = plt.figure(figsize=(10, 8))
            title = "Session " + str(session) + " - Hourly Action of Eps " + str(first_eps_idx) + " and Eps " + str(last_eps_idx) + ", action space: " + str(action_space)
            
            x_axis = [x for x in range(len(self.session_action_history[session][0]))]
            eps0_hist = []
            for el in self.session_action_history[session][0]:
                eps0_hist.append(el[1])
            eps1_hist = []
            for el in self.session_action_history[session][-1]:
                eps1_hist.append(el[1])
            plt.plot(x_axis, eps0_hist, label = "Eps 0")
            plt.plot(x_axis, eps1_hist, label = "Eps " + str(last_eps_idx))
            
            plt.legend()
            plt.xlabel("Hours")
            plt.ylabel("Number of Bikes Moved")
            plt.title(title)
            
            fig.savefig(file_path + "/action_history_" + str(session) + timestamp)
            plt.close()
        
        
        # --- Comparison Line Chart of Simulated and Rebalanced Bike Stock --- #
        file_path = dir_path + "/stock_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        #matplotlib.rcParams.update({'font.size': 22})
        for session in range(len(self.session_stock_history)):
            
            first_eps_idx = 0
            last_eps_idx = len(self.session_action_history[session])-1
            
            for key, value in self.sim_stock[session].items():
                fig = plt.figure(figsize=(10, 8))
                title = "STATION: "+ key + " [" + self.method + "]" + "Session " + str(session) + " - Original vs. Balanced Bike Stock after " + str(first_eps_idx) + " and Eps " + str(last_eps_idx) + ", action space: " + str(action_space)
                eps0_hist = []
                for el in self.session_stock_history[session][0]:
                    eps0_hist.append(ast.literal_eval(el).get(key))
                eps1_hist = []
                for el in self.session_stock_history[session][-1]:
                    eps1_hist.append(ast.literal_eval(el).get(key))
                x_axis = [x for x in range(len(self.session_stock_history[session][0]))]
                plt.plot(x_axis, value, label = "Original without Balancing")
                #plt.plot(x_axis, eps0_hist, label = "Balanced Bike Stock - Eps 0")
                #plt.plot(x_axis, eps1_hist, label = "Balanced Bike Stock - Eps " + str(last_eps_idx))
                
                plt.axhline(y = self.upper_limit, c = "r", ls = "--", label = "Upper Stock Limit")
                plt.axhline(y = self.lower_limit, c = "r", ls = "--", label = "Lower Stock Limit")
                
                plt.legend()
                plt.xlabel("Hours")
                plt.ylabel("Number of Bike Stock")
                plt.title(title)
                
                fig.savefig(file_path + "/"+  key + "_stock_history_" + str(session) + timestamp)
                plt.close()
            
            fig = plt.figure(figsize=(10,8))
            title = "ALL STATIONS: [" + self.method + "]" + "Session " + str(session) + " - Balanced Bike Stock after Eps " + str(last_eps_idx) + ", action space: " + str(action_space)
            x_axis = [x for x in range(len(self.session_stock_history[session][0]))]
            for key, value in self.sim_stock[session].items():
                eps1_hist = []
                for el in self.session_stock_history[session][-1]:
                    eps1_hist.append(ast.literal_eval(el).get(key))
                plt.plot(x_axis, eps1_hist, label = f"Station {key}, episode: {last_eps_idx}")
            plt.axhline(y = self.upper_limit, c = "r", ls = "--", label = "Upper Stock Limit")
            plt.axhline(y = self.lower_limit, c = "r", ls = "--", label = "Lower Stock Limit") 
            plt.legend()
            plt.xlabel("Hours")
            plt.ylabel("Number of Bike Stock")
            plt.title(title)
            fig.savefig(file_path + "/all_stock_history_" + str(session) + timestamp)
            plt.close()

            # --- Write Parameters to File ---
            fname = dir_path + "/parameters.txt"
            with open(fname, 'w') as f:
                f.write(f"num_stations: {self.num_stations}\n")
                f.write(f"num_sessions: {self.num_sessions}\n")
                f.write(f"action_space: {self.action_space}\n")
                f.write(f"fixed_stations: {self.fixed_stations}\n")
                f.write(f"use_expected_balances_only: {self.use_expected_balances_only}\n")
                f.write(f"forecasting: {self.forecasting}\n")
                f.write(f"num_clusters: {self.num_clusters}\n")
                f.write(f"active_cluster: {self.active_cluster}\n")
                f.write(f"remove_only: {self.remove_only}\n")
                f.write(f"forecast_hours: {self.forecast_hours}\n")
                f.write(f"stock_mode: {self.stock_mode}\n")
        return

