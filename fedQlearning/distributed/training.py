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

class trainer():
    def __init__(self, station_history, forecasting, remove_only, num_stations, num_sessions, action_space, fixed_stations, use_expected_balances_only, num_clusters, active_cluster, forecast_hours, stock_mode, IDs, assign_mode, upper_limit, lower_limit):
        # Session Properties
        self.episodes = []
        self.central_server = None
        self.operators = None
        self.sim_stock = []
        self.IDs = IDs
        self.logging = False
        self.station_history = station_history

        # Performance Metric
        self.success_ratio = 0
        self.rewards = []  # [[r from session 1], [r from session 2] ...]
        self.avg_rewards = {} #[np.mean([r from session 1]), np.mean([r from session 2])...]
        self.final_stocks = [] # [[stock from session 1], [stock from session 2] ...]
        self.episode_action_history = {}
        self.episode_stock_history = {}
        for i in self.IDs:
            self.episode_action_history[int(i)] = []
            self.episode_stock_history[int(i)] = []
        # self.episode_action_history = dict.fromkeys(self.operators.keys(), [])
        # self.episode_stock_history = dict.fromkeys(self.operators.keys(), [])
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
        self.assign_mode = assign_mode
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit


    def start(self, episodes, logging, action_space):
        self.episodes = episodes
        self.logging = logging
        self.method = 'QLN'
        idx = 0
        for eps in self.episodes:
            # Initiate new evironment and RL agent
            self.central_server = env(IDs = self.IDs, station_history = self.station_history, forecasting = self.forecasting, remove_only = self.remove_only, stock_mode = self.stock_mode, assign_mode = self.assign_mode, forecast_hours = self.forecast_hours, upper_limit = self.upper_limit, lower_limit = self.lower_limit)
            self.sim_stock.append(self.central_server.get_sim_stock())
            self.operators = {}
            for i in self.IDs:
                self.operators[i] = agent(epsilon = 0.9, lr = 0.01, gamma = 0.9, current_stock = self.central_server.current_stock().get(str(i)), IDs = self.IDs, action_space=action_space, expected_stock = self.central_server.get_expected_bike_stock(), forecasting = self.forecasting, remove_only = self.remove_only, forecast_hours = self.forecast_hours, id = i)#, expected_stock = self.central_server.get_expected_stock())
            rewards, final_stocks = self.train_operator(idx, len(self.episodes), eps)
            # Log the results from this training session
            self.rewards.append(rewards)
            self.final_stocks.append(final_stocks)
            my_qtable = {}
            for key in self.operators.keys():
                #print(f"self.operators.get(key).get_q_table(): {self.operators.get(key).get_q_table()}")
                #q_table_list.append()
                my_qtable[key] = self.operators.get(key).get_q_table()
            self.q_tables.append(my_qtable)
            self.session_action_history.append(self.episode_action_history)
            #print(f"self.session_stock_history: {self.episode_stock_history}")
            #forzajuve
            self.session_stock_history.append(self.episode_stock_history)
            #print(f"self.session_stock_history: {self.session_stock_history}")
            self.reset_episode_history()
            
            # Destroy the environment and agent objects
            if idx != len(self.episodes)-1:
                self.central_server = None
                self.operators = None
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
        for eps in range(episodes): 
            self.central_server.reset()
            while True:
                # Agent picks an action (number of bikes to move)
                # Agent sends the action to bike station environment
                # Agent gets feedback from the environment (e.g. reward of the action, new bike stock after the action, etc.)
                # Agent "learn" the feedback by updating its Q-Table (state, action, reward)
                # Repeat until end of day (23 hours)
                # Reset bike station environment to start a new day, repeat all
                actions = {}
                self.old_stocks = {}
                self.new_stocks = {}
                self.dones = {}
                self.game_overs = {}
                self.expected_stocks = {}
                current_hour = 0
                for key in self.operators.keys():
                    if self.forecasting:
                        action = self.operators.get(key).choose_action(self.central_server.get_old_stock().get(str(key)),self.central_server.get_expected_bike_stock(), key, current_hour)
                    else:
                        action = self.operators.get(key).choose_action(self.central_server.get_old_stock().get(str(key)),None, key, current_hour)
                    actions[key] = action
                    self.central_server.my_ping(action, key, self.central_server.get_expected_bike_stock())
                #current_hour, self.old_stocks, self.new_stocks, my_rewards, self.dones, self.game_overs, self.expected_stocks  = self.central_server.ping(actions, self.central_server.get_expected_bike_stock()) #expected_stock,
                current_hour, self.old_stocks, self.new_stocks, my_rewards, self.dones, self.game_overs, self.expected_stocks  = self.central_server.update_turn() #expected_stock,
                if all(value == True for value in self.dones.values()):
                    print(f"Session {idx+1} of {num_sessions} | Episode: {eps} | Final Stock: {self.old_stocks} | Final Reward: {rewards}")
                    reward_list.append(rewards)
                    final_stocks.append(self.old_stocks)
                    rewards = 0
                    #print(f"self.episode_stock_history: {self.episode_stock_history}")
                    for key in self.operators.keys():
                        # print(f"key: {key}")
                        # print(f"self.operators.get(key).get_hourly_actions(): {self.operators.get(key).get_hourly_actions()}")
                        # print(f"self.operators.get(key).get_hourly_stocks(): {self.operators.get(key).get_hourly_stocks()}")
                        self.episode_action_history[key].append(self.operators.get(key).get_hourly_actions())
                        self.episode_stock_history[key].append(self.operators.get(key).get_hourly_stocks())
                        self.operators.get(key).reset_hourly_history()
                    #print(f"self.episode_stock_history: {self.episode_stock_history}")
                    # print(f"self.episode_stock_history: {self.episode_stock_history}")
                    # print("=====================")
                    break
                for key in self.operators.keys():
                    if self.forecasting:
                        self.operators.get(key).learn(self.old_stocks[str(key)], actions[key], my_rewards[str(key)], self.new_stocks[str(key)], self.game_overs[str(key)], self.expected_stocks[str(key)], key, current_hour)
                    else:
                        #print(f"self.old_stocks: {self.old_stocks}")
                        #print(f"self.new_stocks: {self.new_stocks}")
                        #print(f"self.game_overs: {self.game_overs}")
                        #print(f"my_rewards: {my_rewards}")
                        self.operators.get(key).learn(self.old_stocks[str(key)], actions[key], my_rewards[str(key)], self.new_stocks[str(key)], self.game_overs[str(key)], None, key, current_hour)

                step +=1
                rewards += sum(my_rewards.values())
        return reward_list, final_stocks

    def get_timestamp(self, replace):
        if replace == True:
            return str(datetime.datetime.now()).replace(" ", "").replace(":", "").replace(".", "").replace("-", "")
        else:
            return str(datetime.datetime.now())

    def reset_episode_history(self):
        self.episode_action_history = {}
        self.episode_stock_history = {}
        for i in self.IDs:
            self.episode_action_history[int(i)] = []
            self.episode_stock_history[int(i)] = []

    def cal_performance(self):
        successful_stocking = []
        #print(f"self.final_stocks: {self.final_stocks}")
        #print(f"len(self.final_stocks): {len(self.final_stocks)}")
        #print(f"SELF.FINAL_STOCKS: {self.final_stocks}")
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
            #print(f"self.session_stock_history[session]: {self.session_stock_history[session]}")
            for station in self.session_stock_history[session]:
                #print(f"self.session_stock_history[session][station]: {self.session_stock_history[session][station]}")
                for value in self.session_stock_history[session][station][-1]:
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
            #print(f"SELF.REWARDS: {self.rewards}")
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
        
        ###=============NOT WORKING RIGHT==========###
        for session in range(len(self.q_tables)):
            for key in self.q_tables[session].keys():
                    self.q_tables[session][key].to_csv(dir_path + "/q_table_" + str(key) + "_" + str(session) + timestamp + ".csv")
        # for station in range(len(self.q_tables)):
        #     for key in station.keys():
        #         for el in self.q_tables[key][session]:
        #             el.to_csv(dir_path + "/q_table_session_" + \
        #                         str(session) + timestamp + ".csv")
        
        # --- Comparison Line Chart of First and Last Episode for each Session ---
        
        file_path = dir_path + "/action_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)       
        
        
        for session in range(len(self.session_action_history)):
            first_eps_idx = 0
            for key, value in self.sim_stock[session].items():
                last_eps_idx = len(self.session_action_history[session].get(int(key)))-1
                fig = plt.figure(figsize=(10, 8))
                title = "Session " + str(session) + " - Hourly Action of Eps " + str(first_eps_idx) + " and Eps " + str(last_eps_idx) + ", action space: " + str(action_space)
                x_axis = [x for x in range(len(self.session_action_history[session].get(int(key))[0]))]
                plt.plot(x_axis, self.session_action_history[session].get(int(key))[0], label = "Eps 0")
                plt.plot(x_axis, self.session_action_history[session].get(int(key))[-1], label = "Eps " + str(last_eps_idx))
                
                plt.legend()
                plt.xlabel("Hours")
                plt.ylabel("Number of Bikes Moved")
                plt.title(title)
                
                fig.savefig(file_path + "/" +  key + "_action_history_" + str(session) + timestamp)
                plt.close()
            
        
        # --- Comparison Line Chart of Simulated and Rebalanced Bike Stock --- #
        file_path = dir_path + "/stock_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        for session in range(len(self.session_stock_history)):
            first_eps_idx = 0
            
            
            #print(f"self.sim_stock[session]: {self.sim_stock[session]}")
            #print(f"self.session_stock_history[session]: {self.session_stock_history[session]}")
            for key, value in self.sim_stock[session].items():
                last_eps_idx = len(self.session_stock_history[session].get(int(key)))-1
                fig = plt.figure(figsize=(10, 8))
                title = "STATION: "+ key + " [" + self.method + "]" + "Session " + str(session) + " - Original vs. Balanced Bike Stock after " + str(first_eps_idx) + " and Eps " + str(last_eps_idx) + ", action space: " + str(action_space)
                # eps0_hist = []
                # for el in self.session_stock_history[session][0]:
                #     eps0_hist.append(ast.literal_eval(el).get(key))
                # eps1_hist = []
                # for el in self.session_stock_history[session][-1]:
                #     eps1_hist.append(ast.literal_eval(el).get(key))
                # print(f"key: {key}")
                # print(f"self.session_stock_history[session]: {self.session_stock_history[session]}")
                # print(f"self.session_stock_history[session].get(key): {self.session_stock_history[session].get(int(key))}")
                x_axis = [x for x in range(len(self.session_stock_history[session].get(int(key))[0]))]
                #print(f"x_axis: {x_axis}")
                plt.plot(x_axis, value, label = "Original without Balancing")
                # print(f"self.session_action_history[session]: {self.session_action_history[session]}")
                # print(f"self.session_action_history[session].get(int({key}))[0]: {self.session_action_history[session].get(int(key))[0]}")
                plt.plot(x_axis, self.session_stock_history[session].get(int(key))[0], label = "Balanced Bike Stock - Eps 0")
                plt.plot(x_axis, self.session_stock_history[session].get(int(key))[-1], label = "Balanced Bike Stock - Eps " + str(last_eps_idx))
                #print(f"0: {self.session_action_history[session].get(int(key))[0]}")
                #print(f"-1: {self.session_action_history[session].get(int(key))[0]}")
                self.write_history_to_file(key, dir_path, timestamp, self.session_stock_history[session].get(int(key))[-1])
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
            x_axis = [x for x in range(len(self.session_stock_history[session].get(int(key))[0]))]
            for key, value in self.sim_stock[session].items():
                plt.plot(x_axis, self.session_stock_history[session].get(int(key))[-1], label = f"Station {key}, episode: {last_eps_idx}")
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

            # fname = dir_path + "/action_history_last_epsiode.txt"
            # with open(fname, 'w') as f:
            #     for key in self.episode_action_history.keys():
            #         f.write(f"{key}")
            #         f.write(f"{self.episode_action_history[key]}")
            #         f.write("==================================")
        return


    def write_history_to_file(self, station, dir_path, timestamp, history):
            fname = dir_path + "/stock_history - " + timestamp + ".txt"
            with open(fname, 'a') as f:
                f.write(str(station) + str(history).replace("[", "").replace("]", "") + "\n")
                #for el in history:
                #    f.write(str(el))