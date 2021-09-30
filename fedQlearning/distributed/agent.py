import pandas as pd
import numpy as np
import json
# https://github.com/ianxxiao/reinforcement_learning_project/blob/master/code/rl_brain.py

class agent():
    def __init__(self, epsilon, lr, gamma, current_stock, IDs, action_space, expected_stock, forecasting, remove_only, forecast_hours, id):
        self.IDs = IDs
        self.actions = []
        self.remove_only = remove_only
        if self.remove_only:
            for i in action_space:
                self.actions.append(i)
            self.actions.append(0)
        else:    
            for my_id in self.IDs:
                if my_id != id:
                    for i in action_space:
                        self.actions.append((i,my_id))
            self.actions.append((0,0))
        #else:
        #    for i in action_space:
        #        self.actions.append(i)
        #        self.actions.append(-i)
        
        self.reward = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.current_stock = current_stock
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)
        self.hourly_action_history = []
        self.hourly_stock_history = []
        self.forecasting = forecasting
        if self.forecasting:
            self.expected_stock = expected_stock
        self.forecast_hours = forecast_hours
        self.id = id
    
    def choose_action(self, current_stock, ex, key, current_hour):
        self.check_state_exist(current_stock)
        self.current_stock = current_stock
        if self.forecasting:
            self.expected_stock = ex
            #print(f"CURRENT_STOCK: {current_stock}")
            try:
                avg_dict = int(round((current_stock + ex.get(key)[current_hour + self.forecast_hours])/2))
                    #avg = int(round(0.5*current_stock + 0.5*ex))
            except:
                avg_dict = current_stock
            self.check_state_exist(avg_dict)
            valid_state_action = self.q_table.loc[avg_dict, :]
        else:
            valid_state_action = self.q_table.loc[current_stock, :]
            #print(f"valid_state_action: {valid_state_action}")
        if np.random.uniform() < self.epsilon:
            try:
                # find the action with the highest expected reward
                valid_state_action = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                #valid_state_action_cp = valid_state_action.copy()
                # for key in valid_state_action_cp.keys():
                #     if(key[0] != 0):
                #         if current_stock.get(str(key[0])) <= key[2]:
                #             valid_state_action_cp.drop(index=key, inplace=True)
                action = valid_state_action.idxmax()
            except:
                # if action list is null, default to 0
                action = (0,0)
        else:
            # randomly choose an action
            # re-pick if the action leads to negative stock
            try:
                action = np.random.choice(valid_state_action.index)
            except:
                action = (0,0)
        if self.remove_only:
            self.hourly_action_history.append(action)
        else:
            self.hourly_action_history.append(action[0])
        self.hourly_stock_history.append(current_stock)
        return action
    
    def learn(self, current_stock, current_action, reward_received, new_stock, end_of_day, ex, key, current_hour):
        self.check_state_exist(new_stock)
        #q_predict = self.q_table.loc[current_stock, [current_action]]#(str(current_action[0]), str(current_action[1]), str(current_action[2]))]
        if self.forecasting:
            #print(f"ex: {ex}")
            if len(ex) > 1:
                avg_dict = int(round((current_stock + ex[self.forecast_hours])/2))
            else:
                avg_dict = current_stock
            self.check_state_exist(avg_dict)
            #print(f"AVG_DICT: {json.dumps(avg_dict)}")
            q_predict = self.q_table.loc[avg_dict, [current_action]]
        else:
            q_predict = self.q_table.loc[current_stock, [current_action]]
        if end_of_day == False:
            # Updated Q Target Value if it is not end of day  
            q_target = reward_received + self.gamma * self.q_table.loc[new_stock, :].max()
        else:
            # Update Q Target Value as Immediate reward if end of day
            q_target = reward_received
        #print(f"current_stock: {current_stock}")
        #print(f"q_target: {q_target}")
        #print(f"q_predict: {q_predict}")
        self.q_table.loc[current_stock, [current_action]] += self.lr * (q_target - q_predict)
        return
    
    
    def check_state_exist(self, state):
        # Add a new row with state value as index if not exist
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                        [0]*len(self.actions), 
                        index = self.q_table.columns,
                        name = state
                        )
                )
        return
    
    def find_valid_action(self, state_action):
        for action in self.actions:
            if self.current_stock + action < 0:
                state_action.drop(index = action, inplace = True)
        return state_action
    

    def print_q_table(self):
        print(self.q_table)

    def get_q_table(self):
        return self.q_table

    
    def get_hourly_actions(self):
        return self.hourly_action_history
    
    def get_hourly_stocks(self):
        #print(f"get_hourly_stocks for ID {self.id}: {self.hourly_stock_history}")
        return self.hourly_stock_history

    
    def reset_hourly_history(self):
        self.hourly_action_history = []
        self.hourly_stock_history = []