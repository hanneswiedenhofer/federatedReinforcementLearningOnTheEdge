import pandas as pd
import numpy as np
import json
# https://github.com/ianxxiao/reinforcement_learning_project/blob/master/code/rl_brain.py

class agent():
    def __init__(self, epsilon, lr, gamma, current_stock, IDs, action_space, expected_stock, forecasting, remove_only, forecast_hours):
        self.IDs = IDs
        self.actions = []
        self.remove_only = remove_only
        if self.remove_only:
            for i in action_space:
                for ss in self.IDs:
                    self.actions.append((ss,i))
            self.actions.append((0,0))
        else:
            for i in action_space:
                for s1 in self.IDs:
                    for s2 in self.IDs:
                        if s2 != s1:
                            self.actions.append((s2,s1,i))
            self.actions.append((0,0,0))
        #self.actions = action_space
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
    
    def choose_action(self, current_stock, ex, current_hour):
        current_stock_new = json.dumps(current_stock)
        self.check_state_exist(current_stock_new)
        self.current_stock = current_stock_new
        if self.forecasting:
            self.expected_stock = ex
            #print(f"CURRENT_STOCK: {current_stock}")
            try:
                avg_dict = {}
                for key,value in current_stock.items():
                    #######!!!!!!!!!TO CHANGE!!!!!!!!!!!##########
                    avg_dict[key] = int(round((value + ex.get(key)[current_hour + self.forecast_hours])/2))
                    #print(f"ex: {ex}")
                    #print(f"avg_dict[key]: {avg_dict[key]}")
                #avg = int(round(0.5*current_stock + 0.5*ex))
            except:
                #print(f"except!")
                avg_dict = current_stock_new
            self.check_state_exist(json.dumps(avg_dict))
            valid_state_action = self.q_table.loc[json.dumps(avg_dict), :]
        else:
            valid_state_action = self.q_table.loc[current_stock_new, :]
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
        self.hourly_action_history.append(action)
        self.hourly_stock_history.append(current_stock_new)
        return action
    
    def learn(self, current_stock, current_action, reward_received, new_stock,  end_of_day, ex, current_hour):
        new_stock = json.dumps(new_stock)
        self.check_state_exist(new_stock)
        current_stock_new = json.dumps(current_stock)
        #q_predict = self.q_table.loc[current_stock, [current_action]]#(str(current_action[0]), str(current_action[1]), str(current_action[2]))]
        if self.forecasting:
            avg_dict = {}
            for key,value in current_stock.items():
                if len(ex.get(key)) > self.forecast_hours:
                    avg_dict[key] = int(round((value + ex.get(key)[self.forecast_hours])/2))
                else:
                    avg_dict[key] = value
            self.check_state_exist(json.dumps(avg_dict))
            #print(f"AVG_DICT: {json.dumps(avg_dict)}")
            q_predict = self.q_table.loc[json.dumps(avg_dict), [current_action]]
        else:
            q_predict = self.q_table.loc[current_stock_new, [current_action]]
        if end_of_day == False:
            # Updated Q Target Value if it is not end of day  
            q_target = reward_received + self.gamma * self.q_table.loc[new_stock, :].max()
        else:
            # Update Q Target Value as Immediate reward if end of day
            q_target = reward_received
        self.q_table.loc[current_stock_new, [current_action]] += self.lr * (q_target - q_predict)
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
        return self.hourly_stock_history

    
    def reset_hourly_history(self):
        self.hourly_action_history = []
        self.hourly_stock_history = []