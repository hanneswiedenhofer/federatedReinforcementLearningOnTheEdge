import pandas as pd
import numpy as np

q_table = pd.read_csv("distributed/performance_log/20210813125818436022/q_table_127_020210813125818436022.csv")
q_table.columns = ["state","1","3","5","10","0"]
q_table['max']=q_table[["1","3","5","10","0"]].idxmax(axis=1)
print(q_table)
q_table[["state","max"]].sort_values(by=['state'], ascending=False).to_csv("q_table.csv", index=False)