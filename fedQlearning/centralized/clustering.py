import pandas as pd
import numpy as np
import sklearn.cluster
import json
with open('EXPECTED_BALANCES.json') as json_data:
    expected_balance = json.load(json_data)


class clustering():
    def __init__(self, n_clusters, new_file):
        self.n_clusters = n_clusters
        self.new_file = new_file

    def cluster(self,expected_balances_only):
        if self.new_file:
            df = pd.read_csv("../data/201709-citibike-tripdata.csv")
            df['starttime'] = pd.to_datetime(df['starttime'], infer_datetime_format= True)
            df['stoptime'] = pd.to_datetime(df['stoptime'], infer_datetime_format= True)
            df.drop_duplicates(inplace=True)
            df["day"] = df["starttime"].dt.day
            df["month"] = df["starttime"].dt.month
            df['start_hour'] = df['starttime'].dt.hour
            df['end_hour'] = df['stoptime'].dt.hour

            df = df[df['day'] == 1]
            df = df[df["month"] == 9]

            if expected_balances_only:
                df = df[df["start station id"].isin(expected_balance.keys())]
                df = df[df["end station id"].isin(expected_balance.keys())]

            pivot = df.pivot_table(values='tripduration', index='start station id', columns='end station id',
                    fill_value=0, aggfunc=lambda x: x.value_counts().count())
            pivot.to_csv("../data/ODMatrix.csv")
            ward = sklearn.cluster.AgglomerativeClustering(n_clusters = self.n_clusters, linkage="ward")
            ward.fit(pivot) 
            res = pd.DataFrame({"Station": pivot.iloc[:,0], "Cluster": ward.labels_})
            res = res["Cluster"]
            res.to_csv("../data/clusters.csv")
            #print(res)
        else:
            res = pd.read_csv("../data/clusters.csv", index_col=0, squeeze=True)
            #print(res)
        
        my_dict = res.to_dict()
        clusters = {}
        for i, v in my_dict.items():
            clusters[v] = [i] if v not in clusters.keys() else clusters[v] + [i]
        return clusters