import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from joblib import load

# ///////////////////////////////////////////////
#function to get all clients churn probability: 0 means no churn, 1 means churn
def getChurnProbabilities(random_forest, x):
    return random_forest.predict_proba(x)

def showProbabilities(low,mid,high, proba_matrix, x):
    clients_permanent = []
    clients_low = []
    clients_mid = []
    clients_high = []
    i = 0
    
    #for each client in the data set
    for client in proba_matrix:
        #get all their data and their churn chance into one list
        client_index = x.index[i]
        client_info = x.loc[client_index].values
        client_info = np.append(client_info,client[1])
        #store client data into profiles(permanent, low, mid, high) list
        if client[1] < low:
            clients_permanent.append(client_info)
        elif client[1] < mid:
            clients_low.append(client_info)
        elif client[1] < high:
            clients_mid.append(client_info)
        else:
            clients_high.append(client_info)
        i += 1
    return clients_permanent, clients_low, clients_mid, clients_high

# -----------------------------------------------
def make_clusters(file_path, file_name):
    df = dd.read_csv(file_path + "/" + file_name + ".csv")

    cluster = []
    for i in range(4):
        cluster.append(file_path + "/cluster/cluster_" + str(i))
    
    for i in cluster:
        df.to_csv(i + "/cluster.csv", index=False, single_file=True)
    return cluster

# cluster: archivo csv donde se encuentra un cluster
# cs1, cs2, cs3: rangos de churn segment
def make_perfiles(cluster, cs1, cs2, cs3):
    # Lee el cluster
    df = pd.read_csv(cluster + "/cluster.csv")
    x = df.drop(columns=['TARGET', 'Unnamed: 0'])

    # Obtiene la probabilidad de churn de todos los elementos del cluster
    random_forest = load('random_forest_churn.joblib')
    proba_matrix = getChurnProbabilities(random_forest, x)

    # Segmenta los elementos del cluster
    segments = []
    segments = showProbabilities(cs1, cs2, cs3, proba_matrix, x)

    names = x.columns.to_list()
    names[0] = "CUSTOMER_ID"
    names.append('CHURN_PERCENTAGE')
    
    i = 0
    for segment in segments:
        df = pd.DataFrame(segment, columns = names)
        df.to_csv(cluster +  "/" + str(i) + ".csv")
        i += 1

def make_perfiles_info(cluster):
    info = [{}, {}, {}, {}]
    for i in range(4):
        df = dd.read_csv(cluster + "/" + str(i) + ".csv")

        AMOUNT = df.shape[0].compute()
        BILL_AMOUNT = float(df['BILL_AMOUNT'].sum().compute())
        PREPAID_LINES = float(df['PREPAID_LINES'].sum().compute())
        POSTPAID_LINES = float(df['POSTPAID_LINES'].sum().compute())
        OTHER_LINES = float(df['OTHER_LINES'].sum().compute())
        PARTY_REV = float(df['PARTY_REV'].sum().compute())

        info[i] = {
            "amount" : AMOUNT,
            "bill amount" : BILL_AMOUNT,
            "lines" : [{"type" : "PREPAID_LINES", "amount" : PREPAID_LINES},{"type" : "POSTPAID_LINES", "amount" : POSTPAID_LINES}, {"type" : "OTHER_LINES", "amount" : OTHER_LINES}],
            "revenues" : PARTY_REV
        }

    return info
