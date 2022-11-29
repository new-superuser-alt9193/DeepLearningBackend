import dask.dataframe as dd
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist

from joblib import load

import random

import pandas as pd
import dask.dataframe as dd
from dask_ml import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

target = ["TARGET", "Target", "target", "CHURN", "Churn", "churn", "RESULT", "Result", "result"]
# ///////////////////////////////////////////////
#get K value for K Means
def getK(x):
    scaled_x = x
    distortions = {}
    i = 1
    while True:
        #fit k means clustering according to i
        km = KMeans(
            n_clusters= i, init='random',
            n_init=10, max_iter=300, 
            tol=1e-04, random_state=0
        ).fit(scaled_x)
        #get distortion of actual k, which is the sum distance between clusters and their centroid
        current_distortion = sum(np.min(cdist(scaled_x, km.cluster_centers_,'euclidean'), axis=1)) / scaled_x.shape[0] 
        distortions[i] = current_distortion
        #getting 3 iterations
        if i >= 3:
            #get slope between i -1 and i - 2, i and i - 1
            m1 = distortions[i - 2] - distortions[i - 1]
            m2 = distortions[i - 1] - distortions[i]
            #get the differential between slopes and addition
            m_dif = m1 - m2
            m_sum = m1 + m2
            #get the percentage representation of differential, since 100% equals to the sum of slope values
            dif_percentage = (m_dif * 100) / m_sum
            #if this percentage is less than 25%, it means that  distortion will have a linear behaviour as more k iterations
            #so we can say that a correct k value for optimal clustering is i - 2.
            if dif_percentage < 25.0:
                break
        i += 1
    return i - 2

def makeClusters(n):
    km = KMeans(
        n_clusters= n, init='random',
        n_init=10, max_iter=300, 
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(x)
    cluster_labels = km.labels_
    return km, y_km, cluster_labels

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
# Obtiene una lista con las columnas categoricas a partir de una diferencia de conjuntos
def getCategoricalColumns(df):
    cols = set(df.columns)
    numColumns = set(df._get_numeric_data().columns)
    
    return list(cols - numColumns)

# Convierte las variables categoricas en numericas en base a label encoding
def getNumericDataset(df):
    categoricalColumns = getCategoricalColumns(df)
    numDataset = df.drop(columns= categoricalColumns)
    categoricalDataset = df[categoricalColumns]

    le = preprocessing.LabelEncoder()
    for i in categoricalColumns:
        categoricalDataset[i] = le.fit_transform(categoricalDataset[i])
    
    for i in categoricalColumns:
        numDataset[i] = categoricalDataset[i]
    
    return numDataset

def scaler(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def reduce_csv(csv_file):
    df = dd.read_csv(csv_file)
    result = df[df.columns.intersection(target)]
    df = df.drop(columns= target + ["Unnamed: 0"], errors='ignore')
    
    x = getNumericDataset(df)
    columns = list(x.columns)
    x = scaler(x)
    
    pca = PCA(n_components=.8, svd_solver='full')
    pca.fit(x)

    reduced = []
    for i in range(0, len(pca.explained_variance_ratio_)):
        print(i)
        reduced.append(columns[i])   

    result[reduced] = df[reduced]

    result.to_csv(csv_file, index=False, single_file=True)

# -----------------------------------------------
def make_clusters(file_path, file_name):
    if True:
        df = dd.read_csv(file_path + "/" + file_name + ".csv")

        cluster = []
        for i in range(4):
            cluster.append(file_path + "/cluster/cluster_" + str(i))

        for i in cluster:
            df.to_csv(i + "/cluster.csv", index=False, single_file=True)
        return [{"name": "cluster_1", "percentage": 25}, {"name": "cluster_2", "percentage": 25}, {"name": "cluster_3", "percentage": 25}, {"name": "cluster_4", "percentage": 25}]
    else:
        df = dd.read_csv(file_path + "/" + file_name + ".csv")
        df = df.drop(columns=["Unnamed: 0"], errors='ignore')

        x = np.array(df.drop(columns=target, errors='ignore'))

        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        K = range(1, 10)
        scaled_x = x 
        for k in K:
            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k).fit(scaled_x)
        
            distortions.append(sum(np.min(cdist(scaled_x, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / scaled_x.shape[0])
            inertias.append(kmeanModel.inertia_)
        
            mapping1[k] = sum(np.min(cdist(scaled_x, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / scaled_x.shape[0]
            mapping2[k] = kmeanModel.inertia_


        n = getK(x)
        km, y_km, km_labels = makeClusters(n)

        clusters = pd.DataFrame(data = {'cluster': y_km})

        # Une la clasificacion con los datos del dataset
        df_clusters = dd.merge(clusters, df.drop(columns=target, errors='ignore'), left_index=True, right_index=True)

        df_clusters = dd.merge(df_clusters, df[['TARGET']], left_index=True, right_index=True)

        # sort the dataframe
        df_clusters = df_clusters.sort_values(by=['cluster'])

        info = []
        amount = 0
        for i in range (n):
            df_to_csv = df_clusters[df_clusters['cluster'] == i]
            clusters_amount = df_to_csv.shape[0].compute()
            amount += clusters_amount 
            info.append({"name": "cluster_" + str(i + 1), "percentage": clusters_amount})
            df_to_csv.to_csv(file_path + "/cluster/cluster_" + str(i) + "/cluster.csv", single_file=True)
        for i in info:
            i["percentage"] = i["percentage"] / amount
        
        return info

# cluster: archivo csv donde se encuentra un cluster
# cs1, cs2, cs3: rangos de churn segment
def make_perfiles(cluster, cs1, cs2, cs3, model_file):
    # Lee el cluster
    df = pd.read_csv(cluster + "/cluster.csv")
    x = df.drop(columns= target + ['Unnamed: 0'], errors='ignore')

    # Obtiene la probabilidad de churn de todos los elementos del cluster
    random_forest = load(model_file)
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
