import dask.dataframe as dd
import pandas as pd
import numpy as np

from dask_ml import preprocessing
from dask_ml.model_selection import train_test_split
from dask_ml.wrappers import ParallelPostFit

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from joblib import load, dump

from scipy.spatial.distance import cdist
import random
from collections import OrderedDict

target = ["TARGET", "Target", "target", "CHURN", "Churn", "churn", "RESULT", "Result", "result", "EXITED", "Exited", "exited"]
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

def makeClusters(n, x):
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
    numDataset = df.drop(columns= categoricalColumns, errors='ignore')
    categoricalDataset = df[categoricalColumns]
    del df

    le = preprocessing.LabelEncoder()
    for i in categoricalColumns:
        categoricalDataset[i] = le.fit_transform(categoricalDataset[i])
    
    for i in categoricalColumns:
        numDataset[i] = categoricalDataset[i]
    del categoricalDataset
    
    return numDataset

def scaler(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def min_max_scaler(df):
    scaler = MinMaxScaler()
    return scaler.fit_transform(df)

def reduce_csv(csv_file):
    df = dd.read_csv(csv_file)
    result = df[df.columns.intersection(target)]
    drop_columns = df.columns[df.isna().compute().sum()/ df.shape[0].compute() >= .2].tolist()
    df = df.drop(columns= target + ["Unnamed: 0"] + drop_columns, errors='ignore')
    df = df.fillna(df.mode().compute().iloc[0])

    x = getNumericDataset(df)
    columns = list(x.columns)
    df = x
    x = scaler(x)
    
    pca = PCA()
    pca.fit(x)

    to_reduce = {}
    for i in range(len(columns)):
        to_reduce[pca.explained_variance_ratio_[i]] = columns[i]
    to_reduce = OrderedDict(sorted(to_reduce.items(), reverse=True))

    reduced = []
    explain_ratio = 0
    for i in to_reduce:
        if explain_ratio < .8:
            reduced.append(to_reduce[i])
            explain_ratio += i
        else:
            del to_reduce
            break   

    result[reduced] = df[reduced]

    result.to_csv(csv_file, index=False, single_file=True)
    return str(list(result.columns))

def format_csv(csv_file, pca_columns):
    pca_columns = pca_columns.replace('\'', '').replace("[", "").replace("]", "").replace(",", "").split(" ")
    df = dd.read_csv(csv_file)
    df = df.drop(columns=["Unnamed: 0"], errors='ignore')
    df = df[df.columns.intersection(pca_columns)]
    
    x = df.drop(columns= target + ["Unnamed: 0"], errors='ignore')
    x = x.fillna(x.mode().compute().iloc[0])
    x = getNumericDataset(x)
    merge_columns = x.columns
    
    for i in merge_columns:
        df[i] =x[i]
    df.to_csv(csv_file, index=False, single_file=True)

def create_model(csv_file, model_file, working_dir):
    df = dd.read_csv(csv_file)
    y = df[df.columns.intersection(target)]
    y = y.fillna(y.mode().compute().iloc[0])
    y = y.replace(["yes", "YES", "Yes"], 1)
    y = y.replace(["no", "NO", "No"], 0)

    x = df.drop(columns= target + ["Unnamed: 0"], errors='ignore')
    # x = dd.from_array(scaler(df)).to_dask_array(lengths=True)

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state= 1)

    random_forest = RandomForestClassifier(max_depth = 3, random_state = 1)

    # kfold = KFold(n_splits = 5, random_state=42, shuffle=True)
    # cv_results = cross_val_score(random_forest, x,y, cv = kfold, scoring='accuracy', verbose = 0)
    
    random_forest = random_forest.fit(x_train, y_train)

    # Matriz de confusion
    y_pred = random_forest.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_plot = ConfusionMatrixDisplay(cm).plot()
    plt.savefig(working_dir + "/confusion_matrix.png")
    plt.close()

    dump(random_forest, model_file)

def plotChurnProfileMean(cluster, churn_profile, churn_profile_df, columns_to_drop):
    churn_profile_df = churn_profile_df.drop(columns = columns_to_drop, errors='ignore')
    names = churn_profile_df.columns
    if churn_profile_df.shape[0] > 0:
        x = churn_profile_df.mean()
    else:
        x = [0] * len(names)
    y = names
    plt.plot(x, y, marker = 'o')
    for i, value in enumerate(x):
        value = round(value,2)
        plt.annotate(value, (x[i], y[i]))
    plt.savefig(cluster + "/" + churn_profile + "_profile_mean_data.png", bbox_inches = "tight")
    plt.close()

# -----------------------------------------------
def make_clusters(file_path, file_name):
    if False:
        df = dd.read_csv(file_path + "/" + file_name + ".csv")

        cluster = []
        for i in range(4):
            cluster.append(file_path + "/cluster/cluster_" + str(i))

        for i in cluster:
            df.to_csv(i + "/cluster.csv", index=False, single_file=True)
        return [{"name": "cluster_1", "percentage": 25}, {"name": "cluster_2", "percentage": 25}, {"name": "cluster_3", "percentage": 25}, {"name": "cluster_4", "percentage": 25}]
    else:
        df = pd.read_csv(file_path + "/" + file_name + ".csv")
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
        km, y_km, km_labels = makeClusters(n, x)

        clusters = pd.DataFrame(data = {'cluster': y_km})

        # Une la clasificacion con los datos del dataset
        df_clusters = dd.merge(clusters, df.drop(columns=target, errors='ignore'), left_index=True, right_index=True)

        df_clusters = dd.merge(df_clusters, df[df.columns.intersection(target)], left_index=True, right_index=True)

        # sort the dataframe
        df_clusters = df_clusters.sort_values(by=['cluster'])

        #Graficacion de los cluster
        clusters_plot = df_clusters.drop(columns=['Unnamed: 0'], errors='ignore').groupby(by=['cluster']).mean().reset_index()
        clusters_plot = min_max_scaler(clusters_plot)
        plot_columns = df_clusters.columns
        clusters_plot = pd.DataFrame(data=clusters_plot, columns=plot_columns)
        fig = make_subplots(rows=n, cols=1, specs = list([[{"type" : "polar"}]] * n), 
        )
        for i in range (0,n):
            cluster_color = '#' + str(random.randint(100000,999999)) #random color
            #plot cluster
            fig.append_trace(
                go.Scatterpolar(
                    r = clusters_plot.loc[i].values,
                    theta = clusters_plot.columns,
                    fill = 'toself',
                    name = 'Cluster ' + str(i),
                    fillcolor = cluster_color, line = dict(color = cluster_color),
                    showlegend = True, opacity = 0.6
                ), row = i + 1, col = 1
            )
        fig.update_layout(height=2000, showlegend=True)
        fig.write_image(file_path + "/clusters_polar.png")

        info = []
        amount = 0
        for i in range (n):
            df_to_csv = df_clusters[df_clusters['cluster'] == i]
            df_to_csv = df_to_csv.drop(columns=['cluster'], errors='ignore')

            # Porpocion de los clusters
            clusters_amount = df_to_csv.shape[0]
            amount += clusters_amount 
            info.append({"name": "cluster_" + str(i + 1), "percentage": clusters_amount})
            
            # Guardado de los clusters
            df_to_csv = dd.from_pandas(df_to_csv,  npartitions=1)
            df_to_csv.to_csv(file_path + "/cluster/cluster_" + str(i) + "/cluster.csv", single_file=True)
        for i in info:
            i["percentage"] = i["percentage"] / amount
        
        return info

# cluster: archivo csv donde se encuentra un cluster
# cs1, cs2, cs3: rangos de churn segment
def make_perfiles(cluster, cs1, cs2, cs3, model_file):
    # Lee el cluster
    df = dd.read_csv(cluster + "/cluster.csv")
    x = df.drop(columns= target + ['Unnamed: 0'], errors='ignore')
    # Obtiene la probabilidad de churn de todos los elementos del cluster
    random_forest = load(model_file)
    proba_matrix = getChurnProbabilities(random_forest, x)

    # Segmenta los elementos del cluster
    segments = []
    perfil = ["permanent", "low", "mid", "high"]
    churn_bill_value = []
    segments = showProbabilities(cs1, cs2, cs3, proba_matrix, x.compute().reset_index(drop=True))

    names = x.columns.to_list()
    # names[0] = "CUSTOMER_ID"
    names.append('CHURN_PERCENTAGE')
    
    bill_amount = list(set(["BILL_AMOUNT", "Bill_amount", "bill_amount"]).intersection(names))
    i = 0
    for segment in segments:
        df = pd.DataFrame(segment, columns = names)

        # Obtencion de datos y graficacion de cada perfil
        if len(bill_amount) > 0:
            churn_bill_value.append((df[bill_amount[0]] * df['CHURN_PERCENTAGE']).sum())
        plotChurnProfileMean(cluster, perfil[i], df, ["CUSTOMER_ID", "Unnamed: 0"])

        # Guardado de perfil
        df.to_csv(cluster +  "/" + str(i) + ".csv")
        i += 1

    # Graficacion de todos los perfiles
    if len(bill_amount) > 0:
        perfil = ["permanente","bajo","medio","alto"]
        plt.bar(perfil, churn_bill_value)
        plt.title('Valor monetario de cada perfil de churn')
        plt.ylabel('Bill amount')
        plt.xlabel('Perfil de churn')
        for i in range (0, len(perfil)):
            plt.annotate("$" + str(round(churn_bill_value[i], 2)),(i,i), xytext = (0,10),textcoords="offset points", ha = "center")
        plt.savefig(cluster + "/churn_profile_bill_amount.png")
        plt.close()

def make_perfiles_info(cluster):
    info = [{}, {}, {}, {}]
    for i in range(4):
        df = dd.read_csv(cluster + "/" + str(i) + ".csv")
        df_colums = list(df.columns)

        AMOUNT = df.shape[0].compute()
        
        if "BILL_AMOUNT" in df_colums:
            bill_amount = float(df['BILL_AMOUNT'].sum().compute())
        else:
            bill_amount = "-1"

        if "PREPAID_LINES" in df_colums:
            prepaid_lines = float(df['PREPAID_LINES'].sum().compute())
        else:
            prepaid_lines = "-1"
        
        if "POSTPAID_LINES" in df_colums:
            postpaid_lines = float(df['POSTPAID_LINES'].sum().compute())
        else:
            postpaid_lines = "-1"
        
        if "OTHER_LINES" in df_colums:
            other_lines = float(df['OTHER_LINES'].sum().compute())
        else:
            other_lines = "-1"
        
        if "PARTY_REV" in df_colums:
            party_rev = float(df['PARTY_REV'].sum().compute())
        else:
            PARTY_REV = "-1"

        info[i] = {
            "amount" : AMOUNT,
            "bill amount" : bill_amount,
            "lines" : [{"type" : "PREPAID_LINES", "amount" : prepaid_lines},{"type" : "POSTPAID_LINES", "amount" : postpaid_lines}, {"type" : "OTHER_LINES", "amount" : other_lines}],
            "revenues" : party_rev
        }

    return info
