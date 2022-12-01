import os
import json
import csv
import zlib
import shutil
from . import model

# Utilidades para manejar los jsons
# -----------------------------------------------

def write_json_file(json_file, data):
    with open(json_file + "", 'w') as outfile:
        json.dump(data, outfile)

def read_json_file(json_file):
    with open(json_file) as read_file:
        data = json.load(read_file)
    return data

def rewrite_json_file(json_file, attribute, new_data):
    data = {}
    with open(json_file) as read_file:
        data = json.load(read_file)
    data[attribute] = new_data
    write_json_file(json_file, data)

def json_to_csv(data, csv_file):
    data_file = open(csv_file, 'w', newline='')
    csv_writer = csv.writer(data_file)

    # Nombres de columnas
    csv_writer.writerow(data["rows"][0])

    # Informacion de columnas
    for row in data["rows"][1:]:
        if row != [""]:
            csv_writer.writerow(row)
    data_file.close()

    return str(data["rows"][0])

def check_file(file):
    return os.path.exists(file)
    
def name_model():
    new_model = hex(len(os.listdir("./models")) + 1)
    return new_model

# Utilidades para el servidor
# -----------------------------------------------
def get_server_dir(SERVER_FILE):
    return os.path.dirname(os.path.realpath(SERVER_FILE))

def write_server_file(SERVER_FILE, working_dir):
    working_data = {
        "working_dir" : working_dir,
        
    }

    server_file = get_server_dir(SERVER_FILE) + "/server.json"
    write_json_file(server_file, working_data)

# Utilidad para generar los perfiles de todos los cluster
# -----------------------------------------------
def set_churn_segment(x1, x2, x3, working_dir, model_file):
    clusters_dir = working_dir + "/cluster/"
    clusters = os.listdir(clusters_dir)

    for i in clusters:
        model.make_perfiles(clusters_dir + i, x1, x2, x3, model_file)
    
    for i in clusters:
        cluster = clusters_dir + i
        write_json_file(cluster + "/perfil.json", model.make_perfiles_info(cluster))

# Utilidad para manejar un nuevo projecto
# -----------------------------------------------
def new_dir(SERVER_FILE, name, data):
    # Iniciacion
    working_dir = get_server_dir(SERVER_FILE) + "/uploads/" + name
    if check_file(working_dir):
        shutil.rmtree(working_dir)
    os.makedirs(working_dir, exist_ok=True)
    write_server_file(SERVER_FILE, working_dir)

    # Subida del csv
    csv_file = working_dir + "/" + name + ".csv"
    csv_format = json_to_csv(data, csv_file)
    del data
    print("Csv guardado")

    model_file=""
    # Comprobacion de un modelo existente
    model_trainded = read_json_file("./model.json")
    if csv_format in model_trainded["model"]:
        model_file = model_trainded["model"][csv_format]
    else:
        if csv_format in model_trainded["format"]:
            pca_columns = model_trainded["format"][csv_format]
            model.format_csv(csv_file, pca_columns)
            model_file = model_trainded["model"][pca_columns]
        else:
            # PCA
            pca_columns = model.reduce_csv(csv_file)
            model_trainded["format"][csv_format] = pca_columns
            print("PCA finalizado")
            # Traing model
            if not os.path.exists("./models"):
                os.mkdir("./models")
            model_file = "./models/" + name_model() +".joblib"
            model.create_model(csv_file, model_file, working_dir)
            model_trainded["model"][pca_columns] = model_file
            write_json_file("./model.json", model_trainded)
            print("Modelo entrenado")
    
    # Calculo de churn con el modelo
    # Creacion de clusters
    clusters = model.make_clusters(working_dir, name)
    # "churn_segment" : [porcentaje1, porcentaje2, porcentaje3],
    # sort growing / decreasing
    # "group" : [{"name" : "uno",  "percentage": 20}]
    about = {
        "model_file" : model_file,
        "churn_segment" : [.20, .50, .70],
        "sort" : "growing",
        "group" : clusters
    }
    write_json_file(working_dir + "/about.json", about)

    # Creacion de perfiles
    set_churn_segment(20, 50, 70, working_dir, model_file)