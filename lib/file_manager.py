import os
import json
import csv

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
        csv_writer.writerow(row)
    data_file.close()

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

# Utilidad para manejar un nuevo projecto
# -----------------------------------------------
def new_dir(SERVER_FILE, name, data):
    working_dir = get_server_dir(SERVER_FILE) + "/uploads/" + name
    os.makedirs(working_dir, exist_ok=True)
    write_server_file(SERVER_FILE, working_dir)

    csv_file = working_dir + "/" + name + ".csv"
    json_to_csv(data, csv_file)
    del data
    
    clusters = model.make_clusters(working_dir, name)
    # "churn_segment" : [porcentaje1, porcentaje2, porcentaje3],
    # sort growing / decreasing
    # "group" : [{"name" : "uno",  "percentage": 20}]
    about = {
        "churn_segment" : [.20, .50, .70],
        "sort" : "growing",
        "group" : []
    }

    write_json_file(working_dir + "/about.json", about)

    for i in clusters:
        model.make_perfiles(i, .20,.50, .70)
    
    for i in clusters:
        # print(model.make_perfiles_info(i))
        write_json_file(i + "/perfil.json", model.make_perfiles_info(i))