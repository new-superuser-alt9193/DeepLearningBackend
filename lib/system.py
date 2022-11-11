import os
from pathlib import Path
import json

def get_server_dir(SERVER_FILE):
    return os.path.dirname(os.path.realpath(SERVER_FILE))

def write_server_file(SERVER_FILE, working_file, working_dir):
    working_data = {
        "working_file" : working_file,
        "working_dir" : working_dir,
        
    }

    server_file = get_server_dir(SERVER_FILE) + "/server.json"
    with open(server_file, 'w') as outfile:
        json.dump(working_data, outfile)

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

def new_dir(SERVER_FILE, name, working_file):
    working_dir = get_server_dir(SERVER_FILE) + "/uploads/" + name
    os.makedirs(working_dir + "/plot", exist_ok=True)
    Path(working_dir + "/about.json").touch() #comentar

    # "churn_segment" : [porcentaje1, porcentaje2],
    # sort growing / decreasing
    # "group" : [{"name" : "uno",  "percentage": 20}]
    about = {
        "churn_segment" : [20, 50],
        "sort" : "growing",
        "group" : []
    }

    write_json_file(working_dir + "/about.json", about)
    write_server_file(SERVER_FILE, working_file, working_dir)

    return working_dir + " " + working_file