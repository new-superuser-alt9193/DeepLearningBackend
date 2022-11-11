import os
from pathlib import Path
import json

def get_server_dir(SERVER_FILE):
    return os.path.dirname(os.path.realpath(SERVER_FILE))

def write_server_file(SERVER_FILE, working_file, working_dir):
    working_data = {
        "working_file" : working_file,
        "working_dir" : working_dir,
        "orden" : "normal"
    }

    server_file = get_server_dir(SERVER_FILE) + "/server.json"
    with open(server_file, 'w') as outfile:
        json.dump(working_data, outfile)

def read_json_file(json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
    return data

def new_dir(SERVER_FILE, name, working_file):
    working_dir = get_server_dir(SERVER_FILE) + "/uploads/" + name
    os.makedirs(working_dir + "/plot", exist_ok=True)
    Path(working_dir + "/about.json").touch()

    write_server_file(SERVER_FILE, working_file, working_dir)

    return working_dir + " " + working_file