from flask import Flask, jsonify, redirect
from flask_cors import CORS
from waitress import serve

from lib import model, system

# from predictions import makePrediction


app = Flask(__name__)
CORS(app)
SERVER_FILE = __file__

# ///////////////////////////////////////////////

@app.route('/')
def hello_world():
    return jsonify(hello = "Wolrd")

# Manejo de archivos
# -----------------------------------------------

# Pasas el nombre del arcivo, y la ubicacion donde se encuentra
# Ejemplo
# /upload/hola/%20/mundo
@app.route('/upload/<string:name>/<path:working_file>')
def upload(name, working_file):
    return system.new_dir(SERVER_FILE, name, working_file) #comentar

# Obtiene la ubicacion del csv con el que se esta trabajando
@app.route('/get_working_file/')
def get_working_file():
    server_file = system.get_server_dir(SERVER_FILE) + "/server.json"
    return jsonify({"working_file" : system.read_json_file(server_file)["working_file"]})

# Manejo de datos
# -----------------------------------------------
#???
@app.route('/get_grupo_list/')
def get_grupo_list():
    server_file = system.get_server_dir(SERVER_FILE) + "/server.json"
    about_file = system.read_json_file(server_file)["working_dir"] + "/about.json"
    grupo_list = system.read_json_file(about_file)["group"]
    
    return jsonify({"grupo" : grupo_list})

# @app.route('/get_grupo/<string:nombre>')
# def get_grupo(nombre):
#     return nombre

# @app.route('/get_grupo_distribution')
# def get_grupo_distribution():
#     return " "

# ...............................................
# [x1, x2]
@app.route('/set_churn_segment/<int:x1>/<int:x2>/<int:x3>')
def set_churn_segment(x1, x2, x3):
    server_file = system.get_server_dir(SERVER_FILE) + "/server.json"
    about_file = system.read_json_file(server_file)["working_dir"] + "/about.json"
    about_file = "/home/alt9193/Documents/IA/DeepLearningBackend/uploads/hola/about.json"
    system.rewrite_json_file(about_file, "churn_segment", [x1, x2, x3])
    return about_file

# Regresa una lista con los porcentaje de los rangos de churn
@app.route('/get_churn_segment/')
def get_churn_segment():
    server_file = system.get_server_dir(SERVER_FILE) + "/server.json"
    about_file = system.read_json_file(server_file)["working_dir"] + "/about.json"
    churn_segment = system.read_json_file(about_file)["churn_segment"]
    
    return jsonify({"churn_segment" : churn_segment})

# ...............................................



# ///////////////////////////////////////////////

if __name__ == '__main__':
    # app.run(debug=True)
    serve(app, host='0.0.0.0', port=8080)