from flask import Flask, jsonify
from flask_cors import CORS
from waitress import serve

from lib import file_manager as fm


app = Flask(__name__)
CORS(app)
SERVER_FILE = __file__

def code_200(): 
    return jsonify(success=True)

def update_server():
    server_file = fm.get_server_dir(SERVER_FILE) + "/server.json"
    server = fm.read_json_file(server_file)    
    return server

def get_about():
    server = update_server()
    return server["working_dir"] + "/about.json"


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
    fm.new_dir(SERVER_FILE, name, working_file) #comentar
    return code_200()

# Obtiene la ubicacion del csv con el que se esta trabajando
# @app.route('/get_working_file/')
# def get_working_file():
#     return jsonify({"working_file" : fm.read_json_file(server_file)["working_file"]})

# Manejo de datos
# -----------------------------------------------
#???
@app.route('/get_grupo_list/')
def get_grupo_list():
    grupo_list = fm.read_json_file(get_about())["group"]
    
    return jsonify({"grupo" : grupo_list})

# ...............................................
@app.route('/cluster/<string:name>')
def get_cluster(name):
    return jsonify({"perfil" : []})

@app.route('/cluster/<string:name>/perfil/<string:perfil>')
def get_perfil_from_cluster(name, perfil):
    return jsonify({})

# ...............................................
# [x1, x2, x3]
@app.route('/set_churn_segment/<int:x1>/<int:x2>/<int:x3>')
def set_churn_segment(x1, x2, x3):
    fm.rewrite_json_file(get_about(), "churn_segment", [x1, x2, x3])
    return code_200()

# Regresa una lista con los porcentaje de los rangos de churn
@app.route('/get_churn_segment/')
def get_churn_segment():
    churn_segment = fm.read_json_file(get_about())["churn_segment"]
    
    return jsonify({"churn_segment" : churn_segment})

# ///////////////////////////////////////////////

if __name__ == '__main__':
    # app.run(debug=True)
    serve(app, host='0.0.0.0', port=8080)