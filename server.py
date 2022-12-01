from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from waitress import serve

from lib import file_manager as fm


app = Flask(__name__)
CORS(app)
SERVER_FILE = __file__

def code_200(): 
    return jsonify(success=True)

def code_404(): 
    return jsonify(success=False)

def update_server():
    server_file = fm.get_server_dir(SERVER_FILE) + "/server.json"
    server = fm.read_json_file(server_file)    
    return server

def get_about():
    server = update_server()
    return server["working_dir"] + "/about.json"

def get_working_dir():
    server = update_server()
    return server["working_dir"]


# ///////////////////////////////////////////////

@app.route('/')
def hello_world():
    return jsonify(hello = "Wolrd")

# Manejo de archivos
# -----------------------------------------------

# Pasas el nombre del arcivo, y la ubicacion donde se encuentra
# Ejemplo
# /upload/hola/%20/mundo
@app.route('/upload/<string:name>', methods=['POST'])
def upload(name):
    json_data = request.get_json()
    fm.new_dir(SERVER_FILE, name, json_data)
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
    return jsonify({"perfil" : fm.read_json_file(get_working_dir()+ "/cluster/" + name + "/perfil.json")})

# ...............................................
# [x1, x2, x3]
@app.route('/set_churn_segment/<int:x1>/<int:x2>/<int:x3>')
def set_churn_segment(x1, x2, x3):
    x1 = float(x1)/100
    x2 = float(x2)/100
    x3 = float(x3)/100

    fm.set_churn_segment(x1, x2, x3, get_working_dir())
    fm.rewrite_json_file(get_about(), "churn_segment", [x1, x2, x3])
    return code_200()

# Regresa una lista con los porcentaje de los rangos de churn
@app.route('/get_churn_segment/')
def get_churn_segment():
    churn_segment = fm.read_json_file(get_about())["churn_segment"]
    
    return jsonify({"churn_segment" : churn_segment})

# ...............................................
@app.route('/image/matrix/')
def get_matrix_image():
    working_dir = get_working_dir()
    filename = working_dir + '/confusion_matrix.png'
    if fm.check_file(filename):
        return send_file(filename, mimetype='image/png')
    else:
        return code_404()

@app.route('/image/cluster/<string:name>/<string:perfil>')
def get_clusters_image(name, perfil):
    working_dir = get_working_dir()
    filename = ""
    if perfil == "all":
        filename = working_dir + "/cluster/" + name + '/churn_profile_bill_amount.png'
    elif perfil == "polar":
        filename = working_dir + "/cluster/" + name + '/cluster.png'
    else:
        filename = working_dir + "/cluster/" + name + "/" +  perfil + "_profile_mean_data.png"
    return send_file(filename, mimetype='image/png')
# ///////////////////////////////////////////////

if __name__ == '__main__':
    app.run(debug=True, port=8080)
    # serve(app, host='0.0.0.0', port=8080)