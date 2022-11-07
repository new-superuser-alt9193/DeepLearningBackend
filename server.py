
from flask import Flask, redirect, jsonify
from waitress import serve
# from predictions import makePrediction
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return jsonify(hello = "Wolrd")

@app.route('/upload/<path:path>')
def upload(path):
    return path

@app.route('/get_working_file')
def get_working_file():
    return "file"

@app.route('/get_grupo_list/<string:orden>')
def get_grupo_list(orden):
    return orden

@app.route('/get_grupo/<string:nombre>')
def get_grupo_list(nombre):
    return nombre

@app.route('/get_grupo_distribution')
def get_grupo_distribution():
    return " "

@app.route('/churn_segment/<float:x1>/<float:x2>')
def churn_segment(x1, x2):
    return x1 + x2

@app.route('/get_churn_segment')
def get_churn_segment():
    return "orden"

@app.route('/plot/<string:pltType>/<string:x>/<string:y>')
def plot(pltType, x, y):
    return pltType + " " + x + " " + y


# @app.route('/csv', methods = ['POST', 'GET'])
# def csv(csv):
#     return makePrediction(csv)


if __name__ == '__main__':
    # app.run(debug=True)
    serve(app, host='0.0.0.0', port=8080)