import flask
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


import numpy as np
import json
from basic_model_redacted import *
import os

DEBUG = bool(int(os.environ.get('DEBUG', 1)))

FLASK_PORT = int(os.environ.get('FLASK_PORT', 9897))


@app.route('/api/predict_basic',methods=['POST'])
@cross_origin()
def predict_basic():
    payload = {}

    data = request.get_json()
    obs = np.asarray(data['obs'])
    valid_moves = data['valid_moves']
    transcript = data['transcript']

    action, value = model.action_value(obs[None, :], valid_moves)

    resp = {
        'action':int(action)
    }
    print(resp)

    return json.dumps(resp)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=FLASK_PORT, threaded=DEBUG)


def jsonify(obj, status=200, headers=None):
    """ Custom JSONificaton to support obj.to_dict protocol. """
    data = NpEncoder().encode(obj)
    if 'callback' in request.args:
        cb = request.args.get('callback')
        data = '%s && %s(%s)' % (cb, cb, data)
    return Response(data, headers=headers, status=status,
                    mimetype='application/json')
