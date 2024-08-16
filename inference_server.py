from flask import Flask, send_file, make_response, request
from flask_cors import CORS, cross_origin
from inferer import LLMInfer

app = Flask(__name__)
cors = CORS(app)


@app.route("/get_config", methods=['POST'])
def get_configuration():
    print('request received' + str(request.get_json()))
    data = request.get_json()
    user_prompt = data['user_prompt']
    print('user prompt ' + str(user_prompt))
    config = Inferer.infer(user_prompt)
    print('configuration ' + str(config))
    return config

@app.route("/get_model_architecture", methods=['POST'])
def get_model_architecture():
    print('request received' + str(request.get_json()))
    data = request.get_json()
    print('data type ' + str(type(data)))
    config = data['config']
    print('config type ' + str(type(config)))
    model_arch = Inferer.infer_model(config)
    print('model architecture ' + str(model_arch))
    return model_arch


if __name__ == '__main__':
    Inferer = LLMInfer()
    Inferer.load_model()
    app.run(debug=True, host='0.0.0.0', port=8000)
