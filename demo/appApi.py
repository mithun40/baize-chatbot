import os
import logging
import sys
import torch
from app_modules.utils import *
from app_modules.presets import *
from app_modules.overwrites import *

from flask import Flask, request, jsonify
import argparse
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

# base_model = sys.argv[1]
base_model="decapoda-research/llama-7b-hf"
lora_model="project-baize/baize-lora-7B"
adapter_model = "project-baize/baize-lora-7B"
tokenizer, model, device = load_tokenizer_and_model(base_model, adapter_model)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text not provided"}), 400

    # Set your parameters as needed
    top_p = 0.95
    temperature = 1
    max_length_tokens = 512
    max_context_length_tokens = 2048
    history = []

    # Get the chatbot response
    response = ""
    for chatbot, h, status in predict(
        text,
        [],
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
    ):
        history = h
        response = status

    return jsonify({"response": response})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='List of arguments')
    parser = argparse.ArgumentParser(description='List of arguments')
    parser.add_argument('-p','--port', help='Port number', required=False) 
    args = vars(parser.parse_args())
    if 'port' in args:
        portToUse = args['port']
        print ("ChatGPT flask app running on port :: ", portToUse)

    serve(app, host='0.0.0.0', port=portToUse, threads=8)
    # app.run(debug=True)
