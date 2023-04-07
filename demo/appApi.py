import os
import logging
import sys
import torch
from app_modules.utils import *
from app_modules.presets import *
from app_modules.overwrites import *
from waitress import serve

from flask import Flask, request, jsonify
import argparse
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

base_model="decapoda-research/llama-7b-hf"
lora_model="project-baize/baize-lora-7B"
adapter_model = "project-baize/baize-lora-7B"
tokenizer, model, device = load_tokenizer_and_model(base_model, adapter_model)

app = Flask(__name__)

def predict(
    text,
    chatbot,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
):
    if text == "":
        yield chatbot, history, "Empty context."
        return

    inputs = generate_prompt_with_history(
        text, history, tokenizer, max_length=max_context_length_tokens
    )
    if inputs is None:
        yield chatbot, history, "Input too long."
        return
    else:
        prompt, inputs = inputs
        begin_length = len(prompt)
    input_ids = inputs["input_ids"][:, -max_context_length_tokens:].to(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for x in sample_decode(
            input_ids,
            model,
            tokenizer,
            stop_words=["[|Human|]", "[|AI|]"],
            max_length=max_length_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            if is_stop_word_or_prefix(x, ["[|Human|]", "[|AI|]"]) is False:
                if "[|Human|]" in x:
                    x = x[: x.index("[|Human|]")].strip()
                if "[|AI|]" in x:
                    x = x[: x.index("[|AI|]")].strip()
                x = x.strip(" ")
                a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
                    [text, convert_to_markdown(x)]
                ], history + [[text, x]]
                yield a, b, "Generating..."
            if shared_state.interrupted:
                shared_state.recover()
                try:
                    yield a, b, "Stop: Success"
                    return
                except:
                    pass
    torch.cuda.empty_cache()
    print(prompt)
    print(x)
    print("=" * 80)
    try:
        yield a, b, "Generate: Success"
    except:
        pass

@app.route('/predict', methods=['GET'])
def api_predict():
    text = request.args.get('text')
    if not text:
        return jsonify({"error": "Text not provided"}), 400

    top_p = 0.95
    temperature = 1
    max_length_tokens = 512
    max_context_length_tokens = 2048
    history = []

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
    parser.add_argument('-p','--port', help='Port number', required=False) 
    args = vars(parser.parse_args())
    if 'port' in args:
        portToUse = args['port']
        print ("ChatGPT flask app running on port :: ", portToUse)

    serve(app, host='0.0.0.0', port=portToUse, threads=8)
    # app.run(debug=True)
