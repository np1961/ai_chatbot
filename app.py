
from flask import Flask, jsonify, request
from helpers import RAG

chatbot = RAG()
app = Flask(__name__)

@app.route('/load_model', methods=['POST'])
def load_model():
    """
    No input
    """

    res = chatbot.embedding.load_embeddings(amount=3)
    return jsonify(res)

@app.route('/search', methods=['POST'])
def search():
    """
    input_question - str
    """

    if "input_question" not in request.form:
        return jsonify({
            "status": 500, "message": "input_question not defined"}), 500
    input_question = request.form['input_question']
    res = chatbot.embedding.ask_question(input_question=input_question)
    return jsonify(res)


@app.route('/search_and_generate', methods=['POST'])
def llm_generate():
    """
    input_question - str
    """

    if "input_question" not in request.files:
        return jsonify({
            "status": 500, "message": "input_question not defined"}), 500
    input_question = request.files['input_question'].readlines()


    res = chatbot.embedding.ask_question(input_question=input_question)
    return jsonify(res)

    #TODO
    # add llm model gpt3.5

    return "<h1>Not implemented</h1>"


@app.route('/document_search', methods=['POST'])
def document_search():
    """
    input_question - str
    """

    # request.files
    return "<h1>Not implemented</h1>"


@app.route('/', methods=['GET'])
def start():
    return "<h1>Hello this is FAQ api</h1>"


if __name__ == '__main__':
    app.run(debug=True)
