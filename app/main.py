import os 
from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify, make_response
from .utils.rag import * #uncomment this to run using docker
# from utils.rag import * #uncomment this to run locally


app = Flask(__name__, static_url_path='/static', template_folder='templates/')
    

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/rag")
def rag():
    return render_template('rag-qna.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    storage = request.form.get('storage')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file_path = 'static/'+ file.filename
    file.save(file_path)

    print("Upload knowledge store to: ", storage)

    if storage =='wd':
        response =load_docs(file_path)
        print(response)
    else:
        collection_name = "collection"
        embed_pdf_text(document_id="policy", file_path=file_path, milvus_connection_alias="default", collection_name=collection_name)
        build_vector_index(milvus_connection_alias="default", collection_name=collection_name)

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    else:
        print(f"The file {file_path} does not exist.")
    
    response_data = {"output": "milvus received file", "file_name": file.filename}
    return jsonify(response_data)



@app.route('/qna_wx', methods=['POST'])
def ask_wxai():
    data = request.get_json()
    user_question = data.get('user_question')
    db_selection = data.get('db')
    llm_model = data.get('llm')
    answer, eta_retrieve, eta_wxai = query_wxai(user_question, db_selection, llm_model)
    response_data = {"answer": answer, "eta_retrieve":eta_retrieve , "eta_wxai":eta_wxai}
    return jsonify(response_data)



@app.route('/qna_wx_stream', methods=['POST'])
def ask_wxai_stream():
    data = request.get_json()
    user_question = data.get('user_question')
    db_selection = data.get('db')
    llm_model = data.get('llm')
    return Response(query_wxai(user_question, db_selection, llm_model,streaming=True), content_type='text/event-stream')
    

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
