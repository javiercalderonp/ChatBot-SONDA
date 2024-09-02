import os
import base64
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from langchain_ibm import WatsonxLLM
from dotenv import load_dotenv

app = Flask(__name__)

# Cargar variables de entorno
load_dotenv()

# Cargar el documento TXT
loader = TextLoader("inputs/Poliza.txt")
documents = loader.load()

# Ajustar la segmentación del texto
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1900,  # Fragmentos más pequeños para respuestas más precisas
    chunk_overlap=300,
)
split_documents = text_splitter.split_documents(documents)

# Cargar el modelo SentenceTransformer para embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Definir una clase para envolver el modelo y obtener embeddings
class EmbeddingsWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts)
    
    def __call__(self, text):
        return self.model.encode([text])[0]

# Crear una instancia de la clase de envoltura
embedding_wrapper = EmbeddingsWrapper(model)

# Crear embeddings para los documentos divididos
embeddings = embedding_wrapper.embed_documents([d.page_content for d in split_documents])

# Crear una vectorstore usando FAISS
vectorstore = FAISS.from_texts(texts=[d.page_content for d in split_documents], embedding=embedding_wrapper)

# Configurar el retriever
retriever = vectorstore.as_retriever(search_type="similarity", k=3)  # Ajustar el tipo de búsqueda para mejorar la relevancia

# Configurar parámetros del modelo LLM (puedes cambiar a Mistral si lo prefieres)
watsonx_url = os.getenv("WATSONX_URL")
watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
watsonx_api_key = os.getenv("WATSONX_API_KEY")

parameters = {
    "decoding_method": "greedy",
    "temperature": 0.10,
    "top_p": 1,
    "top_k": 1,
    "min_new_tokens": 5,
    "max_new_tokens": 300,
    "repetition_penalty": 1,
    "stop_sequences": [],
    "return_options": {
        "input_tokens": True,
        "generated_tokens": True,
        "token_logprobs": True,
        "token_ranks": True,
    },
}

llm = WatsonxLLM(
    model_id="meta-llama/llama-3-70b-instruct",
    #model_id= "mistralai/mixtral-8x7b-instruct-v01",
    url=watsonx_url,
    apikey=watsonx_api_key,
    project_id=watsonx_project_id,
    params=parameters
)

# Definir un prompt template más específico
template = """sistema
Analiza el documento proporcionado y responde las preguntas del usuario utilizando únicamente la información encontrada en el documento.

Contexto: {contexto}

usuario

Pregunta: {pregunta}

Asistente: 

"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

chain = (
    {"contexto": retriever | format_docs, "pregunta": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route('/')
def index():
    # Renderizar la página principal
    return render_template('index.html')

# Almacenar el historial de la conversación
conversation_history = []

# Función para actualizar y recuperar el historial relevante
def get_conversation_context(question, conversation_history, max_turns=2):
    conversation_context = ""
    if len(conversation_history) > 0:
        relevant_history = conversation_history[-max_turns:]
        for entry in relevant_history:
            conversation_context += f"Pregunta: {entry['question']}\nRespuesta: {entry['answer']}\n"
    return conversation_context

# En la función `ask_question`, antes de enviar la pregunta al LLM:
@app.route('/ask', methods=['GET'])
def ask_question():
    question = request.args.get('question')

    # Obtener el contexto relevante de la conversación
    context = get_conversation_context(question, conversation_history)

    # Obtener documentos relevantes
    relevant_docs = retriever.invoke(question)
    context += format_docs(relevant_docs)
    
    def generate_response():
        full_response = ""
        for word in chain.stream({"contexto": context, "pregunta": question}):
            full_response += word
            yield f"data: {word}\n\n"
        
        # Actualizar el historial de la conversación
        conversation_history.append({"question": question, "answer": full_response})
    
    return Response(generate_response(), content_type='text/event-stream')

# Sirviendo la carpeta 'images' como estática
@app.route('/images/<path:filename>')
def custom_static(filename):
    return send_from_directory('images', filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
