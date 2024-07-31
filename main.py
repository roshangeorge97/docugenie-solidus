import time
import datetime
import uuid
import uvicorn
import requests
from bs4 import BeautifulSoup as Soup
import gradio as gr
from fastapi import FastAPI, Request, Header, HTTPException
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.utils import filter_complex_metadata
from utils.status_codes import StatusCodes
from utils.version import API_VERSION, SERVICE_NAME
from api.models import Request, LangChainRequestCall, UploadRequestCall, TaskRequest
from fastapi import FastAPI, Request, Header
import time
import uuid
from dotenv import load_dotenv
import os

# Constants
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
FAISS_INDEX_PATH = "./vectorstore/lc-faiss-multi-mpnet-500"
SUPPORTED_METHOD = ["ask", "upload_links"]

# Initialize FastAPI app
app = FastAPI()
task_status = {}
task_results = {} 
    
# Helper functions
def response_template(request_id: str, 
                      trace_id: str, 
                      process_duration: int,
                      isResponseImmediate: bool,
                      response: dict,
                      error_code: dict):
    now = datetime.datetime.now().isoformat()
    response_data = {
        "requestId": request_id,
        "traceId": trace_id,
        "apiVersion": API_VERSION,
        "service": SERVICE_NAME,
        "datetime": now,
        "isResponseImmediate": isResponseImmediate,
        "processDuration": process_duration,
        "response": response,
        "errorCode": error_code,
    }
    return response_data

# LangChain setup
urls = ["https://langchain-ai.github.io/langgraph/#example"]
docs = []
for url in urls:
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=9,
        extractor=lambda x: Soup(x, "html.parser").text
    )
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.create_documents(
    [doc.page_content for doc in docs], 
    metadatas=[doc.metadata for doc in docs]
)

model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

db = FAISS.from_documents(filter_complex_metadata(chunks), embeddings)
db.save_local(FAISS_INDEX_PATH)

model_id = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 1024,
        "repetition_penalty": 1.2,
        "return_full_text": False
    },
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

template = """
You are the friendly documentation buddy Solido, who helps novice programmers in using LangChain with simple explanations and examples.\
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question :
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""

prompt = PromptTemplate.from_template(template=template)
memory = ConversationBufferMemory(memory_key="history", input_key="question")

qa = RetrievalQA.from_chain_type(
    llm=model_id,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True,
    chain_type_kwargs={
        "verbose": True,
        "memory": memory,
        "prompt": prompt
    }
)

# API routes
@app.post("/ask")
async def ask_question(request: LangChainRequestCall):
    start_time = time.time()
    try:
        history = [(q, a) for q, a in request.request.payload.history if q or a]
        response = qa({"query": request.request.payload.query, "history": history})
        process_duration = int((time.time() - start_time) * 1000)
        result = response.get('result', response.get('answer', str(response)))
        
        return response_template(
            request_id=request.requestId,
            trace_id=str(uuid.uuid4()),
            process_duration=process_duration,
            isResponseImmediate=True,
            response={"result": result},
            error_code={}
        )
    except Exception as e:
        process_duration = int((time.time() - start_time) * 1000)
        return response_template(
            request_id=request.requestId,
            trace_id=str(uuid.uuid4()),
            process_duration=process_duration,
            isResponseImmediate=True,
            response={},
            error_code={"code": "ERROR", "message": str(e)}
        )

@app.post("/upload_links")
async def upload_links(request: UploadRequestCall):
    start_time = time.time()
    try:
        links = request.request.payload.urls
        global docs, chunks, db, retriever, qa
        
        docs = []
        for url in links:
            loader = RecursiveUrlLoader(url=url, max_depth=9, extractor=lambda x: Soup(x, "html.parser").text)
            docs.extend(loader.load())
        
        chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
        
        db = FAISS.from_documents(filter_complex_metadata(chunks), embeddings)
        db.save_local(FAISS_INDEX_PATH)
        
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=model_id,
            chain_type="stuff",
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": True,
                "memory": memory,
                "prompt": prompt
            }
        )
        
        process_duration = int((time.time() - start_time) * 1000)
        return response_template(
            request_id=request.requestId,
            trace_id=str(uuid.uuid4()),
            process_duration=process_duration,
            isResponseImmediate=True,
            response={"message": f"Processed {len(docs)} documents from {len(links)} URLs"},
            error_code={}
        )
    except Exception as e:
        process_duration = int((time.time() - start_time) * 1000)
        return response_template(
            request_id=request.requestId,
            trace_id=str(uuid.uuid4()),
            process_duration=process_duration,
            isResponseImmediate=True,
            response={},
            error_code={"code": "ERROR", "message": str(e)}
        )

def response_template(request_id: str, trace_id: str, process_duration: int,
                      isResponseImmediate: bool, response: dict, error_code: dict):
    now = datetime.datetime.now().isoformat()
    return {
        "requestId": request_id,
        "traceId": trace_id,
        "apiVersion": "1.0",
        "service": "LangChainQA",
        "datetime": now,
        "isResponseImmediate": isResponseImmediate,
        "processDuration": process_duration,
        "response": response,
        "errorCode": error_code,
    }


@app.post("/call")
async def call_endpoint(request: Request, x_user_id: str = Header(None)):
    start_time = time.time()
    request_data = await request.json()

    if not x_user_id:
        return response_template(
            request_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            process_duration=-1,
            isResponseImmediate=True,
            response={},
            error_code={"status": StatusCodes.INVALID_REQUEST, "reason": "userToken is invalid"}
        )

    request_id = request.headers.get('x-request-id')
    if not request_id:
        return response_template(
            request_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            process_duration=-1,
            isResponseImmediate=True,
            response={},
            error_code={"status": StatusCodes.INVALID_REQUEST, "reason": "requestId is invalid"}
        )

    method = request_data.get('method')
    if not method:
        return response_template(
            request_id=request_id,
            trace_id=str(uuid.uuid4()),
            process_duration=-1,
            isResponseImmediate=True,
            response={},
            error_code={"status": StatusCodes.INVALID_REQUEST, "reason": "method is invalid"}
        )
    elif method not in SUPPORTED_METHOD:
        return response_template(
            request_id=request_id,
            trace_id=str(uuid.uuid4()),
            process_duration=-1,
            isResponseImmediate=True,
            response={},
            error_code={"status": StatusCodes.UNSUPPORTED, "reason": f"unsupported method {method}"}
        )

    task_id = str(uuid.uuid4())
    task_status[task_id] = StatusCodes.PENDING  # Set initial status as PENDING

    process_duration = int((time.time() - start_time) * 1000)

    # Start a background thread to process the request
    threading.Thread(target=process_task, args=(task_id, request_data, x_user_id, request_id)).start()

    return response_template(
        request_id=request_id,
        trace_id=str(uuid.uuid4()),
        process_duration=process_duration,
        isResponseImmediate=False,
        response={"taskId": task_id},
        error_code={"status": StatusCodes.PENDING, "reason": "Task is pending"}
    )

# Define your task processing function
def process_task(task_id, request_data, user_id, request_id):
    start_time = time.time()
    task_status[task_id] = StatusCodes.INPROGRESS  # Update status to INPROGRESS

    method = request_data.get('method')
    payload = request_data.get('payload', {})
    
    try:
        if method == "ask":
            query = payload.get('query', '')
            history = payload.get('history', [])
            response = qa({"query": query, "history": history})
            result = response.get('result', response.get('answer', str(response)))
            
        elif method == "upload_links":
            urls = payload.get('urls', [])
            global docs, chunks, db, retriever
            
            docs = []
            for url in urls:
                loader = RecursiveUrlLoader(url=url, max_depth=9, extractor=lambda x: Soup(x, "html.parser").text)
                docs.extend(loader.load())
            
            chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
            
            db = FAISS.from_documents(filter_complex_metadata(chunks), embeddings)
            db.save_local(FAISS_INDEX_PATH)
            
            retriever = db.as_retriever()
            qa = RetrievalQA.from_chain_type(
                llm=model_id,
                chain_type="stuff",
                retriever=retriever,
                verbose=True,
                return_source_documents=True,
                chain_type_kwargs={
                    "verbose": True,
                    "memory": memory,
                    "prompt": prompt
                }
            )
            result = {"message": f"Processed {len(docs)} documents from {len(urls)} URLs"}
            
        else:
            raise ValueError("Unsupported method")

        task_status[task_id] = StatusCodes.SUCCESS  # Update status to SUCCESS

    except Exception as e:
        task_status[task_id] = StatusCodes.ERROR  # Update status to ERROR
        result = {"message": str(e)}

    process_duration = int((time.time() - start_time) * 1000)
    task_results[task_id] = result
    print(f"Task {task_id} completed with result: {result}")
    

    # You might want to store the result somewhere so it can be retrieved later
    # For example, you could use a dictionary to store results:
    # task_results[task_id] = result


    # Optionally, implement a callback function to notify the client of completion
    # send_callback(user_id, task_id, request_id, process_duration, result)

# Uncomment and implement this function if you need to send a callback notification
# def send_callback(user_id, task_id, request_id, process_duration, result):
#     callback_message = {
#         "taskId": task_id,
#         "data": result,
#         "processDuration": process_duration,
#         "status": "success"
#     }
#     # Replace with actual callback URL and send the request
#     # requests.post("callback_url", json=callback_message)


def response_template(request_id, trace_id, process_duration, isResponseImmediate, response, error_code):
    return {
        "request_id": request_id,
        "trace_id": trace_id,
        "process_duration": process_duration,
        "isResponseImmediate": isResponseImmediate,
        "response": response,
        "error_code": error_code
    }

def success_response(task_id, data, data_type, request_id, trace_id, process_duration):
    return response_template(
        request_id=request_id,
        trace_id=trace_id,
        process_duration=process_duration,
        isResponseImmediate=True,
        response={
            "taskId": task_id,
            "data": data,
            "dataType": data_type
        },
        error_code={"status": StatusCodes.SUCCESS, "reason": "Task completed successfully"}
    )

@app.post("/result")
async def result(request: Request, task_request: TaskRequest, x_user_id: str = Header(None), x_request_id: str = Header(None)):
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    result_request_id = str(uuid.uuid4())

    if x_user_id is None or not x_user_id.strip():
        error_code = {"status": StatusCodes.ERROR, "reason": "No User ID found in headers"}
        response_data = response_template(result_request_id, trace_id, -1, True, {}, error_code)
        raise HTTPException(status_code=400, detail=response_data)

    if x_request_id is None or not x_request_id.strip():
        error_code = {"status": StatusCodes.ERROR, "reason": "No request ID found in headers"}
        response_data = response_template(result_request_id, trace_id, -1, True, {}, error_code)
        raise HTTPException(status_code=400, detail=response_data)
    
    if task_request.taskId is None or not task_request.taskId.strip():
        error_code = {"status": StatusCodes.ERROR, "reason": "No task ID found in body"}
        response_data = response_template(result_request_id, trace_id, -1, True, {}, error_code)
        raise HTTPException(status_code=400, detail=response_data)

    status_code = task_status.get(task_request.taskId, StatusCodes.ERROR)
    if status_code == StatusCodes.SUCCESS:
        # Retrieve the actual result
        result = task_results.get(task_request.taskId, {"message": "No result found"})
        response_data = success_response(
            task_request.taskId, result, "RESULT_DATA", x_request_id, trace_id, int((time.time() - start_time) * 1000)
        )
    elif status_code == StatusCodes.PENDING:
        response_data = response_template(
            request_id=result_request_id,
            trace_id=trace_id,
            process_duration=-1,
            isResponseImmediate=False,
            response={"taskId": task_request.taskId},
            error_code={"status": StatusCodes.PENDING, "reason": "Task is still pending"}
        )
    elif status_code == StatusCodes.INPROGRESS:
        response_data = response_template(
        request_id=result_request_id,
        trace_id=trace_id,
        process_duration=-1,
        isResponseImmediate=False,
        response={"taskId": task_request.taskId},
        error_code={"status": StatusCodes.INPROGRESS, "reason": "Task is processing"}
    )
    else:
        response_data = response_template(
            request_id=result_request_id,
            trace_id=trace_id,
            process_duration=-1,
            isResponseImmediate=True,
            response={},
            error_code={"status": StatusCodes.ERROR, "reason": "Task failed or error occurred"}
        )

    return response_data

# Gradio interface
def infer(question, history):
    formatted_history = [(str(q) if q is not None else "", str(a) if a is not None else "") for q, a in history]
    
    response = requests.post("http://localhost:8000/ask", 
                             json={"userToken": "dummy_token",
                                   "requestId": "dummy_id",
                                   "request": {
                                       "method": "ask",
                                       "payload": {"query": question, "history": formatted_history}
                                   }})
    return response.json()

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history[:-1])
    if 'response' in response and 'result' in response['response']:
        history[-1][1] = response['response']['result']
    else:
        history[-1][1] = "Sorry, I couldn't generate a response."
    return history

def upload_links_ui(links):
    urls = [url.strip() for url in links.split(',')]
    response = requests.post("http://localhost:8000/upload_links", 
                             json={
                                 "userToken": "dummy_token",
                                 "requestId": "dummy_id",
                                 "request": {
                                     "method": "upload_links",
                                     "payload": {
                                         "urls": urls
                                     }
                                 }
                             })
    response_data = response.json()
    
    if 'response' in response_data and 'message' in response_data['response']:
        return response_data['response']['message']
    elif 'errorCode' in response_data and 'message' in response_data['errorCode']:
        return f"Error: {response_data['errorCode']['message']}"
    else:
        return str(response_data)

with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        chatbot = gr.Chatbot([], elem_id="chatbot")
        clear = gr.Button("Clear")

        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")

        with gr.Row():
                    links_input = gr.Textbox(label="Upload Links", placeholder="Enter comma-separated URLs")
                    upload_button = gr.Button("Upload")
                    upload_status = gr.Textbox(label="Upload Status")

        question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
            bot, chatbot, chatbot
        )

        clear.click(lambda: None, None, chatbot, queue=False)
        
        upload_button.click(upload_links_ui, inputs=[links_input], outputs=[upload_status])

# Run both FastAPI and Gradio
if __name__ == "__main__":
    import threading
    
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    def run_gradio():
        demo.launch(share=False)
    
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()
    
    # Run Gradio in the main thread
    run_gradio()