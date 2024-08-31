import threading
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
import os
import uuid
from typing import Dict, List
from fastapi import FastAPI, Request, Header, HTTPException
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.utils import filter_complex_metadata
from bs4 import BeautifulSoup as Soup
import gradio as gr
import requests
import time

qa = None

# Constants
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
FAISS_INDEX_PATH = "./vectorstore/lc-faiss-multi-mpnet-500"
SUPPORTED_METHOD = ["ask", "upload_links", "create_chatbot", "load_chatbot"]
MARKETPLACE_CALLBACK_URL = "https://example.com"

# Initialize FastAPI app
app = FastAPI()
task_status = {}
task_results = {}

# LangChain setup
global docs, chunks, db, retriever
docs = []


urls = ["https://example.com/"]

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
You are the friendly documentation buddy Solido, who helps novice programmers with simple explanations and examples of code snippets.\
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

# Define ChatbotManager
class ChatbotManager:
    def __init__(self):
        self.chatbots: Dict[str, dict] = {}
        self.model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def create_chatbot(self, name: str) -> str:
        chatbot_id = str(uuid.uuid4())
        self.chatbots[chatbot_id] = {
            "name": name,
            "vectorstore": None,
            "qa": None
        }
        return chatbot_id

    def get_chatbot(self, chatbot_id: str) -> dict:
        return self.chatbots.get(chatbot_id)

    def load_chatbot(self, chatbot_id: str):
        if chatbot_id not in self.chatbots:
            raise ValueError(f"Chatbot with ID {chatbot_id} not found")
        
        chatbot = self.chatbots[chatbot_id]
        if chatbot["vectorstore"] is None:
            vectorstore_path = f"./vectorstore/chatbot_{chatbot_id}"
            if os.path.exists(vectorstore_path):
                chatbot["vectorstore"] = FAISS.load_local(vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                chatbot["qa"] = self.create_qa_chain(chatbot["vectorstore"].as_retriever())
            else:
                raise ValueError(f"No vectorstore found for chatbot {chatbot_id}")

    def train_chatbot(self, chatbot_id: str, docs: List[dict]):
        if chatbot_id not in self.chatbots:
            raise ValueError(f"Chatbot with ID {chatbot_id} not found")
        
        chatbot = self.chatbots[chatbot_id]
        chunks = text_splitter.create_documents(
            [doc["content"] for doc in docs], 
            metadatas=[doc["metadata"] for doc in docs]
        )
        
        vectorstore = FAISS.from_documents(filter_complex_metadata(chunks), self.embeddings)
        vectorstore_path = f"./vectorstore/chatbot_{chatbot_id}"
        vectorstore.save_local(vectorstore_path)
        
        chatbot["vectorstore"] = vectorstore
        chatbot["qa"] = self.create_qa_chain(vectorstore.as_retriever())

    def create_qa_chain(self, retriever):
        return RetrievalQA.from_chain_type(
            llm=model_id,
            chain_type="stuff",
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
            chain_type_kwargs={
                "verbose": True,
                "memory": ConversationBufferMemory(memory_key="history", input_key="question"),
                "prompt": prompt
            }
        )

# Initialize ChatbotManager
chatbot_manager = ChatbotManager()

# Call Endpoint
@app.post("/call")
async def call_endpoint(request: Request, x_user_id: str = Header(None), x_marketplace_token: str = Header(None), x_user_role: str = Header(None)):
    start_time = time.time()
    request_data = await request.json()

    if not x_user_id or not x_marketplace_token or not x_user_role:
        return response_template(
            request_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            process_duration=-1,
            isResponseImmediate=True,
            response={},
            error_code={"status": StatusCodes.INVALID_REQUEST, "reason": "Missing required headers"}
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
    if not method or method not in SUPPORTED_METHOD:
        return response_template(
            request_id=request_id,
            trace_id=str(uuid.uuid4()),
            process_duration=-1,
            isResponseImmediate=True,
            response={},
            error_code={"status": StatusCodes.INVALID_REQUEST, "reason": "Unsupported or missing method"}
        )

    payload = request_data.get('payload', {})
    
    task_id = str(uuid.uuid4())
    task_status[task_id] = StatusCodes.PENDING

    process_duration = int((time.time() - start_time) * 1000)

    # Start a background thread to process the request
    threading.Thread(target=process_task, args=(task_id, method, payload, x_user_id, request_id, x_marketplace_token, x_user_role)).start()

    return response_template(
        request_id=request_id,
        trace_id=str(uuid.uuid4()),
        process_duration=process_duration,
        isResponseImmediate=False,
        response={"taskId": task_id},
        error_code={"status": StatusCodes.PENDING, "reason": "Task is pending"}
    )

def process_task(task_id, method, payload, user_id, request_id, marketplace_token, user_role):
    global chatbot_manager
    start_time = time.time()
    task_status[task_id] = StatusCodes.INPROGRESS

    try:
        if method == "create_chatbot":
            chatbot_name = payload.get("name", "New Chatbot")
            chatbot_id = chatbot_manager.create_chatbot(chatbot_name)
            
            # Automatically upload a sample link
            sample_url = "https://example.com"
            docs = []
            loader = RecursiveUrlLoader(url=sample_url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text)
            docs.extend(loader.load())
            
            chatbot_manager.train_chatbot(chatbot_id, [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs])
            
            result = {
                "chatbot_id": chatbot_id,
                "sample_link_uploaded": sample_url
            }

        elif method == "load_chatbot":
            chatbot_id = payload.get("chatbot_id")
            if not chatbot_id:
                raise ValueError("Chatbot ID is required")
            chatbot_manager.load_chatbot(chatbot_id)
            result = {"message": f"Chatbot {chatbot_id} loaded successfully"}

        elif method == "upload_links":
            chatbot_id = payload.get("chatbot_id")
            urls = payload.get("urls", [])
            if not chatbot_id:
                raise ValueError("Chatbot ID is required")
            
            docs = []
            for url in urls:
                loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text)
                docs.extend(loader.load())
            
            chatbot_manager.train_chatbot(chatbot_id, [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs])
            result = {"message": f"Processed {len(docs)} documents from {len(urls)} URLs for chatbot {chatbot_id}"}

        elif method == "ask":
            chatbot_id = payload.get("chatbot_id")
            query = payload.get("query")
            history = payload.get("history", [])
            
            if not chatbot_id:
                raise ValueError("Chatbot ID is required")
            if not query:
                raise ValueError("Query is required")
            
            chatbot = chatbot_manager.get_chatbot(chatbot_id)
            if chatbot is None:
                raise ValueError(f"Chatbot with ID {chatbot_id} not found")
            
            response = chatbot["qa"]({"query": query, "history": history})
            result = {"result": response.get('result', response.get('answer', str(response)))}

        else:
            raise ValueError(f"Unsupported method: {method}")

        task_status[task_id] = StatusCodes.SUCCESS
        process_duration = int((time.time() - start_time) * 1000)
        task_results[task_id] = result

        # Send callback to marketplace
        callback_data = {
            "apiVersion": API_VERSION,
            "service": SERVICE_NAME,
            "datetime": datetime.datetime.now().isoformat(),
            "processDuration": process_duration,
            "taskId": task_id,
            "isResponseImmediate": False,
            "extraType": "others",
            "response": {
                "dataType": "META_DATA",
                "data": result
            },
            "errorCode": {
                "status": StatusCodes.SUCCESS,
                "reason": "success"
            }
        }
        
        headers = {
            "x-marketplace-token": marketplace_token,
            "x-request-id": request_id,
            "x-user-id": user_id,
            "x-user-role": user_role
        }
        
        return callback_data

    except Exception as e:
        task_status[task_id] = StatusCodes.ERROR
        error_message = str(e)
        task_results[task_id] = {"error": error_message}
        
        # Send error callback to marketplace
        callback_data = {
            "apiVersion": API_VERSION,
            "service": SERVICE_NAME,
            "datetime": datetime.datetime.now().isoformat(),
            "processDuration": int((time.time() - start_time) * 1000),
            "taskId": task_id,
            "isResponseImmediate": False,
            "extraType": "others",
            "response": {
                "dataType": "META_DATA",
                "data": {"error": error_message}
            },
            "errorCode": {
                "status": StatusCodes.ERROR,
                "reason": error_message
            }
        }
        
        headers = {
            "x-marketplace-token": marketplace_token,
            "x-request-id": request_id,
            "x-user-id": user_id,
            "x-user-role": user_role
        }
        
        return callback_data

# Result API
@app.post("/result")
async def result(request: Request, task_request: TaskRequest, x_user_id: str = Header(None), x_request_id: str = Header(None), x_marketplace_token: str = Header(None), x_user_role: str = Header(None)):
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    result_request_id = str(uuid.uuid4())

    if not all([x_user_id, x_request_id, x_marketplace_token, x_user_role]):
        error_code = {"status": StatusCodes.ERROR, "reason": "Missing required headers"}
        response_data = response_template(result_request_id, trace_id, -1, True, {}, error_code)
        raise HTTPException(status_code=400, detail=response_data)
    
    if task_request.taskId is None or not task_request.taskId.strip():
        error_code = {"status": StatusCodes.ERROR, "reason": "No task ID found in body"}
        response_data = response_template(result_request_id, trace_id, -1, True, {}, error_code)
        raise HTTPException(status_code=400, detail=response_data)

    status_code = task_status.get(task_request.taskId, StatusCodes.ERROR)
    if status_code == StatusCodes.SUCCESS:
        result = task_results.get(task_request.taskId, {"message": "No result found"})
        response_data = success_response(
            task_request.taskId, result, "META_DATA", x_request_id, trace_id, int((time.time() - start_time) * 1000)
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

# Stats API
@app.post("/stats")
async def stats(request: Request, x_user_id: str = Header(None), x_request_id: str = Header(None), x_marketplace_token: str = Header(None), x_user_role: str = Header(None)):
    if not all([x_user_id, x_request_id, x_marketplace_token, x_user_role]):
        error_code = {"status": StatusCodes.ERROR, "reason": "Missing required headers"}
        response_data = response_template(x_request_id, str(uuid.uuid4()), -1, True, {}, error_code)
        raise HTTPException(status_code=400, detail=response_data)

    stats_data = {
        "numRequestSuccess": 100,
        "numRequestFailed": 10
    }

    return response_template(
        request_id=x_request_id,
        trace_id=str(uuid.uuid4()),
        process_duration=0,
        isResponseImmediate=True,
        response=stats_data,
        error_code={"status": StatusCodes.SUCCESS, "reason": "Stats retrieved successfully"}
    )

# Response Templates
def response_template(request_id, trace_id, process_duration, isResponseImmediate, response, error_code):
    return {
        "requestId": request_id,
        "traceId": trace_id,
        "apiVersion": API_VERSION,
        "service": SERVICE_NAME,
        "datetime": datetime.datetime.now().isoformat(),
        "processDuration": process_duration,
        "isResponseImmediate": isResponseImmediate,
        "extraType": "others",
        "response": response,
        "errorCode": error_code
    }

def success_response(task_id, data, data_type, request_id, trace_id, process_duration):
    return response_template(
        request_id=request_id,
        trace_id=trace_id,
        process_duration=process_duration,
        isResponseImmediate=True,
        response={
            "taskId": task_id,
            "dataType": data_type,
            "data": data
        },
        error_code={"status": StatusCodes.SUCCESS, "reason": "Task completed successfully"}
    )

# Gradio interface
def infer(question, history):
    formatted_history = [(str(q) if q is not None else "", str(a) if a is not None else "") for q, a in history]
    
    response = requests.post("http://localhost:8000/call", 
                             headers={"x-user-id": "dummy_user_id", "x-request-id": str(uuid.uuid4())},
                             json={
                                 "method": "ask",
                                 "payload": {"query": question, "history": formatted_history}
                             })
    return response.json()

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history[:-1])
    if 'response' in response and 'taskId' in response['response']:
        task_id = response['response']['taskId']
        # Poll for result
        while True:
            result_response = requests.post("http://localhost:8000/result",
                                            headers={"x-user-id": "dummy_user_id", "x-request-id": str(uuid.uuid4())},
                                            json={"taskId": task_id})
            result_data = result_response.json()
            if result_data['error_code']['status'] == StatusCodes.SUCCESS:
                if isinstance(result_data['response'], dict) and 'data' in result_data['response']:
                    if isinstance(result_data['response']['data'], dict) and 'result' in result_data['response']['data']:
                        history[-1][1] = result_data['response']['data']['result']
                    else:
                        history[-1][1] = str(result_data['response']['data'])
                else:
                    history[-1][1] = str(result_data['response'])
                break
            elif result_data['error_code']['status'] in [StatusCodes.ERROR, StatusCodes.UNSUPPORTED]:
                history[-1][1] = f"Error: {result_data['error_code']['reason']}"
                break
            time.sleep(1)  # Wait for 1 second before polling again
    else:
        history[-1][1] = "Sorry, I couldn't generate a response."
    return history

def upload_links_ui(links):
    urls = [url.strip() for url in links.split(',')]
    response = requests.post("http://localhost:8000/call", 
                             headers={"x-user-id": "dummy_user_id", "x-request-id": str(uuid.uuid4())},
                             json={
                                 "method": "upload_links",
                                 "payload": {
                                     "urls": urls
                                 }
                             })
    response_data = response.json()
    
    if 'response' in response_data and 'taskId' in response_data['response']:
        task_id = response_data['response']['taskId']
        # Poll for result
        while True:
            result_response = requests.post("http://localhost:8000/result",
                                            headers={"x-user-id": "dummy_user_id", "x-request-id": str(uuid.uuid4())},
                                            json={"taskId": task_id})
            result_data = result_response.json()
            if result_data['error_code']['status'] == StatusCodes.SUCCESS:
                return result_data['response']['data']['message']
            elif result_data['error_code']['status'] in [StatusCodes.ERROR, StatusCodes.UNSUPPORTED]:
                return f"Error: {result_data['error_code']['reason']}"
            time.sleep(1)  # Wait for 1 second before polling again
    else:
        return f"Error: {response_data.get('error_code', {}).get('reason', 'Unknown error')}"


def call_endpoint(method, payload):
    response = requests.post(
        "http://localhost:8000/call",
        headers={
            "x-user-id": "dummy_user_id",
            "x-request-id": str(uuid.uuid4()),
            "x-marketplace-token": "some_token",  
            "x-user-role": "some_role"
        },
        json={"method": method, "payload": payload}
    )
    return response.json()

def poll_for_result(task_id):
    while True:
        result_response = requests.post(
            "http://localhost:8000/result",
            headers={
                "x-user-id": "dummy_user_id",
                "x-request-id": str(uuid.uuid4()),
                "x-marketplace-token": "some_token",  
                "x-user-role": "some_role"
            },
            json={"taskId": task_id}
        )
        result_data = result_response.json()

        if result_data['errorCode']['status'] == StatusCodes.SUCCESS: 
            return result_data['response']['data']
        elif result_data['errorCode']['status'] in  [StatusCodes.ERROR, StatusCodes.UNSUPPORTED]: 
            return {"error": result_data['errorCode']['reason']}
        
        time.sleep(1)  # Wait for 1 second before polling again

def create_new_chatbot(name):
    response = call_endpoint("create_chatbot", {"name": name})
    task_id = response.get("response", {}).get("taskId")
    if task_id:
        result = poll_for_result(task_id)
        new_chatbot_id = result.get("chatbot_id", "Error creating chatbot")
        return new_chatbot_id, new_chatbot_id
    return None, "Error: Task ID not found"

def load_existing_chatbot(chatbot_id):
    response = call_endpoint("load_chatbot", {"chatbot_id": chatbot_id})
    task_id = response.get("response", {}).get("taskId")
    if task_id:
        result = poll_for_result(task_id)
        if "error" not in result:
            return chatbot_id, f"Loaded chatbot: {chatbot_id}"
    return None, "Error loading chatbot"

def ask_question(question, history, chatbot_id):
    if not chatbot_id:
        return history + [(question, "Error: No chatbot loaded. Please create or load a chatbot first.")]
    
    response = call_endpoint("ask", {"chatbot_id": chatbot_id, "query": question, "history": history})
    task_id = response.get("response", {}).get("taskId")
    if task_id:
        result = poll_for_result(task_id)
        answer = result.get("result", "Error getting response")
        return history + [(question, answer)]
    return history + [(question, "Error: Task ID not found")]

def upload_links_ui(links, chatbot_id):
    if not chatbot_id:
        return "Error: No chatbot loaded. Please create or load a chatbot first."
    
    urls = [url.strip() for url in links.split(',')]
    response = call_endpoint("upload_links", {"chatbot_id": chatbot_id, "urls": urls})
    task_id = response.get("response", {}).get("taskId")
    if task_id:
        result = poll_for_result(task_id)
        return result.get("message", "Error uploading links")
    return "Error: Task ID not found"

with gr.Blocks() as demo:
    chatbot_id = gr.State(None)
    
    with gr.Column(elem_id="col-container"):
        chatbot_name = gr.Textbox(label="Chatbot Name", placeholder="Enter chatbot name")
        create_button = gr.Button("Create Chatbot")
        load_chatbot = gr.Textbox(label="Load Chatbot", placeholder="Enter chatbot ID")
        load_button = gr.Button("Load Chatbot")
        current_chatbot_id = gr.Textbox(label="Current Chatbot ID", interactive=False)
        
        chatbot = gr.Chatbot([], elem_id="chatbot")
        clear = gr.Button("Clear")

        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter")

        with gr.Row():
            links_input = gr.Textbox(label="Upload Links", placeholder="Enter comma-separated URLs")
            upload_button = gr.Button("Upload")
            upload_status = gr.Textbox(label="Upload Status")

    create_button.click(create_new_chatbot, inputs=[chatbot_name], outputs=[chatbot_id, current_chatbot_id])
    load_button.click(load_existing_chatbot, inputs=[load_chatbot], outputs=[chatbot_id, current_chatbot_id])
    
    question.submit(ask_question, inputs=[question, chatbot, chatbot_id], outputs=[chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)
    upload_button.click(upload_links_ui, inputs=[links_input, chatbot_id], outputs=[upload_status])

# Run both FastAPI and Gradio
if __name__ == "__main__":
    import nest_asyncio
    from fastapi.middleware.cors import CORSMiddleware
    
    nest_asyncio.apply()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=8000)