# DocuGenie - Business-Tailored AI Chatbots in Seconds!

## Overview

DocuGenie is an L2 API solution designed to simplify the creation of custom AI chatbots. Using website links as input, developers can generate AI-powered bots without requiring extensive data collection or pre-trained models. The service utilizes FastAPI, LangChain, and Gradio to provide a secure, fast, and tailored AI chatbot experience for businesses.

## Docker Image

To get started with DocuGenie, you can pull the pre-built Docker image from Docker Hub:

```
docker pull roshangeorge97/docugenie:latest
```

## Youtube Demo:

Check out the [YouTube Demo](https://youtu.be/UvwQrvpDN_Q?feature=shared) to see how DocuGenie works!

## Key Features

- **Rapid Chatbot Development**: Create custom AI chatbots within minutes.
- **Website Link Processing**: Automatically extract and process information from URLs.
- **Secure Data Handling**: Protects sensitive business data throughout the process.
- **Tailored Results**: Provides responses specifically aligned with business needs.
- **Asynchronous Processing**: Manages long-running tasks without blocking server resources.

## Technologies Used

- **FastAPI**: Backend API framework.
- **LangChain**: Document loading, text splitting, embedding generation, and QA (Question Answering).
- **Gradio**: Web interface for user interaction.
- **FAISS**: Vector store indexing for fast document retrieval.
- **Hugging Face Hub**: Access to pre-trained models for text embedding and QA.

## Core Components

1. **Document Processing**: 
   - `RecursiveUrlLoader`: Fetches documents recursively from URLs.
   - `RecursiveCharacterTextSplitter`: Splits large text documents into smaller chunks.

2. **Embedding and Indexing**:
   - `HuggingFaceEmbeddings`: Converts text into vector embeddings.
   - `FAISS`: Indexes document embeddings for fast and efficient retrieval.

3. **Question Answering**:
   - `RetrievalQA`: Utilizes LangChain to answer questions based on processed documents.

4. **Chatbot Management**:
   - `ChatbotManager`: Manages multiple chatbots, each with its own vector store and QA chain.

## API Endpoints

### 1. `/call`
Main endpoint for initiating tasks, such as creating chatbots, uploading links, or asking questions.

- **Methods**: `"ask"`, `"upload_links"`, `"create_chatbot"`, `"load_chatbot"`.
- **Headers Required**: `x_user_id`, `x_marketplace_token`, `x_user_role`, `x_request_id`.
- **Response**: Task ID for asynchronous processing.

### 2. `/result`
Retrieve the results of previously processed tasks.

- **Requires**: A valid task ID.
- **Headers Required**: `x_user_id`, `x_request_id`, `x_marketplace_token`, `x_user_role`.

### 3. `/stats`
Provides API usage statistics, including the number of successful and failed requests.

- **Headers Required**: `x_user_id`, `x_request_id`, `x_marketplace_token`, `x_user_role`.

## Core Functions

- **upload_links(urls)**: Processes a list of URLs for document retrieval.
- **create_chatbot(config)**: Initializes and sets up a new chatbot instance.
- **load_chatbot(chatbot_id)**: Loads an existing chatbot from the system.
- **ask(chatbot_id, question)**: Sends a question to a chatbot for an answer.

## Error Handling and Security

- **Error Handling**: Includes detailed status codes such as `SUCCESS`, `ERROR`, `PENDING`, and `INPROGRESS`.
- **Security**: Includes input validation, header authentication, and recommended rate limiting.

## Future Improvements

- Enhanced task management with advanced queuing.
- API key-based authentication for additional security.
- Improved logging and monitoring for better traceability.

## Unique Value Proposition

DocuGenie offers a rapid, secure, and highly customized AI chatbot development experience, ideal for businesses requiring specialized AI tools without the overhead of traditional model development.


