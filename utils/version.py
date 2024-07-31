import json

API_VERSION = "1.0"
SERVICE_NAME = "LangChainQA"
AI_APP_VERSION=""
with open("config.json") as fp:
    version = json.load(fp)
    SERVICE_NAME = version["service_name"]
    AI_APP_VERSION = version["ai_app_version"]
    API_VERSION = version["api_version"]