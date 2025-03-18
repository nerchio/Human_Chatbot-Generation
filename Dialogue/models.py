from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_deepseek import ChatDeepSeek
import yaml
import os

api_path = 'api.yaml'
api_keys = yaml.safe_load(open(api_path, 'r'))

setting_path = 'setting.yaml'
setting = yaml.safe_load(open(setting_path, 'r'))

GPT4o = ChatOpenAI(api_key=api_keys['openai'], model=setting['gpt4o']['model'], temperature=setting['gpt4o']['temperature'], max_tokens=setting['gpt4o']['max_tokens'], max_retries=setting['gpt4o']['max_retries'])
GPT4oMini = ChatOpenAI(api_key=api_keys['openai'], model=setting['gpt4o-mini']['model'], temperature=setting['gpt4o-mini']['temperature'], max_tokens=setting['gpt4o-mini']['max_tokens'], max_retries=setting['gpt4o-mini']['max_retries'])
DeepSeek = ChatDeepSeek(api_key=api_keys['deepseek'], model=setting['deepseek']['model'], temperature=setting['deepseek']['temperature'], max_tokens=setting['deepseek']['max_tokens'], max_retries=setting['deepseek']['max_retries'])
