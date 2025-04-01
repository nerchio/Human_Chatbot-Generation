# Models used for the evaluation purpose

from langchain_openai import ChatOpenAI
import yaml

api_path = 'api.yaml'
api_keys = yaml.safe_load(open(api_path, 'r'))

setting_path = 'setting.yaml'
setting = yaml.safe_load(open(setting_path, 'r'))

GPT4o = ChatOpenAI(api_key=api_keys['openai'], model=setting['gpt4o']['model'], temperature=setting['gpt4o']['temperature'], max_tokens=setting['gpt4o']['max_tokens'], max_retries=setting['gpt4o']['max_retries'])
GPT4oMini = ChatOpenAI(api_key=api_keys['openai'], model=setting['gpt4o-mini']['model'], temperature=setting['gpt4o-mini']['temperature'], max_tokens=setting['gpt4o-mini']['max_tokens'], max_retries=setting['gpt4o-mini']['max_retries'])