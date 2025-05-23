from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_deepseek import ChatDeepSeek
import yaml
import os
# from huggingface_hub import InferenceClient
# from langchain.schema import HumanMessage, AIMessage, SystemMessage


api_path = 'api.yaml'
api_keys = yaml.safe_load(open(api_path, 'r'))

setting_path = 'setting.yaml'
setting = yaml.safe_load(open(setting_path, 'r'))

# HuggingFace Serverless Inference
# class ChatHuggingFaceInference:
#     def __init__(self, api_key, model, provider, temperature, max_tokens, max_retries):
#         self.client = InferenceClient(provider=provider, api_key=api_key)
#         self.model = model
#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         self.max_retries = max_retries

#     def invoke(self, messages):
#         # Convert LangChain message objects to dictionaries
#         formatted_messages = []
#         for message in messages:
#             if isinstance(message, HumanMessage):
#                 formatted_messages.append({"role": "user", "content": message.content})
#             elif isinstance(message, AIMessage):
#                 formatted_messages.append({"role": "assistant", "content": message.content})
#             elif isinstance(message, SystemMessage):
#                 formatted_messages.append({"role": "system", "content": message.content})
#             else:
#                 raise ValueError(f"Unknown message type: {type(message)}")

#         # print("Formatted message: ", formatted_messages)
#         # Call Hugging Face Inference API
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=formatted_messages,  # Use converted messages
#             max_tokens=self.max_tokens,
#         )

#         # Return the assistant's response
#         return AIMessage(content=response.choices[0].message["content"])

GPT4o = ChatOpenAI(api_key=api_keys['openai'], model=setting['gpt4o']['model'], temperature=setting['gpt4o']['temperature'], max_tokens=setting['gpt4o']['max_tokens'], max_retries=setting['gpt4o']['max_retries'])
GPT4oMini = ChatOpenAI(api_key=api_keys['openai'], model=setting['gpt4o-mini']['model'], temperature=setting['gpt4o-mini']['temperature'], max_tokens=setting['gpt4o-mini']['max_tokens'], max_retries=setting['gpt4o-mini']['max_retries'])
DeepSeek = ChatDeepSeek(api_key=api_keys['deepseek'], model=setting['deepseek']['model'], temperature=setting['deepseek']['temperature'], max_tokens=setting['deepseek']['max_tokens'], max_retries=setting['deepseek']['max_retries'])
Claude3_7 = ChatAnthropic(api_key=api_keys['anthropic'], model=setting['claude3_7']['model'], temperature=setting['claude3_7']['temperature'], max_tokens=setting['claude3_7']['max_tokens'], max_retries=setting['claude3_7']['max_retries'])
GeminiFlash = ChatGoogleGenerativeAI(google_api_key=api_keys['google'], model=setting['GeminiFlash']['model'], temperature=setting['GeminiFlash']['temperature'], max_tokens=setting['GeminiFlash']['max_tokens'], max_retries=setting['GeminiFlash']['max_retries'])
# llama3_2_3B = ChatHuggingFaceInference(api_key=api_keys['huggingface'],model=setting['llama3_2_3B']['model'], provider = setting['llama3_2_3B']['provider'], temperature=setting['llama3_2_3B']['temperature'], max_tokens=setting['llama3_2_3B']['max_tokens'], max_retries=setting['llama3_2_3B']['max_retries'] )
# llama3_3_70B = ChatHuggingFaceInference(api_key=api_keys['huggingface'],model=setting['llama3_3_70B']['model'], provider = setting['llama3_3_70B']['provider'], temperature=setting['llama3_3_70B']['temperature'], max_tokens=setting['llama3_3_70B']['max_tokens'], max_retries=setting['llama3_3_70B']['max_retries'] )
llama_1B = ChatOpenAI(base_url="https://router.huggingface.co/novita/v3/openai", api_key=api_keys['huggingface'], model=setting['llama3_2_1B']['model'], temperature=setting['llama3_2_1B']['temperature'], max_tokens=setting['llama3_2_1B']['max_tokens'], max_retries=setting['llama3_2_1B']['max_retries'])
llama_3B = ChatOpenAI(base_url="https://router.huggingface.co/novita/v3/openai", api_key=api_keys['huggingface'], model=setting['llama3_2_3B']['model'], temperature=setting['llama3_2_3B']['temperature'], max_tokens=setting['llama3_2_3B']['max_tokens'], max_retries=setting['llama3_2_3B']['max_retries'])
llama_8B = ChatOpenAI(base_url="https://router.huggingface.co/novita/v3/openai", api_key=api_keys['huggingface'], model=setting['llama3_1_8B']['model'], temperature=setting['llama3_1_8B']['temperature'], max_tokens=setting['llama3_1_8B']['max_tokens'], max_retries=setting['llama3_1_8B']['max_retries'])
llama_70B = ChatOpenAI(base_url="https://router.huggingface.co/novita/v3/openai", api_key=api_keys['huggingface'], model=setting['llama3_3_70B']['model'], temperature=setting['llama3_3_70B']['temperature'], max_tokens=setting['llama3_3_70B']['max_tokens'], max_retries=setting['llama3_3_70B']['max_retries'])
mistral_7B = ChatOpenAI(base_url="https://router.huggingface.co/novita/v3/openai", api_key=api_keys['huggingface'], model=setting['mistral_7B']['model'], temperature=setting['mistral_7B']['temperature'], max_tokens=setting['mistral_7B']['max_tokens'], max_retries=setting['mistral_7B']['max_retries'])
qwen_14B = ChatOpenAI(base_url="https://router.huggingface.co/novita/v3/openai", api_key=api_keys['huggingface'], model=setting['qwen_14B']['model'], temperature=setting['qwen_14B']['temperature'], max_tokens=setting['qwen_14B']['max_tokens'], max_retries=setting['qwen_14B']['max_retries'])
# DeepSeek = ChatOpenAI(base_url="https://router.huggingface.co/novita/v3/openai", api_key=api_keys['huggingface'], model=setting['deepseek']['model'], temperature=setting['deepseek']['temperature'], max_tokens=setting['deepseek']['max_tokens'], max_retries=setting['deepseek']['max_retries'])
gemma_27b = ChatOpenAI(base_url="https://router.huggingface.co/novita/v3/openai", api_key=api_keys['huggingface'], model=setting['gemma_27b']['model'], temperature=setting['gemma_27b']['temperature'], max_tokens=setting['gemma_27b']['max_tokens'], max_retries=setting['gemma_27b']['max_retries'])

llama_3b_v1 = ChatOpenAI(base_url=setting['llama_3b_v1']['base_url'], api_key=api_keys['huggingface'], model=setting['llama_3b_v1']['model'], temperature=setting['llama_3b_v1']['temperature'], max_tokens=setting['llama_3b_v1']['max_tokens'], max_retries=setting['llama_3b_v1']['max_retries'])
llama_3b_v2 = ChatOpenAI(base_url=setting['llama_3b_v2']['base_url'], api_key=api_keys['huggingface'], model=setting['llama_3b_v2']['model'], temperature=setting['llama_3b_v2']['temperature'], max_tokens=setting['llama_3b_v2']['max_tokens'], max_retries=setting['llama_3b_v2']['max_retries'])
llama_8b_v1 = ChatOpenAI(base_url=setting['llama_8b_v1']['base_url'], api_key=api_keys['huggingface'], model=setting['llama_8b_v1']['model'], temperature=setting['llama_8b_v1']['temperature'], max_tokens=setting['llama_8b_v1']['max_tokens'], max_retries=setting['llama_8b_v1']['max_retries'])
llama_8b_v2 = ChatOpenAI(base_url=setting['llama_8b_v2']['base_url'], api_key=api_keys['huggingface'], model=setting['llama_8b_v2']['model'], temperature=setting['llama_8b_v2']['temperature'], max_tokens=setting['llama_8b_v2']['max_tokens'], max_retries=setting['llama_8b_v2']['max_retries'])
mistral_v1 = ChatOpenAI(base_url=setting['mistral_v1']['base_url'], api_key=api_keys['huggingface'], model=setting['mistral_v1']['model'], temperature=setting['mistral_v1']['temperature'], max_tokens=setting['mistral_v1']['max_tokens'], max_retries=setting['mistral_v1']['max_retries'])
mistral_v2 = ChatOpenAI(base_url=setting['mistral_v2']['base_url'], api_key=api_keys['huggingface'], model=setting['mistral_v2']['model'], temperature=setting['mistral_v2']['temperature'], max_tokens=setting['mistral_v2']['max_tokens'], max_retries=setting['mistral_v2']['max_retries'])