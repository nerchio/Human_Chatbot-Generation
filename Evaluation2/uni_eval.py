from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
import ast
# from prompts import GET_EVALUATOER_PROMPT

from evaluation_metrics import uni_eval, uni_eval_Deepseek, uni_eval_geminiflash, uni_eval_claude

import re

import json

import sys


# Function to read and parse a JSONL file
def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        i = 1
        for line in file:
            try:
                data.append(json.loads(line))  # Parse each line as a JSON object
            except json.JSONDecodeError as e:
                print("line %d" % i)
                print(f"Error parsing line: {e}")
            i=i+1
    return data


if __name__ == "__main__":
    dialogue_data_folder = "/home/haozhu2/Human_Chatbot-Generation/Evaluation3/data_arena/"
    saved_file_folder = "/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_arena/GPT4o_Evaluator/uni_eval/"

    dialogue_data_file_names = ["arena_llama_3b_v1_GPT4oMini_6.jsonl", "arena_llama_3b_v2_GPT4oMini_6.jsonl", "arena_llama_8b_v1_GPT4oMini_6.jsonl", "arena_llama_8b_v2_GPT4oMini_6.jsonl","arena_mistral_v1_GPT4oMini_6.jsonl", "arena_mistral_v2_GPT4oMini_6.jsonl"]

    for dialogue_data_file_name in dialogue_data_file_names:

        dialogue_data_path = dialogue_data_folder  + dialogue_data_file_name

        saved_result_file_path = saved_file_folder + dialogue_data_file_name
        saved_result_file = open(saved_result_file_path, "w", encoding="utf-8")

        dialogue_data = parse_jsonl(dialogue_data_path)


        dialogue_index = 0
        for entry in dialogue_data:

            dialogue_index = dialogue_index + 1

            print("dialugoe data: " + dialogue_data_file_name)
            print("dialogue: " + str(dialogue_index))

            conversation = entry["conversation"]


            # **************GPT4o Evaluator uni_eval*****************
            uni_eval_response = uni_eval(conversation, "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", "gpt-4o")
            json_obj = json.loads(uni_eval_response)
            saved_result_file.write(json.dumps(json_obj) + "\n")



            # **************Deepseek Evaluator uni_eval*****************
            # uni_eval_DeepSeek_response = uni_eval_Deepseek(conversation)
            # raw_content = uni_eval_DeepSeek_response.content

            # # print(raw_content)

            # # Extract JSON block safely
            # match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            # if match:
            #     try:
            #         json_str = match.group(0)
            #         parsed_json = json.loads(json_str)
            #         print(parsed_json)
            #         saved_result_file.write(json.dumps(parsed_json) + "\n")
            #     except:
            #         parsed_json = ast.literal_eval(json_str)
            #         print(parsed_json)
            #         saved_result_file.write(json.dumps(parsed_json) + "\n")
            # else:
            #     print("No JSON object found.")


            # uni_eval_llama_70B_response = uni_eval_llama3_2_70B(conversation)

            # # print(uni_eval_llama_70B_response)
            # json_obj = json.loads(uni_eval_llama_70B_response)
            # saved_result_file.write(json.dumps(json_obj) + "\n")



            # **************geminiflash Evaluator uni_eval*****************
            # uni_eval_geminiflash_response = uni_eval_geminiflash(conversation)
            # raw_content = uni_eval_geminiflash_response.content
            # # Extract JSON block safely
            # match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            # if match:
            #     try:
            #         json_str = match.group(0)
            #         parsed_json = json.loads(json_str)
            #         print(parsed_json)
            #         saved_result_file.write(json.dumps(parsed_json) + "\n")
            #     except:
            #         parsed_json = ast.literal_eval(json_str)
            #         print(parsed_json)
            #         saved_result_file.write(json.dumps(parsed_json) + "\n")
            # else:
            #     sys.exit("No JSON object found.")
            #     print("No JSON object found.")




            # **************Claude Evaluator uni_eval*****************
            # uni_eval_claude_response = uni_eval_claude(conversation)
            # # print(uni_eval_claude_response)
            # raw_content = uni_eval_claude_response.content

            # # Extract JSON block safely
            # match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            # if match:
            #     json_str = match.group(0)
            #     parsed_json = json.loads(json_str)
            #     print(parsed_json)
            #     saved_result_file.write(json.dumps(parsed_json) + "\n")
            # else:
            #     print("No JSON object found.")





