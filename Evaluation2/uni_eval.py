from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
# from prompts import GET_EVALUATOER_PROMPT

from evaluation_metrics import uni_eval

import re

import json


# Function to read and parse a JSONL file
def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))  # Parse each line as a JSON object
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
    return data


if __name__ == "__main__":

    dialogue_data_file_names = ["oasst1_en_DeepSeek_GPT4oMini_6.jsonl", "oasst1_en_GPT4oMini_GPT4oMini_6.jsonl", "oasst1_en_GPT4o_GPT4oMini_6.jsonl", "oasst1_en_llama_3B_GPT4oMini_6.jsonl", "oasst1_en_llama_70B_GPT4oMini_6.jsonl", "oasst1_en_min_6_turns_summary.jsonl"]

    for dialogue_data_file_name in dialogue_data_file_names:

        dialogue_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/" + dialogue_data_file_name

        saved_result_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result/uni_eval/" + dialogue_data_file_name
        saved_result_file = open(saved_result_file_path, "w", encoding="utf-8")

        dialogue_data = parse_jsonl(dialogue_data_path)

        dialogue_index = 0
        for entry in dialogue_data:

            dialogue_index = dialogue_index + 1

            print("dialugoe data: " + dialogue_data_file_name)
            print("dialogue: " + str(dialogue_index))

            conversation = entry["conversation"]

            # print(type(conversation))
            # print(conversation)

            # print(type(conversation[0]))

            uni_eval_response = uni_eval(conversation, "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", "gpt-4o")

            
            # print(uni_eval_response)

            # print(type(uni_eval_response))

            json_obj = json.loads(uni_eval_response)
            saved_result_file.write(json.dumps(json_obj) + "\n")



