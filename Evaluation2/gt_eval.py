from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
# from prompts import GET_EVALUATOER_PROMPT

from evaluation_metrics import gt_eval

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

    # index_pair_list = [(5,0), (5,1), (5,2), (5,3), (5,4)]

    index_pair_list = [(5,1), (5,2)]


    for index_pair in index_pair_list:
        
        dialogue_A_file_name = dialogue_data_file_names[index_pair[0]]
        dialogue_B_file_name = dialogue_data_file_names[index_pair[1]]

        dialogue_A_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/" + dialogue_A_file_name
        dialogue_B_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/" + dialogue_B_file_name

        dialogue_A_data = parse_jsonl(dialogue_A_data_path)
        dialogue_B_data = parse_jsonl(dialogue_B_data_path)

        saved_result_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result/gt_eval/" + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name
        saved_result_file = open(saved_result_file_path, "w", encoding="utf-8")

        # conversation_A = dialogue_A_data[0]
        # conversation_B = dialogue_B_data[0]

        # pair_eval_response = pair_eval(conversation_A, conversation_B, "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", "gpt-4o")


        # print(pair_eval_response)
        for dialoue_index in range(len(dialogue_A_data)):

            print("dialogue A: " + dialogue_A_file_name)
            print("dialogue B: " + dialogue_B_file_name)
            print("dialoue_index: " + str(dialoue_index))

            conversation_A = dialogue_A_data[dialoue_index]["conversation"]
            conversation_B = dialogue_B_data[dialoue_index]["conversation"]

            # print(type(conversation_A))
            # print(type(conversation_A[0]))


            pair_eval_response = gt_eval(conversation_A, conversation_B, "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", "gpt-4o")

            json_obj = json.loads(pair_eval_response)
            saved_result_file.write(json.dumps(json_obj) + "\n")

