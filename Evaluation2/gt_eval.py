from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
# from prompts import GET_EVALUATOER_PROMPT

from evaluation_metrics import gt_eval, gt_eval_Deepseek, gt_eval_geminiflash

import re

import json

import ast

import sys

import time

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


def clean_quotes(s):
    keep_positions = set()

    # Match "choice" or "reason" key and the start of their value
    for match in re.finditer(r'"(choice|reason)"\s*:\s*"', s):
        key = match.group(1)
        key_quoted_start = match.start(0)
        key_quoted_end = match.start(0) + len(f'"{key}"')

        # Keep quotes around the key
        keep_positions.add(key_quoted_start)                    # opening quote of key
        keep_positions.add(key_quoted_end - 1)                  # closing quote of key

        value_start_quote_pos = match.end() - 1
        keep_positions.add(value_start_quote_pos)               # opening quote of value

        val_start = match.end()

        if key == "choice":
            # Look for the next comma or newline block ending
            comma_pos = s.find(",", val_start)
            if comma_pos == -1:
                comma_pos = s.find("}", val_start)
            quote_pos = s.rfind('"', val_start, comma_pos)
        else:  # "reason"
            quote_pos = s.rfind('"', val_start)

        if quote_pos != -1:
            keep_positions.add(quote_pos) 
        
    result = ''.join(c for i, c in enumerate(s) if c != '"' or i in keep_positions)

    return result



if __name__ == "__main__":
    dialogue_data_folder = "/home/haozhu2/Human_Chatbot-Generation/Evaluation3/data_arena/"
    saved_result_folder = "/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_arena/GPT4o_Evaluator/gt_eval/"

    dialogue_data_file_names = ["arena_llama_3b_v1_GPT4oMini_6.jsonl", "arena_llama_3b_v2_GPT4oMini_6.jsonl", "arena_llama_8b_v1_GPT4oMini_6.jsonl", "arena_llama_8b_v2_GPT4oMini_6.jsonl","arena_mistral_v1_GPT4oMini_6.jsonl", "arena_mistral_v2_GPT4oMini_6.jsonl", "arena_model_a_summaries.jsonl"]

    index_pair_list = [(6,0), (6,1), (6,2), (6,3), (6,4), (6,5)]

    for index_pair in index_pair_list:
        
        dialogue_A_file_name = dialogue_data_file_names[index_pair[0]]
        dialogue_B_file_name = dialogue_data_file_names[index_pair[1]]

        dialogue_A_data_path = dialogue_data_folder + dialogue_A_file_name
        dialogue_B_data_path = dialogue_data_folder + dialogue_B_file_name

        dialogue_A_data = parse_jsonl(dialogue_A_data_path)
        dialogue_B_data = parse_jsonl(dialogue_B_data_path)

        saved_result_file_path = saved_result_folder  + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name
        saved_result_file = open(saved_result_file_path, "w", encoding="utf-8")

        for dialoue_index in range(len(dialogue_A_data)):

            print("dialogue A: " + dialogue_A_file_name)
            print("dialogue B: " + dialogue_B_file_name)
            print("dialoue_index: " + str(dialoue_index))

            conversation_A = dialogue_A_data[dialoue_index]["conversation"]
            conversation_B = dialogue_B_data[dialoue_index]["conversation"]


            # ************GPT 4o Evaluator gt_eval**********************
            pair_eval_response = gt_eval(conversation_A, conversation_B, "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", "gpt-4o")

            json_obj = json.loads(pair_eval_response)
            saved_result_file.write(json.dumps(json_obj) + "\n")


            # ************Deepseek Evaluator gt_eval**********************
            # gt_eval_DeepSeek_response = gt_eval_Deepseek(conversation_A, conversation_B)
            # raw_content = gt_eval_DeepSeek_response.content

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


            # ************geminiflash Evaluator gt_eval**********************
            # try_times = 0
            # write_succeed = False
            # while write_succeed == False:

            #     gt_eval_geminiflash_response = gt_eval_geminiflash(conversation_A, conversation_B)
            #     raw_content = gt_eval_geminiflash_response.content

            #     # Extract JSON block safely
            #     match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            #     if match:
            #         try:
            #             json_str = match.group(0)
            #             # json_str = json_str.replace("\\", "")
            #             print(json_str)
            #             # x = clean_reason_quotes(json_str)
            #             # print(x)
            #             # json_str = json_str.replace('"im the goat"', "'im the goat'")


            #             try:
            #                 parsed_json = json.loads(json_str)
            #             except:
            #                 cleaned_json_str = clean_quotes(json_str)
            #                 print("clean double qoutes:")
            #                 print(cleaned_json_str)
            #                 parsed_json = json.loads(cleaned_json_str)

            #             #print(parsed_json)
            #             saved_result_file.write(json.dumps(parsed_json) + "\n")
            #             write_succeed = True
            #         except:
            #             try:
            #                 parsed_json = ast.literal_eval(json_str)
            #                 #print(parsed_json)
            #                 saved_result_file.write(json.dumps(parsed_json) + "\n")
            #                 write_succeed = True
            #             except:
            #                 try_times = try_times + 1
            #                 if try_times < 10:
            #                     print("wait for 10s before the next run, already run %d times" % try_times)
            #                     time.sleep(10)
            #                     continue
            #                 else:
            #                     sys.exit("Tryied 10 times but still encounter errors.")

            #     else:
            #         sys.exit("No JSON object found.")
            #         print("No JSON object found.")


