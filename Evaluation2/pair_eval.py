from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
# from prompts import GET_EVALUATOER_PROMPT

from evaluation_metrics import pair_eval, pair_eval_geminishflash

import re

import json

import ast

import time

import sys


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

    # dialogue_data_file_names = ["oasst1_en_DeepSeek_GPT4oMini_6.jsonl", "oasst1_en_GPT4oMini_GPT4oMini_6.jsonl", "oasst1_en_GPT4o_GPT4oMini_6.jsonl", "oasst1_en_llama_3B_GPT4oMini_6.jsonl", "oasst1_en_llama_70B_GPT4oMini_6.jsonl", "oasst1_en_min_6_turns_summary.jsonl"]

    dialogue_data_file_names = ["arena_DeepSeek_GPT4oMini_10.jsonl", "arena_GPT4oMini_GPT4oMini_10.jsonl", "arena_GPT4o_GPT4oMini_10.jsonl", "arena_llama_3B_GPT4oMini_10.jsonl", "arena_llama_70B_GPT4oMini_10.jsonl", "arena_mistral_7B_GPT4oMini_10.jsonl", "arena_model_a_summaries.jsonl"]


    # index_pair_list = [(0,2), (1,2), (3,4), (4,2)]

    index_pair_list = [(2,1), (2,0), (2, 4)]

    index_pair_list = [(2,0), (2, 4)]

    for index_pair in index_pair_list:
        
        dialogue_A_file_name = dialogue_data_file_names[index_pair[0]]
        dialogue_B_file_name = dialogue_data_file_names[index_pair[1]]

        dialogue_A_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data_arena/" + dialogue_A_file_name
        dialogue_B_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data_arena/" + dialogue_B_file_name

        dialogue_A_data = parse_jsonl(dialogue_A_data_path)
        dialogue_B_data = parse_jsonl(dialogue_B_data_path)

        saved_result_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result_arena/GeminiFlash_Evaluator/pair_eval/" + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name
        saved_result_file = open(saved_result_file_path, "w", encoding="utf-8")

        # conversation_A = dialogue_A_data[0]
        # conversation_B = dialogue_B_data[0]

        # pair_eval_response = pair_eval(conversation_A, conversation_B, "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", "gpt-4o")


        # print(pair_eval_response)

        for dialoue_index in range(len(dialogue_A_data)):
        # for dialoue_index in range(939, len(dialogue_A_data)):

            print("dialogue A: " + dialogue_A_file_name)
            print("dialogue B: " + dialogue_B_file_name)
            print("dialoue_index: " + str(dialoue_index))

            conversation_A = dialogue_A_data[dialoue_index]["conversation"]
            conversation_B = dialogue_B_data[dialoue_index]["conversation"]

            # print(type(conversation_A))
            # print(type(conversation_A[0]))


            # pair_eval_response = pair_eval(conversation_A, conversation_B, "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", "gpt-4o")

            # json_obj = json.loads(pair_eval_response)
            # saved_result_file.write(json.dumps(json_obj) + "\n")

            
            try_times = 0
            write_succeed = False
            while write_succeed == False:

                gt_eval_geminiflash_response = pair_eval_geminishflash(conversation_A, conversation_B)
                raw_content = gt_eval_geminiflash_response.content

                # print(raw_content)

                # Extract JSON block safely
                match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                if match:
                    try:
                        json_str = match.group(0)
                        # json_str = json_str.replace("\\", "")
                        print(json_str)
                        # x = clean_reason_quotes(json_str)
                        # print(x)
                        # json_str = json_str.replace('"im the goat"', "'im the goat'")


                        try:
                            parsed_json = json.loads(json_str)
                        except:
                            try:
                                clean_double_qoute_json_str = clean_quotes(json_str)
                                print("clean double qoutes:")
                                print(clean_double_qoute_json_str)
                                parsed_json = json.loads(clean_double_qoute_json_str)
                            except:
                                clean_double_qoute_slash_json_str = clean_double_qoute_json_str.replace("\\", "")
                                print("clean double qoutes and slash:")
                                print(clean_double_qoute_slash_json_str)
                                parsed_json = json.loads(clean_double_qoute_slash_json_str)                                                              

                        #print(parsed_json)
                        saved_result_file.write(json.dumps(parsed_json) + "\n")
                        write_succeed = True
                    except:
                        try:
                            parsed_json = ast.literal_eval(json_str)
                            #print(parsed_json)
                            saved_result_file.write(json.dumps(parsed_json) + "\n")
                            write_succeed = True
                        except:
                            try_times = try_times + 1
                            if try_times < 10:
                                print("wait for 10s before the next run, already run %d times" % try_times)
                                time.sleep(10)
                                continue
                            else:
                                sys.exit("Tryied 10 times but still encounter errors.")

                else:
                    # sys.exit("No JSON object found.")
                    try_times = try_times + 1
                    if try_times < 10:
                        print("No JSON object found. Wait for 10s before the next run, already run %d times" % try_times)
                        time.sleep(10)
                        continue
                    else:
                        sys.exit("Try Ten times and No JSON object found.")


