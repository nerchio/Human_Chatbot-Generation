from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
from models import *
from prompts import GET_EVALUATOER_PROMPT
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




# class uni_eval_state(TypedDict):
#     messages: Annotated[list, add_messages]
#     turns: int
#     evaluator_system_prompt: str
#     evaluator_prompt: str
#     uni_eval_response: str

class uni_eval_state(TypedDict):
    messages: list
    turns: int
    evaluator_system_prompt: str
    evaluator_prompt: str
    uni_eval_response: str

def uni_evaluator(state: uni_eval_state):
    evaluator_system_prompt = SystemMessage(content=state["evaluator_system_prompt"])
    evaluator_prompt = HumanMessage(content=state["evaluator_prompt"])
    evaluator_message = [evaluator_system_prompt] + state["messages"] + [evaluator_prompt]

    # print(evaluator_message)

    evaluator_response = uni_evaluator_llm.invoke(evaluator_message)

    # evaluator_response = ""

    return {"turns": state["turns"] + 1,
            "uni_eval_response": evaluator_response}

# Judge if the uni_evaluator's response satisfies the output format
# But ChatGPT4 may not need this
def uni_eval_stop_judger(state: uni_eval_state):
    eval_response = uni_eval_state["uni_eval_response"]


EVALUATOR_SYSTEM_PROMPT, UNI_EVALUATOR_PROMPT = GET_EVALUATOER_PROMPT()

# Uni-eval
uni_eval_graph_builder = StateGraph(uni_eval_state)
uni_eval_graph_builder.add_node("uni_evaluator", uni_evaluator)

uni_eval_graph_builder.add_edge(START, "uni_evaluator")
uni_eval_graph_builder.add_edge("uni_evaluator", END)

uni_eval_graph = uni_eval_graph_builder.compile()

def uni_eval_graph_update(conversation: list):
    messages = []
    for index, entry in enumerate(conversation):
        if index % 2 == 0:
        #     messages.append(HumanMessage(content=entry["content"], additional_kwargs={"source": "conversation"}))
            messages.append("HH: " + entry["content"])

        else:
        #     messages.append(AIMessage(content=entry["content"], additional_kwargs={"source": "conversation"}))
            messages.append("CC: " + entry["content"])

    
    # print(messages)
    # print(type(messages))
    # print(len(messages))

    # print(messages[0])
    # print(messages[1])
    # print(messages[2])
    # print(messages[3])
    initial_state = {"messages": messages,
                     "turns": 0,
                     "evaluator_system_prompt": EVALUATOR_SYSTEM_PROMPT,
                     "evaluator_prompt": UNI_EVALUATOR_PROMPT,
                     "uni_eval_response": ""

    }

    final_state = uni_eval_graph.invoke(initial_state)

    return final_state
    




uni_evaluator_llm = GPT4o



if __name__ == "__main__":
    saved_result_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/result/uni_eval_DeepSeek_prompt1.jsonl"
    saved_result_file = open(saved_result_file_path, "w", encoding="utf-8")

    dialogue_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/oasst1_en_DeepSeek_GPT4oMini_6.jsonl"

    dialogue_data = parse_jsonl(dialogue_data_path)

    dialogue_index = 1
    # Print or process the parsed data
    for entry in dialogue_data:  # Display the first two entries for verification

        print("Dialogue: " + str(dialogue_index))
        dialogue_index = dialogue_index + 1

        # print(json.dumps(entry, indent=4))
        # print(entry.keys())
        conversation = entry["conversation"]
        # print(type(conversation))
        # print(len(conversation))
        # print(conversation[0])

        final_state = uni_eval_graph_update(conversation)

        # # Use a regex pattern to extract all key-value pairs
        # matches = re.findall(r'(\w+): (.*?)(?=\n\w+:|\Z)', final_state["uni_eval_response"].content, re.DOTALL)


        # # Convert to dictionary
        # uni_eval_result = {key: value.strip() for key, value in matches}

        # print(type(final_state["uni_eval_response"].content))
        # print(final_state["uni_eval_response"].content)

        # print(type(final_state["uni_eval_response"]))
        # print(final_state["uni_eval_response"])
        # print(type(vars(final_state["uni_eval_response"])))
        # print(vars(final_state["uni_eval_response"]))


        saved_result_file.write(json.dumps(vars(final_state["uni_eval_response"])) + "\n")




