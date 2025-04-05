from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
from models import *
from prompts import GET_PAIR_EVALUATOER_PROMPT
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


class pair_eval_state(TypedDict):
    messages: list
    turns: int
    evaluator_system_prompt: str
    evaluator_prompt: str
    pair_eval_response: str


def pair_evaluator(state: pair_eval_state):
    evaluator_system_prompt = SystemMessage(content=state["evaluator_system_prompt"])
    evaluator_prompt = HumanMessage(content=state["evaluator_prompt"])
    evaluator_message = [evaluator_system_prompt] + state["messages"] + [evaluator_prompt]

    # print(evaluator_message)

    evaluator_response = pair_evaluator_llm.invoke(evaluator_message)

    # evaluator_response = ""

    return {"turns": state["turns"] + 1,
            "pair_eval_response": evaluator_response}

PAIR_EVALUATOR_SYSTEM_PROMPT, PAIR_EVALUATOR_PROMPT = GET_PAIR_EVALUATOER_PROMPT()

# Uni-eval
pair_eval_graph_builder = StateGraph(pair_eval_state)
pair_eval_graph_builder.add_node("pair_evaluator", pair_evaluator)

pair_eval_graph_builder.add_edge(START, "pair_evaluator")
pair_eval_graph_builder.add_edge("pair_evaluator", END)

pair_eval_graph = pair_eval_graph_builder.compile()



def pair_eval_graph_update(conversationA: list, conversationB: list):
    messages = []
    for index, entry in enumerate(conversationA):
        if index % 2 == 0:
        #     messages.append(HumanMessage(content=entry["content"], additional_kwargs={"source": "conversation"}))
            messages.append("ConversationA HH: " + entry["content"])

        else:
        #     messages.append(AIMessage(content=entry["content"], additional_kwargs={"source": "conversation"}))
            messages.append("ConversationA CC: " + entry["content"])
    
    for index, entry in enumerate(conversationB):
        if index % 2 == 0:
        #     messages.append(HumanMessage(content=entry["content"], additional_kwargs={"source": "conversation"}))
            messages.append("ConversationB HH: " + entry["content"])

        else:
        #     messages.append(AIMessage(content=entry["content"], additional_kwargs={"source": "conversation"}))
            messages.append("ConversationB CC: " + entry["content"])    

    
    initial_state = {"messages": messages,
                     "turns": 0,
                     "evaluator_system_prompt": PAIR_EVALUATOR_SYSTEM_PROMPT,
                     "evaluator_prompt": PAIR_EVALUATOR_PROMPT,
                     "pair_eval_response": ""

    }

    final_state = pair_eval_graph.invoke(initial_state)

    return final_state



pair_evaluator_llm =GPT4o

if __name__ == "__main__":
    saved_result_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/result/pair_eval_A_GPT4o_B_llama_70B_prompt1.jsonl"
    saved_result_file = open(saved_result_file_path, "w", encoding="utf-8")

    dialogueA_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/oasst1_en_GPT4o_GPT4oMini_6.jsonl"
    dialogueB_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/oasst1_en_llama_70B_GPT4oMini_6.jsonl"

    dialogueA_data = parse_jsonl(dialogueA_data_path)
    dialogueB_data = parse_jsonl(dialogueB_data_path)

    dialogue_index = 1

    # Print or process the parsed data
    for entry_index in range(len(dialogueA_data)):
    # for entry_index in range(10):
        entryA = dialogueA_data[entry_index]
        entryB = dialogueB_data[entry_index]

        print("Dialogue: " + str(dialogue_index))
        dialogue_index = dialogue_index + 1

        conversationA = entryA["conversation"]
        conversationB = entryB["conversation"]

        final_state = pair_eval_graph_update(conversationA, conversationB)

        saved_result_file.write(json.dumps(vars(final_state["pair_eval_response"])) + "\n")