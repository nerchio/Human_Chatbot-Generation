from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
from models import *
from prompts import GET_PROMPT
from tqdm import tqdm
import argparse
import jsonlines
from langchain_community.callbacks.manager import get_openai_callback
from utils import UniversalTokenCounter

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--inquirer_model", type=str, default="GPT4o")
    args.add_argument("--responder_model", type=str, default="GPT4oMini")
    args.add_argument("--data", type=str, default="oasst1_en")
    args.add_argument("--output_path", type=str, default="results/")
    args.add_argument("--max_turns", type=int, default=6)
    args = args.parse_args()


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    turns: int
    inquirer_system_prompt: str
    inquirer_prompt: str
    responder_system_prompt: str

def inquirer(state: State):
    inquirer_system_prompt = SystemMessage(content=state["inquirer_system_prompt"])
    inquirer_prompt = HumanMessage(content=state["inquirer_prompt"])
    inquirer_message = [inquirer_system_prompt] + state["messages"] + [inquirer_prompt]
    inquirer_response = inquirer_llm.invoke(inquirer_message)
    return {"messages": [HumanMessage(
        content=inquirer_response.content,
        additional_kwargs={"source": "generated"}
    )], "turns": state["turns"] + 1}

def responder(state: State):
    responder_prompt = SystemMessage(content=state["responder_system_prompt"])
    responder_message = [responder_prompt] + state["messages"]
    response = responder_llm.invoke(responder_message)
    return {"messages": [AIMessage(
        content=response.content,
        additional_kwargs={"source": "generated"}
    )], "turns": state["turns"] + 1}

def max_turns_condition(state: State):
    if state['turns'] >= args.max_turns:
        return END
    else:
        return "inquirer"

def content_condition(state: State):
    content = state["messages"][-1].content
    if '<EOD>' in content:
        return END
    else:
        return "responder"

graph_builder = StateGraph(State)
graph_builder.add_node("inquirer", inquirer)
graph_builder.add_node("responder", responder)

graph_builder.add_edge(START, "inquirer")
graph_builder.add_conditional_edges("responder", max_turns_condition)
graph_builder.add_conditional_edges("inquirer", content_condition)

graph = graph_builder.compile()

# raw the graph
# image_bytes = graph.get_graph().draw_mermaid_png()
# image = PILImage.open(io.BytesIO(image_bytes))
# image.save("/work/courses/dslab/team3/Human_Chatbot-Generation/Dialogue/graph.png")


# INFERENCE PART
inquirer_llm = globals()[args.inquirer_model]
responder_llm = globals()[args.responder_model]


def graph_update(qa_history: list, task_summary: str):
    messages = []
    for detail in qa_history:
        if detail['role'] == 'human':
            messages.append(HumanMessage(content=detail['content'], additional_kwargs={"source": "qa_history"}))
        elif detail['role'] == 'bot':
            messages.append(AIMessage(content=detail['content'], additional_kwargs={"source": "qa_history"}))
        else:
            raise ValueError("Invalid role")

    INQUIRER_SYSTEM_PROMPT, INQUIRER_PROMPT, RESPONDER_SYSTEM_PROMPT = GET_PROMPT(task_summary)
    initial_state = {"messages": messages, 
                    "turns": len(messages), 
                    "inquirer_system_prompt": INQUIRER_SYSTEM_PROMPT,
                    "inquirer_prompt": INQUIRER_PROMPT,
                    "responder_system_prompt": RESPONDER_SYSTEM_PROMPT}
    
    try:
        with get_openai_callback() as cb:
            final_state = graph.invoke(initial_state)
            token_usage = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens
            }
            print(f"\nOpenAI callback")
    except Exception as e:
        print(f"\nOpenAI callback not available, using UniversalTokenCounter: {str(e)}")
        token_counter = UniversalTokenCounter()
        inquirer_llm.callbacks = [token_counter]
        responder_llm.callbacks = [token_counter]
        final_state = graph.invoke(initial_state)
        token_usage = token_counter.get_stats()
    
    return final_state, token_usage


if __name__ == "__main__":
    data_path = ""
    if args.data == "oasst1_en":
        data_path = "data/oasst1_en_min_6_turns_summary.jsonl"
    else:
        raise ValueError("Invalid data")
    data = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            data.append(obj)
    generated_data = []

    for dialogue in tqdm(data, total=len(data)):
        seed = dialogue['conversation_id'][:2]
        task_summary = dialogue['task_summary']
        seed_conversation = dialogue['conversation'][:2]
        final_state, token_usage = graph_update(seed_conversation, task_summary)

        # for message in final_state["messages"]:
        #     if message.additional_kwargs["source"] == "qa_history":
        #         if isinstance(message, HumanMessage):
        #             print("Human (Seed): " + message.content)
        #         elif isinstance(message, AIMessage):
        #             print("Chatbot (Seed): " + message.content)
        #     else:
        #         if isinstance(message, HumanMessage):
        #             print("Human (Generated): " + message.content)
        #         elif isinstance(message, AIMessage):
        #             print("Chatbot (Generated): " + message.content)
        
        generated_conversation = []
        for message in final_state["messages"]:
            if message.additional_kwargs["source"] == "generated":
                if isinstance(message, HumanMessage):
                    generated_conversation.append({
                        'role': 'human',
                        'content': message.content
                    })
                elif isinstance(message, AIMessage):
                    generated_conversation.append({
                        'role': 'bot',
                        'content': message.content
                    })
        generated_dialogue = {
            'conversation_id': seed + ['']*(len(final_state["messages"])-len(seed)),
            'conversation': seed_conversation + generated_conversation,
            'turns': len(final_state["messages"]),
            'task_summary': task_summary,
            'inquirer_model': args.inquirer_model,
            'responder_model': args.responder_model,
            'token_usage': token_usage
        }
        generated_data.append(generated_dialogue)
    
    output_path = args.output_path + args.data + "_" + args.inquirer_model + "_" + args.responder_model + "_" + str(args.max_turns) + ".jsonl"
    with jsonlines.open(output_path, mode='w') as writer:
        for dialogue in generated_data:
            writer.write(dialogue)
