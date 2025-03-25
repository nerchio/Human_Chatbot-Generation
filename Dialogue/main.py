from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
from models import *
from prompts import GET_PROMPT
import pandas as pd
from tqdm import tqdm

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
    if state['turns'] >= 10:
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
image_bytes = graph.get_graph().draw_mermaid_png()
image = PILImage.open(io.BytesIO(image_bytes))
image.save("/work/courses/dslab/team3/Human_Chatbot-Generation/Dialogue/graph.png")


# INFERENCE PART

inquirer_llm = GPT4o
responder_llm = GPT4oMini
INQUIRER_SYSTEM_PROMPT, INQUIRER_PROMPT, RESPONDER_SYSTEM_PROMPT = GET_PROMPT()

def graph_update(qa_history: list):
    messages = []
    for index, text in enumerate(qa_history):
        if index % 2 == 0:
            messages.append(HumanMessage(content=text, additional_kwargs={"source": "qa_history"}))
        else:
            messages.append(AIMessage(content=text, additional_kwargs={"source": "qa_history"}))
    initial_state = {"messages": messages, 
                    "turns": len(messages), 
                    "inquirer_system_prompt": INQUIRER_SYSTEM_PROMPT,
                    "inquirer_prompt": INQUIRER_PROMPT,
                    "responder_system_prompt": RESPONDER_SYSTEM_PROMPT}
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    df = pd.read_csv('data/qa_test.csv')
    results = pd.DataFrame(columns=['seed','dialogues'])
    for index, row in tqdm(df.iterrows(), total=len(df)):
        qa_history = [row['question'], row['answer']]
        final_state = graph_update(qa_history)
        # for message in final_state["messages"]:
        #     if message.additional_kwargs["source"] == "qa_history":
        #         if message.type == "human":
        #             print("Human (Seed): " + message.content)
        #         elif message.type == "ai":
        #             print("Chatbot (Seed): " + message.content)
        #     else:
        #         if message.type == "human":
        #             print("Human (Generated): " + message.content)
        #         elif message.type == "ai":
        #             print("Chatbot (Generated): " + message.content)
        results.loc[index] = [qa_history, [m.content for m in final_state["messages"]]]
    
    results.to_csv('results/results_qa_test.csv', index=False)