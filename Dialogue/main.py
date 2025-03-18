from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from PIL import Image as PILImage
import io
from models import *
from prompts import GET_PROMPT

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    turns: int
    inquirer_prompt: str
    responder_prompt: str

def inquirer(state: State):
    # system_message = SystemMessage(content="You are an inquirer. Please ask a question about ETHZ.")
    inquirer_prompt = HumanMessage(content=state["inquirer_prompt"])
    inquirer_message = state["messages"] + [inquirer_prompt]
    # print("inquirer_message", inquirer_message)
    inquirer_response = inquirer_llm.invoke(inquirer_message)
    # print("llm_response", llm_response)
    return {"messages": [HumanMessage(content=inquirer_response.content)], "turns": state["turns"] + 1}

def responder(state: State):
    responder_prompt = SystemMessage(content=state["responder_prompt"])
    responder_message = [responder_prompt] + state["messages"]
    return {"messages": [responder_llm.invoke(responder_message)], "turns": state["turns"] + 1}

def max_turns_condition(state: State):
    if state['turns'] >= 7:
        return END
    else:
        return "inquirer"

def proper_content_condition(state: State):
    if False:
        return END
    else:
        return "responder"

graph_builder = StateGraph(State)
graph_builder.add_node("inquirer", inquirer)
graph_builder.add_node("responder", responder)

graph_builder.add_edge(START, "responder")
graph_builder.add_conditional_edges("responder", max_turns_condition)
graph_builder.add_conditional_edges("inquirer", proper_content_condition)

graph = graph_builder.compile()

# draw the graph
# image_bytes = graph.get_graph().draw_mermaid_png()
# image = PILImage.open(io.BytesIO(image_bytes))
# image.save("/work/courses/dslab/team3/Human_Chatbot-Generation/Dialogue/graph.png")


# INFERENCE PART

inquirer_llm = DeepSeek
responder_llm = GPT4oMini
INQUIRER_PROMPT, RESPONDER_PROMPT = GET_PROMPT()

def stream_graph_updates(user_input: str):
    print("Human: " + user_input)
    initial_state = {"messages": [{"role": "user", "content": user_input}], 
                    "turns": 0, 
                    "inquirer_prompt": INQUIRER_PROMPT, 
                    "responder_prompt": RESPONDER_PROMPT}
    for event in graph.stream(initial_state):
        for value in event.values():
            if value["messages"][-1].type == "human":
                print("(AI) Human:", value["messages"][-1].content)
            elif value["messages"][-1].type == "ai":
                print("Chatbot:", value["messages"][-1].content)
            else:
                raise ValueError("Unknown message type")

        # fallback if input() is not available
if __name__ == "__main__":
    user_input = "Hello! Could you tell me something about RTX4090?"
    stream_graph_updates(user_input)