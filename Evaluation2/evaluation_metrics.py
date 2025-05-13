from prompts import create_unieval_prompt, create_paireval_prompt
import openai

from models import *

def uni_eval(conversation:list[dict], openai_api_key:str, model:str="gpt-4o") -> str:
    """
    Uses the UniEval prompt to evaluate a conversation.

    Args:
        conversation (list[dict]): The conversation to evaluate. The format is:
            [
                {"role": "human", "content": "Hello!"},
                {"role": "bot", "content": "Hi! How can I help you today?"}]
        openai_api_key (str): The OpenAI API key.
        model (str): The model to use as the judge for evaluation.
    """
    prompt = create_unieval_prompt(conversation)
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,  # Or whichever model you prefer,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def uni_eval_Deepseek(conversation:list[dict]) -> str:
    prompt = create_unieval_prompt(conversation)

    response = DeepSeek.invoke(prompt)

    return response

def uni_eval_llama3_2_70B(conversation:list[dict]) -> str:
    prompt = create_unieval_prompt(conversation)

    response = llama3_3_70B.invoke(prompt)

    return response

def uni_eval_claude(conversation:list[dict]) -> str:
    prompt = create_unieval_prompt(conversation)

    response = Claude3_7.invoke(prompt)

    return response

def uni_eval_geminiflash(conversation:list[dict]) -> str:
    prompt = create_unieval_prompt(conversation)

    response = GeminiFlash.invoke(prompt)

    return response

def uni_eval_claude(conversation:list[dict]) -> str:
    prompt = create_unieval_prompt(conversation)

    response = Claude3_7.invoke(prompt)

    return response


def pair_eval(conversation_1:list[dict], conversation_2:list[dict], openai_api_key:str, model:str="gpt-4o") -> str:
    """
    Uses the PairEval prompt to evaluate two conversations.
    
    Args:
        conversation_1 (list[dict]): The conversation from one model to evaluate.
        conversation_2 (list[dict]): The conversation from another model evaluate.
        openai_api_key (str): The OpenAI API key.
        model (str): The model to use as the judge for evaluation.
    """
    prompt = create_paireval_prompt(conversation_1, conversation_2)
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,  # Or whichever model you prefer,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content

def pair_eval_Deepseek(conversation_1:list[dict], conversation_2:list[dict]) -> str:
    prompt = create_paireval_prompt(conversation_1, conversation_2)

    response = DeepSeek.invoke(prompt)

    return response


def pair_eval_geminishflash(conversation_1:list[dict], conversation_2:list[dict]) -> str:
    prompt = create_paireval_prompt(conversation_1, conversation_2)

    response = GeminiFlash.invoke(prompt)

    return response

def gt_eval(true_conversation:list[dict], generated_conversation:list[dict], openai_api_key:str, model="gpt-4o") -> str:
    """
    Wrap the Pair-Eval prompt to evaluate a ground truth comparison.
    Args:
        true_conversation (list[dict]): The ground truth conversation.
        generated_conversation (list[dict]): The generated conversation to evaluate.
        openai_api_key (str): The OpenAI API key.
        model (str): The model to use as the judge for evaluation.
    
    """
    return pair_eval(true_conversation, generated_conversation, openai_api_key, model)

def gt_eval_Deepseek(true_conversation:list[dict], generated_conversation:list[dict]) -> str:

    return pair_eval_Deepseek(true_conversation, generated_conversation)

def gt_eval_geminiflash(true_conversation:list[dict], generated_conversation:list[dict]) -> str:

    return pair_eval_geminishflash(true_conversation, generated_conversation)