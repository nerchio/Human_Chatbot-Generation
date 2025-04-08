import json

__all__ = ['create_task_summary_prompt', 'conversation_generation_prompt', 'create_unieval_prompt', 'create_paireval_prompt', 'create_finetuning_user_prompt', 'FT_SYSTEM_MESSAGE']

def create_task_summary_prompt(conversation: list[dict]) -> str:
    """
    Creates the prompt for summarizing the task from the conversation.
    """
    prompt = (
        "You are given a conversation between a human and a bot. \n"
        "Summarize the task or request that the human wants the bot to resolve concisely. \n"
        "The summary should clearly state what the human is requesting the bot to do, using active voice and phrasing such as "
        "\"You are asking the bot to ...\" or \"You want to find out ...\". \n"
        "Include all relevant details, topics, and specific areas the human would like to learn about or explore.\n"
        "Present the summary as instructions for someone else to simulate the same requests to the bot.\n"
        "Don't reveal or leak the bot's answer in the summary, focus on the human's intention.:\n\n"
    )

    for turn in conversation:
        role = turn['role']
        content = turn['content']
        prompt += f"{role.capitalize()}: {content}\n"

    prompt += "\nTask summary:"
    return prompt

def conversation_generation_prompt(n_turns:int, task_summary:str, seed_conversation:str):
    example_json = {"generated_conversation": [
            {"role": "human", "content": "Example user message"},
            {"role": "bot", "content": "Assistant response"}
        ]}
    return f"""
You are supposed to generate a conversation of human chatting with AI aiming to solve the following task:\n
{task_summary}.
\n
You are provided with a seed conversation to start the conversation:\n
{seed_conversation}.
\n
When you are simulating human messages, you are trying to pass the Turing test, which means you need to speak like human as much as possible. 
The conversation flow should be natural and smooth. You can switch to some other topics if you want, but the transition should be natural.

When you are simulating an AI bot responsing human messages, note that you are chatting with human, so do not say too many words in each round (less than 60 words is recommended), and talk like an AI assistant.
\n
Assume the number of turns as the number of messages, thus, one human message and one bot response are considered as 2 turns. 
The generated conversation including the seed_conversation should be {n_turns} turns. 

Return your response strictly formatted as JSON, exactly following this structure (with double quotes):

{json.dumps(example_json, indent=2)}
"""

def create_unieval_prompt(conversation: list[dict]) -> str:
    """
    Generates a UniEval (individual dialogue evaluation) prompt for evaluating conversations to determine if they involve AI-generated content.
    """

    example_json = {
      "choice": "No",
      "index": "None",
      "reason": "BlahBlah"
    }
    
    return f"""
    You are an AI assistant who helps human do the Turing test more easily. 
    You will be provided with a conversation between human and bot, and you need to judge if the conversation is AI generated: \n
    
    {conversation}

    \n
    The conversation format is given as list of dictionaries as follows: \n
      [
      {{
        "role": "human",
        "content": "How can I learn to optimize my webpage for search engines?"
      }},
      {{
        "role": "bot",
        "content": "##### Resources for learning search engine optimization (SEO)\n\nA good place to start with learning about SEO is from search engine companies and browser manufacturers. You could start by reading guides from [Google](https://developers.google.com/search/docs/fundamentals/seo-starter-guide), [Bing](https://www.bing.com/webmasters/help/webmaster-guidelines-30fba23a) and [Mozilla](https://developer.mozilla.org/en-US/docs/Glossary/SEO).\n\nThere are many other resources available on SEO. You could search YouTube for videos by experts on SEO, undertake a free course on a site like [HubSpot](https://academy.hubspot.com/courses/seo-training), or take a paid course offered by an educational institution or local marketing association.\n\n##### (Very) brief introduction to SEO\n\nSEO is a collection of techniques the owner of a website can use to improve their site's placement in search engine results, increasing organic (non-paid) traffic to the site. These techniques try to convince search engine algorithms that the website is a quality source of information that will be useful to the search engine user. \n\nThere are several aspects to SEO, including:\n\n* Keyword research\n* Link building and backlinks\n* Optimising page content\n* Ongoing analysis and metrics"
      }}] \n
    Each utterance is a dictionary with two keys: "role" and "content". The value of "role" is either "human" or "bot", and the value of "content" is the text of the message. \n
    The "human" utterance is the message sent by the human, and the "bot" utterance is the response from the AI assistant.\n
    
    ### Evaluation criteria:
    Focus only on human utterances. Consider a human utterance to be **AI-generated** if it exhibits any of the following characteristics:
    1. the sentence is not as fluent as human conversation; 
    2. the sentence is too long or too and is not likely to appear in human-bot conversations; 
    3. the sentence is too perfect to be created by human, since humans often use natural, sometimes inconsistent phrasing, typos, slang, or emotional nuance.
    3. the sentence is not related to the context or the change of topic is not natural; 
    4. the sentence just repeat some previous sentences (exact repetition or with the same meaning). 
    
    \n
    ### Output instructions:

    Print "Choice: No" if you think the conversation is not AI generated, i.e. there is no "human" utterance generated by AI; or print "Choice: Yes" if you think the conversation is AI generated, i.e. there is "human" utterance generated by AI. \n

    If you print "Choice: Yes", you need also print a number (start from 1, use the format "Index: n" [1 <= n <= the number of chats]) in the new line, 
    indicating the index of the first chat that you think is generated by AI. \n

    You also need to provide your reason for your choice.\n
    Your response should use the following format: \n
    - `choice`: `"Yes"` if you believe any human utterance is AI-generated; otherwise, `"No"`.
    - `index`: If `choice` is `"Yes"`, specify the **1-based index** of the first suspicious human utterance. If `choice` is `"No"`, use `"None"`.
    - `reason`: Briefly explain your decision.

    Return your response strictly formatted as JSON, exactly following this structure (with double quotes):

    {json.dumps(example_json, indent=2)}
    """

def create_paireval_prompt(conversation_1: list[dict], conversation_2: list[dict]) -> str:
    """
    Generates a PairEval (comparative evaluation of two dialogues) prompt for evaluating two conversations to determine if they involve AI-generated content.
    """
    example_json = {
      "choice": "Conversation 1",
      "reason": "BlahBlah"
    }
    return f"""
You are an AI assistant who helps human do the Turing test more easily. \n
You will be provided with two conversations. They are both conversations between human and bot, and there can be AI-generated human utterances in each conversation.\n 
You need to read both conversations and judge if two conversations have AI generated human utterances.\n

Conversation is given in the following format:\n
      [
      {{
        "role": "human",
        "content": "How can I learn to optimize my webpage for search engines?"
      }},
      {{
        "role": "bot",
        "content": "##### Resources for learning search engine optimization (SEO)\n\nA good place to start with learning about SEO is from search engine companies and browser manufacturers. You could start by reading guides from [Google](https://developers.google.com/search/docs/fundamentals/seo-starter-guide), [Bing](https://www.bing.com/webmasters/help/webmaster-guidelines-30fba23a) and [Mozilla](https://developer.mozilla.org/en-US/docs/Glossary/SEO).\n\nThere are many other resources available on SEO. You could search YouTube for videos by experts on SEO, undertake a free course on a site like [HubSpot](https://academy.hubspot.com/courses/seo-training), or take a paid course offered by an educational institution or local marketing association.\n\n##### (Very) brief introduction to SEO\n\nSEO is a collection of techniques the owner of a website can use to improve their site's placement in search engine results, increasing organic (non-paid) traffic to the site. These techniques try to convince search engine algorithms that the website is a quality source of information that will be useful to the search engine user. \n\nThere are several aspects to SEO, including:\n\n* Keyword research\n* Link building and backlinks\n* Optimising page content\n* Ongoing analysis and metrics"
      }}] \n

Each utterance is a dictionary with two keys: "role" and "content". The value of "role" is either "human" or "bot", and the value of "content" is the text of the message. \n
The "human" utterance is the message sent by the human, and the "bot" utterance is the response from the AI assistant. 

### Evaluation criteria:
Focus only on human utterances. Consider a human utterance to be **AI-generated** if it exhibits any of the following characteristics:
1. the sentence is not as fluent as human conversation; 
2. the sentence is too long or too and is not likely to appear in human-bot conversations; 
3. the sentence is too perfect to be created by human, since humans often use natural, sometimes inconsistent phrasing, typos, slang, or emotional nuance.
3. the sentence is not related to the context or the change of topic is not natural; 
4. the sentence just repeat some previous sentences (exact repetition or with the same meaning).\n

Here are the two conversations:\n
Conversation 1:\n
{conversation_1}
\n
Conversation 2:\n
{conversation_2}

If you think only Conversation 1 has AI generated human utterances, choose "Conversation 1" in your response.\n
If you think only Conversation 2 has AI generated human utterances, choose "Conversation 2" in your response.\n
If you think both conversations have AI generated human utterances, choose "Both" in your response.\n
If you think no conversation has AI generated human utterances, choose "Neither" in your response.\n

You also need to provide your reason for your choice.\n

### Output instructions:
Return your response strictly formatted as JSON, exactly following this structure (with double quotes):

{json.dumps(example_json, indent=2)}

The `choice` field must be one of the following:
- "Conversation 1"
- "Conversation 2"
- "Both"
- "Neither"

The `reason` field must briefly explain your reasoning.
"""


def create_finetuning_user_prompt(n_turns:int, task_summary:str, seed_conversation:str) -> str:
    """User prompt for openai fine-tuned model."""
    
    return f"""Generate a conversation of human chatting with AI where the human is aiming to solve the following task:\n
{task_summary}. \n
Here is a seed conversation with 2 utterances to start the conversation:\n
{seed_conversation}. \n
Assume the number of turns as the number of messages, thus, one human message and one bot response are considered as 2 turns. 
The generated conversation including the seed_conversation should be {n_turns} turns. """

example_json = {"generated_conversation": [
            {"role": "human", "content": "Example user message"},
            {"role": "bot", "content": "Assistant response"}
        ]}

FT_SYSTEM_MESSAGE = """You are supposed to generate a conversation of human chatting with AI aiming to follow the instruction given by the user. 
When you are simulating human messages, you are simulating natural-sounding human-AI conversations. 
Make human messages feel realistic and casual, which means you need to speak like a human as much as possible. 
\n The conversation flow should be natural and smooth. You can switch to some other topics if you want, but the transition should be natural.
\n When you are simulating an AI bot responding to human messages, note that you are chatting with a human, so do not say too many words in each round (less than 60 words is recommended), and talk like an AI assistant.
\n Output your response in strict JSON format, exactly matching the structure: {\"generated_conversation\": [ {\"role\": \"human\", \"content\": \"...\"}, {\"role\": \"bot\", \"content\": \"...\"}, ... ] }"""

