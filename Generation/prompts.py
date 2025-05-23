ROLES = ["Human", "Older", "Younger", "Foreigner"]

INQUIRER_SYSTEM_PROMPT = '''
You are required to act as a human. You are chatting with a chatbot to get some knowledge. What you want to know is described in <task>.

<task>
{task}
</task>
'''

# INQUIRER_PROMPT = '''
# The conversation above is the chat history between you and the chatbot. 
# Your task is to determine whether the chatbot provides enough knowledge for what you want to know in <task>. You don't need to provide any analysis. 
# If true, please output "<EOD>". Do not output anything else.
# If false, please output a follow-up question to continue the conversation. 

# ** Important: Your output must be in human style instead of chatbot style. **
# '''

INQUIRER_PROMPT = '''
The conversation above is the chat history between you and the chatbot. 
Your task is to perform as a human to continue the chat. The topic of the chat is described in <task>. You could ask anything you want to know about the topic.

** Important: Your output must be in human style instead of chatbot style. **
'''

RESPONDER_SYSTEM_PROMPT = "You are a helpful assistant. Your output should be no longer than 50 tokens."

def GET_PROMPT(task:str, role:int=0):
    return INQUIRER_SYSTEM_PROMPT.format(task=task), INQUIRER_PROMPT, RESPONDER_SYSTEM_PROMPT