ROLES = ["Human", "Older", "Younger", "Foreigner"]

INQUIRER_SYSTEM_PROMPT = "You are a helpful assistant. Please follow the given instructions."

INQUIRER_PROMPT = "The conversation above is from a conversation between a human and a chatbot. Now please play the role of the human.\
Your task is to determine whether the chatbot's answer clarify your initial question clearly and provide detailed information. If it does, please output <EOD>.\
Otherwise, please comment on the answer and ask a follow-up question to continue the conversation. Do not make statements. Output in human style. Do not output in chatbot style."

RESPONDER_SYSTEM_PROMPT = "You are a helpful assistant. Your output should be no longer than 50 tokens."

def GET_PROMPT(role=0):
    return INQUIRER_SYSTEM_PROMPT, INQUIRER_PROMPT, RESPONDER_SYSTEM_PROMPT