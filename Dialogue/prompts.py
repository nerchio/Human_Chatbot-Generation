ROLES = ["Human", "Older", "Younger", "Foreigner"]

INQUIRER_PROMPT = "Please act like you are a human chatting with the chatbot. Say something according to the above context\
        to complete the conversation. The length limit is 10 words."

RESPONDER_PROMPT = "You are a helpful assistant. Please answer the question no longer than 10 words."

def GET_PROMPT(role=0):
    return INQUIRER_PROMPT, RESPONDER_PROMPT