import json


input_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/parsed_result/pair_eval_A_GPT4o_B_llama_70B_prompt1.jsonl"
output_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/result_summary/pair_eval_A_GPT4o_B_llama_70B_prompt1.txt"


num_total_dialogues = 0
num_choice_A = 0
num_choice_B = 0
num_choice_Both = 0
num_choice_Neither = 0
num_completion_tokens = 0
num_prompt_tokens = 0

# Open and read the JSONL file line by line
with open(input_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Parse each line as a JSON object
        data = json.loads(line)
        
        choice = data.get("Choice")
        completion_tokens = data.get("completion_tokens")
        prompt_tokens = data.get("prompt_tokens")

        num_total_dialogues = num_total_dialogues + 1
        if choice == "A":
            num_choice_A = num_choice_A + 1
        elif choice == "B":
            num_choice_B = num_choice_B + 1
        elif choice == "Both":
            num_choice_Both = num_choice_Both + 1
        elif choice == "Neither":
            num_choice_Neither = num_choice_Neither + 1     

        num_completion_tokens = num_completion_tokens + int(completion_tokens)
        num_prompt_tokens = num_prompt_tokens + int(prompt_tokens)

# Write the list of dicts to a new JSONL file
with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("A Better Rate: " + str(num_choice_B/num_total_dialogues) + "\n")
        f.write("B Better Rate: " + str(num_choice_A/num_total_dialogues) + "\n")
        f.write("Both Failed Rate: " + str(num_choice_Both/num_total_dialogues) + "\n")
        f.write("Both Passed Rate: " + str(num_choice_Neither/num_total_dialogues) + "\n")                
        f.write("# Total Dialogues: " + str(num_total_dialogues) + "\n")
        f.write("# completion prompts: " + str(num_completion_tokens) + "\n")
        f.write("# prompt prompts: " + str(num_prompt_tokens) + "\n")
        f.write("Cost: "+ str(num_completion_tokens/1000000*10 + num_prompt_tokens/1000000*2.5) + "$")