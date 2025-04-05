import json


input_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/parsed_result/uni_eval_DeepSeek_prompt1.jsonl"
output_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/result_summary/uni_eval_DeepSeek_prompt1.txt"


num_total_dialogues = 0
num_passed_dialogues = 0
num_failed_dialogues = 0
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
        if choice == "Yes":
            num_failed_dialogues = num_failed_dialogues + 1
        elif choice == "No":
            num_passed_dialogues = num_passed_dialogues + 1

        num_completion_tokens = num_completion_tokens + int(completion_tokens)
        num_prompt_tokens = num_prompt_tokens + int(prompt_tokens)

# Write the list of dicts to a new JSONL file
with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("Pass Rate: " + str(num_passed_dialogues/num_total_dialogues) + "\n")
        f.write("# Total Dialogues: " + str(num_total_dialogues) + "\n")
        f.write("# Passed Dialogues: " + str(num_passed_dialogues) + "\n")
        f.write("# Failed Dialogues: " + str(num_failed_dialogues) + "\n")
        f.write("# completion prompts: " + str(num_completion_tokens) + "\n")
        f.write("# prompt prompts: " + str(num_prompt_tokens) + "\n")
        f.write("Cost: "+ str(num_completion_tokens/1000000*10 + num_prompt_tokens/1000000*2.5) + "$")

