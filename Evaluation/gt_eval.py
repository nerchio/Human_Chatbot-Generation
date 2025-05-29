from evaluation_metrics import gt_eval
import re
import json

ChatGPT4o_api_key = "YOUR_KEY"

dialogue_data_folder = "./data_arena/"
saved_result_folder = "./"

dialogue_data_file_names = ["arena_llama_3b_v1_GPT4oMini_6.jsonl", "arena_llama_3b_v2_GPT4oMini_6.jsonl", "arena_llama_8b_v1_GPT4oMini_6.jsonl", "arena_llama_8b_v2_GPT4oMini_6.jsonl","arena_mistral_v1_GPT4oMini_6.jsonl", "arena_mistral_v2_GPT4oMini_6.jsonl", "arena_model_a_summaries.jsonl"]

index_pair_list = [(6,0)]

# Function to read and parse a JSONL file
def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))  # Parse each line as a JSON object
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
    return data


def clean_quotes(s):
    keep_positions = set()

    # Match "choice" or "reason" key and the start of their value
    for match in re.finditer(r'"(choice|reason)"\s*:\s*"', s):
        key = match.group(1)
        key_quoted_start = match.start(0)
        key_quoted_end = match.start(0) + len(f'"{key}"')

        # Keep quotes around the key
        keep_positions.add(key_quoted_start)                    # opening quote of key
        keep_positions.add(key_quoted_end - 1)                  # closing quote of key

        value_start_quote_pos = match.end() - 1
        keep_positions.add(value_start_quote_pos)               # opening quote of value

        val_start = match.end()

        if key == "choice":
            # Look for the next comma or newline block ending
            comma_pos = s.find(",", val_start)
            if comma_pos == -1:
                comma_pos = s.find("}", val_start)
            quote_pos = s.rfind('"', val_start, comma_pos)
        else:  # "reason"
            quote_pos = s.rfind('"', val_start)

        if quote_pos != -1:
            keep_positions.add(quote_pos) 
        
    result = ''.join(c for i, c in enumerate(s) if c != '"' or i in keep_positions)

    return result



if __name__ == "__main__":
    for index_pair in index_pair_list:
        
        dialogue_A_file_name = dialogue_data_file_names[index_pair[0]]
        dialogue_B_file_name = dialogue_data_file_names[index_pair[1]]

        dialogue_A_data_path = dialogue_data_folder + dialogue_A_file_name
        dialogue_B_data_path = dialogue_data_folder + dialogue_B_file_name

        dialogue_A_data = parse_jsonl(dialogue_A_data_path)
        dialogue_B_data = parse_jsonl(dialogue_B_data_path)

        saved_result_file_path = saved_result_folder  + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name
        saved_result_file = open(saved_result_file_path, "w", encoding="utf-8")

        for dialoue_index in range(len(dialogue_A_data)):

            print("dialogue A: " + dialogue_A_file_name)
            print("dialogue B: " + dialogue_B_file_name)
            print("dialoue_index: " + str(dialoue_index))

            conversation_A = dialogue_A_data[dialoue_index]["conversation"]
            conversation_B = dialogue_B_data[dialoue_index]["conversation"]


            # ************GPT 4o Evaluator gt_eval**********************
            pair_eval_response = gt_eval(conversation_A, conversation_B, ChatGPT4o_api_key, "gpt-4o")

            json_obj = json.loads(pair_eval_response)
            saved_result_file.write(json.dumps(json_obj) + "\n")