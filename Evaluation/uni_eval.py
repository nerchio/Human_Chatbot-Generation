from evaluation_metrics import uni_eval

import json

ChatGPT4o_api_key = "YOUR_KEY"

dialogue_data_folder = "./data_arena/"
saved_file_folder = "./"

dialogue_data_file_names = ["arena_llama_3b_v1_GPT4oMini_6.jsonl"]


# Function to read and parse a JSONL file
def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        i = 1
        for line in file:
            try:
                data.append(json.loads(line))  # Parse each line as a JSON object
            except json.JSONDecodeError as e:
                print("line %d" % i)
                print(f"Error parsing line: {e}")
            i=i+1
    return data


if __name__ == "__main__":
    for dialogue_data_file_name in dialogue_data_file_names:

        dialogue_data_path = dialogue_data_folder  + dialogue_data_file_name

        saved_result_file_path = saved_file_folder + dialogue_data_file_name
        saved_result_file = open(saved_result_file_path, "w", encoding="utf-8")

        dialogue_data = parse_jsonl(dialogue_data_path)


        dialogue_index = 0
        for entry in dialogue_data:

            dialogue_index = dialogue_index + 1

            print("dialugoe data: " + dialogue_data_file_name)
            print("dialogue: " + str(dialogue_index))

            conversation = entry["conversation"]


            # **************GPT4o Evaluator uni_eval*****************
            uni_eval_response = uni_eval(conversation, ChatGPT4o_api_key, "gpt-4o")
            json_obj = json.loads(uni_eval_response)
            saved_result_file.write(json.dumps(json_obj) + "\n")





