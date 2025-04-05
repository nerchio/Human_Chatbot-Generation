import json

# File path
input_file_path =  "/home/haozhu2/Human_Chatbot-Generation/Evaluation/result/uni_eval_DeepSeek_prompt1.jsonl"
output_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/parsed_result/uni_eval_DeepSeek_prompt1.jsonl"



# List to hold the parsed dicts
parsed_data = []

# Open and read the JSONL file line by line
with open(input_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Parse each line as a JSON object
        data = json.loads(line)
        
        # Extract token fields
        token_info = data.get("response_metadata", {}).get("token_usage", {})
        
        # Start building the entry dict
        parsed_entry = {
            "content": data.get("content"),
            "completion_tokens": token_info.get("completion_tokens"),
            "prompt_tokens": token_info.get("prompt_tokens"),
            "total_tokens": token_info.get("total_tokens"),
        }

        # Parse the content field for Choice, Index, and Reason
        content = parsed_entry["content"]
        if content:
            lines = content.strip().split('\n')
            for line in lines:
                if line.startswith("Choice:"):
                    parsed_entry["Choice"] = line.replace("Choice:", "").strip()
                elif line.startswith("Index:"):
                    parsed_entry["Index"] = line.replace("Index:", "").strip()
                elif line.startswith("Reason:"):
                    reason_index = content.find("Reason:")
                    parsed_entry["Reason"] = content[reason_index + len("Reason:"):].strip()
                    break  # Reason is the last part, break after capturing it
        

        del parsed_entry["content"]

        # Add to list
        parsed_data.append(parsed_entry)


# Preview the first result
print(parsed_data[0])


# Write the list of dicts to a new JSONL file
with open(output_file_path, 'w', encoding='utf-8') as f:
    for entry in parsed_data:
        json_line = json.dumps(entry, ensure_ascii=False)
        f.write(json_line + '\n')
