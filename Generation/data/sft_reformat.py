from prompts import GET_PROMPT
import json
import os

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # 读取每一行JSON数据
            data = json.loads(line.strip())
            conversation = data['conversation']
            turns = data['turns']
            summary = data['task_summary']
            INQUIRER_SYSTEM_PROMPT, INQUIRER_PROMPT, RESPONDER_SYSTEM_PROMPT = GET_PROMPT(summary)

            for i in range(turns//2):
                dialogue_history = conversation[0:2*i]
                processed_data = {"messages":[{"content": INQUIRER_SYSTEM_PROMPT, "role": "system"},
                {"content": str(dialogue_history) + INQUIRER_PROMPT, "role": "user"},
                {"content": conversation[2*i]['content'], "role": "assistant"}]}

                # 将处理后的数据写入新文件
                f_out.write(json.dumps(processed_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_file = "data/oasst1_en_min_6_turns_summary.jsonl"
    output_file = "data/oasst1_en_sft.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    process_jsonl(input_file, output_file)

