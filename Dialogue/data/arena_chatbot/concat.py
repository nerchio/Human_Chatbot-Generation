import json
import os

def merge_jsonl_files(output_file, input_files):
    # 确保输出目录存在
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 处理每个输入文件
        for input_file in input_files:
            print(f"正在处理文件: {input_file}")
            with open(input_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # 直接写入行，保持原始格式
                    outfile.write(line)
    
    print(f"\n合并完成！输出文件: {output_file}")

if __name__ == "__main__":
    # 定义输入文件列表
    input_files = [
        "arena_3_turns_model_a.jsonl",
        "arena_4_turns_model_a.jsonl",
        "arena_5_turns_model_a.jsonl"
    ]
    
    # 定义输出文件
    output_file = "arena_model_a.jsonl"
    
    # 执行合并
    merge_jsonl_files(output_file, input_files)
