import json
import matplotlib.pyplot as plt


dialogue_data_file_names = ["oasst1_en_DeepSeek_GPT4oMini_6.jsonl", "oasst1_en_gemma_27b_GPT4oMini_6.jsonl", "oasst1_en_GPT4o_GPT4oMini_6.jsonl", "oasst1_en_GPT4oMini_GPT4oMini_6.jsonl", "oasst1_en_llama_3B_GPT4oMini_6.jsonl", "oasst1_en_llama_3b_v1_GPT4oMini_6.jsonl", "oasst1_en_llama_3b_v2_GPT4oMini_6.jsonl", "oasst1_en_llama_8B_GPT4oMini_6.jsonl", "oasst1_en_llama_8b_v1_GPT4oMini_6.jsonl", "oasst1_en_llama_8b_v2_GPT4oMini_6.jsonl", "oasst1_en_llama_70B_GPT4oMini_6.jsonl", "oasst1_en_mistral_7B_GPT4oMini_6.jsonl",  "oasst1_en_mistral_v1_GPT4oMini_6.jsonl",  "oasst1_en_mistral_v2_GPT4oMini_6.jsonl", "oasst1_en_min_6_turns_summary.jsonl"]
dialogue_data_file_names = ["oasst1_en_DeepSeek_GPT4oMini_12.jsonl", "oasst1_en_gemma_27b_GPT4oMini_12.jsonl", "oasst1_en_GPT4o_GPT4oMini_12.jsonl", "oasst1_en_GPT4oMini_GPT4oMini_12.jsonl", "oasst1_en_llama_3B_GPT4oMini_12.jsonl", "oasst1_en_llama_3b_v1_GPT4oMini_12.jsonl", "oasst1_en_llama_3b_v2_GPT4oMini_12.jsonl", "oasst1_en_llama_8B_GPT4oMini_12.jsonl", "oasst1_en_llama_8b_v1_GPT4oMini_12.jsonl", "oasst1_en_llama_8b_v2_GPT4oMini_12.jsonl", "oasst1_en_llama_70B_GPT4oMini_12.jsonl", "oasst1_en_mistral_7B_GPT4oMini_12.jsonl",  "oasst1_en_mistral_v1_GPT4oMini_12.jsonl",  "oasst1_en_mistral_v2_GPT4oMini_12.jsonl", "oasst1_en_min_6_turns_summary.jsonl"]
dialogue_data_file_names = ["arena_DeepSeek_GPT4oMini_6.jsonl","arena_gemma_27b_GPT4oMini_6.jsonl", "arena_GPT4o_GPT4oMini_6.jsonl", "arena_GPT4oMini_GPT4oMini_6.jsonl", "arena_llama_3B_GPT4oMini_6.jsonl", "arena_llama_3b_v1_GPT4oMini_6.jsonl", "arena_llama_3b_v2_GPT4oMini_6.jsonl", "arena_llama_8B_GPT4oMini_6.jsonl", "arena_llama_8b_v1_GPT4oMini_6.jsonl","arena_llama_8b_v2_GPT4oMini_6.jsonl", "arena_llama_70B_GPT4oMini_6.jsonl", "arena_mistral_7B_GPT4oMini_6.jsonl","arena_mistral_v1_GPT4oMini_6.jsonl", "arena_mistral_v2_GPT4oMini_6.jsonl", "arena_model_a_summaries.jsonl"]
dialogue_data_file_names = ["arena_DeepSeek_GPT4oMini_12.jsonl","arena_gemma_27b_GPT4oMini_12.jsonl", "arena_GPT4o_GPT4oMini_12.jsonl", "arena_GPT4oMini_GPT4oMini_12.jsonl", "arena_llama_3B_GPT4oMini_12.jsonl", "arena_llama_3b_v1_GPT4oMini_12.jsonl", "arena_llama_3b_v2_GPT4oMini_12.jsonl", "arena_llama_8B_GPT4oMini_12.jsonl", "arena_llama_8b_v1_GPT4oMini_12.jsonl","arena_llama_8b_v2_GPT4oMini_12.jsonl", "arena_llama_70B_GPT4oMini_12.jsonl", "arena_mistral_7B_GPT4oMini_12.jsonl","arena_mistral_v1_GPT4oMini_12.jsonl", "arena_mistral_v2_GPT4oMini_12.jsonl", "arena_model_a_summaries.jsonl"]
model_name_list = ["DS","gemma_\n27b",  "GPT4o", "GPT4o\nMini", "llama_\n3B", "llama_\n3B_\nv1", "llama_\n3B_\nv2",  "llama_\n8B", "llama_\n8B_\nv1", "llama_\n8B_\nv2",  "llama_\n70B", "mistral_\n7B", "mistral_\n7B_\nv1", "mistral_\n7B_\nv2",  "real"]
bar_color_list = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Teal/Cyan
    "#aec7e8",  # Light Blue
    "#ffbb78",  # Light Orange
    "#98df8a",  # Light Green
    "#ff9896",  # Light Red
    "#c5b0d5",  # Light Purple
]




# *****************************uni-eval result summary********************************************************
input_data_folder ="/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_arena/GPT4o_Evaluator/uni_eval/"
output_result_folder = "/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_summary_arena/GPT4o_Evaluator/uni_eval/"

passing_rate_list = []
for dialogue_data_file_name in dialogue_data_file_names:

    input_file_path =input_data_folder + dialogue_data_file_name
    output_file_path = output_result_folder + dialogue_data_file_name

    num_total_dialogues = 0
    num_passed_dialogues = 0
    num_failed_dialogues = 0


    # Open and read the JSONL file line by line
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line as a JSON object
            data = json.loads(line)
            
            choice = data.get("choice")
            index = data.get("index")
            reason = data.get("reason")

            num_total_dialogues = num_total_dialogues + 1
            if choice == "Yes":
                num_failed_dialogues = num_failed_dialogues + 1
            elif choice == "No":
                num_passed_dialogues = num_passed_dialogues + 1

    
    # Write the list of dicts to a new JSONL file
    with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("Pass Rate: " + str(num_passed_dialogues/num_total_dialogues) + "\n")
            f.write("# Total Dialogues: " + str(num_total_dialogues) + "\n")
            f.write("# Passed Dialogues: " + str(num_passed_dialogues) + "\n")
            f.write("# Failed Dialogues: " + str(num_failed_dialogues) + "\n")

    passing_rate_list.append(num_passed_dialogues/num_total_dialogues)

# draw the uni-eval plot
plt.figure(figsize=(10, 6))

# Create bar plot with specified colors
bars = plt.bar(model_name_list, passing_rate_list,color=bar_color_list)
plt.xticks(fontsize=9)


# # Add legend
# for bar, category in zip(bars, model_name_list):
#     bar.set_label(category)
# plt.legend(title='Models')

# Add title and axis labels
plt.title('Passing Rate (No AI-Involved) of different models')
plt.xlabel('Models')
plt.ylabel('Passing Rate')

# Draw horizontal line at y = 20
plt.axhline(y=passing_rate_list[-1], color='red', linestyle='--', linewidth=1.5, label='real')

# Save the figure as a PDF
plt.savefig('/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_summary_arena/GPT4o_Evaluator/uni_eval_12_turns.png', format='png')
plt.close()  # Closes the current figure

# # Show the plot
# plt.show()








# *****************************pair result summary********************************************************
input_data_folder ="/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_arena/GPT4o_Evaluator/pair_eval/"
output_result_folder = "/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_summary_arena/GPT4o_Evaluator/pair_eval/"

index_pair_list = [(1,2), (5,6)]
for index_pair in index_pair_list:
    dialogue_A_file_name = dialogue_data_file_names[index_pair[0]]
    dialogue_B_file_name = dialogue_data_file_names[index_pair[1]]

    input_file_path = input_data_folder + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name
    output_file_path = output_result_folder + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name

    num_total_dialogues = 0
    num_A_win_dialogues = 0
    num_B_win_dialogues = 0
    num_both_passed_dialogues = 0
    num_both_failed_dialogues = 0


    # Open and read the JSONL file line by line
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line as a JSON object
            data = json.loads(line)
            
            choice = data.get("choice")
            reason = data.get("reason")

            num_total_dialogues = num_total_dialogues + 1
            if choice == "Conversation 2":
                num_A_win_dialogues = num_A_win_dialogues + 1
            elif choice == "Conversation 1":
                num_B_win_dialogues = num_B_win_dialogues + 1
            elif choice == "Both":
                num_both_failed_dialogues = num_both_failed_dialogues + 1
            elif choice == "Neither":
                num_both_passed_dialogues = num_both_passed_dialogues + 1

    # Write the list of dicts to a new JSONL file
    with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("Conversation A: " + dialogue_A_file_name + "\n")
            f.write("Conversation B: " + dialogue_B_file_name + "\n")

            f.write("A Better Rate: " + str(num_A_win_dialogues/num_total_dialogues) + "\n")
            f.write("B Better Rate: " + str(num_B_win_dialogues/num_total_dialogues) + "\n")
            f.write("Both Failed Rate: " + str(num_both_failed_dialogues/num_total_dialogues) + "\n")
            f.write("Both Passed Rate: " + str(num_both_passed_dialogues/num_total_dialogues) + "\n")                
            f.write("# Total Dialogues: " + str(num_total_dialogues) + "\n")









# *****************************gt result summary********************************************************
input_data_folder = "/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_arena/GPT4o_Evaluator/gt_eval/" 
output_result_folder = "/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_summary_arena/GPT4o_Evaluator/gt_eval/"

Indistinguishable_rate_list = []
index_pair_list = [(14,0), (14,1), (14,2), (14,3), (14,4), (14,5), (14,6), (14,7), (14,8), (14,9), (14,10), (14,11), (14,12), (14,13)]

for index_pair in index_pair_list:
    dialogue_A_file_name = dialogue_data_file_names[index_pair[0]]
    dialogue_B_file_name = dialogue_data_file_names[index_pair[1]]



    input_file_path = input_data_folder + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name
    output_file_path = output_result_folder + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name

    num_total_dialogues = 0
    num_A_win_dialogues = 0
    num_B_win_dialogues = 0
    num_both_passed_dialogues = 0
    num_both_failed_dialogues = 0

    # Open and read the JSONL file line by line
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line as a JSON object
            data = json.loads(line)
            
            choice = data.get("choice")
            reason = data.get("reason")

            num_total_dialogues = num_total_dialogues + 1
            if choice == "Conversation 2":
                num_A_win_dialogues = num_A_win_dialogues + 1
            elif choice == "Conversation 1":
                num_B_win_dialogues = num_B_win_dialogues + 1
            elif choice == "Both":
                num_both_failed_dialogues = num_both_failed_dialogues + 1
            elif choice == "Neither":
                num_both_passed_dialogues = num_both_passed_dialogues + 1

    # Write the list of dicts to a new JSONL file
    with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("Conversation A: " + dialogue_A_file_name + "\n")
            f.write("Conversation B: " + dialogue_B_file_name + "\n")

            f.write("A Better Rate: " + str(num_A_win_dialogues/num_total_dialogues) + "\n")
            f.write("B Better Rate: " + str(num_B_win_dialogues/num_total_dialogues) + "\n")
            f.write("Both Failed Rate: " + str(num_both_failed_dialogues/num_total_dialogues) + "\n")
            f.write("Both Passed Rate: " + str(num_both_passed_dialogues/num_total_dialogues) + "\n")                
            f.write("# Total Dialogues: " + str(num_total_dialogues) + "\n")

            f.write("# Indistinguishable Rate: " + str((num_total_dialogues-num_A_win_dialogues)/(num_total_dialogues)) + "\n")

    Indistinguishable_rate_list.append((num_total_dialogues-num_A_win_dialogues)/(num_total_dialogues))

plt.figure(figsize=(10, 6))

# draw the gt-eval plot
# Create bar plot with specified colors
bars = plt.bar(model_name_list[:-1], Indistinguishable_rate_list, color=bar_color_list[:-1])


# Add title and axis labels
plt.title('Indistinguishable Rate (compared with real-world data)')
plt.xlabel('Models')
plt.ylabel('Indistinguishable Rate')

# Save the figure as a PDF
plt.savefig('/home/haozhu2/Human_Chatbot-Generation/Evaluation3/result_summary_arena/GPT4o_Evaluator/gt_eval_12_turns.png', format='png')

plt.close()  # Closes the current figure
# # Show the plot
# plt.show()
