import json
import matplotlib.pyplot as plt



dialogue_data_file_names = ["oasst1_en_DeepSeek_GPT4oMini_6.jsonl", "oasst1_en_GPT4oMini_GPT4oMini_6.jsonl", "oasst1_en_GPT4o_GPT4oMini_6.jsonl", "oasst1_en_llama_3B_GPT4oMini_6.jsonl", "oasst1_en_llama_70B_GPT4oMini_6.jsonl", "oasst1_en_min_6_turns_summary.jsonl"]
passing_rate_list = []
model_name_list = ["DeepSeek", "GPT4oMini", "GPT4o", "llama_3B", "llama_70B", "real"]
bar_color_list = ["blue", "grey", "black", "green", "purple", "brown"]

# uni-eval result summary
for dialogue_data_file_name in dialogue_data_file_names:

    input_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result/uni_eval/" + dialogue_data_file_name
    output_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result_summary/uni_eval/" + dialogue_data_file_name

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
# Create bar plot with specified colors
bars = plt.bar(model_name_list, passing_rate_list,color=bar_color_list)


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
plt.savefig('/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result_summary/uni_eval.png', format='png')
plt.close()  # Closes the current figure

# # Show the plot
# plt.show()





# pair-eval result summary
index_pair_list = [(0,2), (1,2), (3,4), (4,2), (1,0), (1,4)]
for index_pair in index_pair_list:
    dialogue_A_file_name = dialogue_data_file_names[index_pair[0]]
    dialogue_B_file_name = dialogue_data_file_names[index_pair[1]]

    # dialogue_A_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/" + dialogue_A_file_name
    # dialogue_B_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/" + dialogue_B_file_name


    input_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result/pair_eval/" + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name
    output_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result_summary/pair_eval/" + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name


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



# gt-eval result summary
Indistinguishable_rate_list = []


index_pair_list = [(5,0), (5,1), (5,2), (5,3), (5,4)]
for index_pair in index_pair_list:
    dialogue_A_file_name = dialogue_data_file_names[index_pair[0]]
    dialogue_B_file_name = dialogue_data_file_names[index_pair[1]]

    # dialogue_A_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/" + dialogue_A_file_name
    # dialogue_B_data_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation/data/" + dialogue_B_file_name


    input_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result/gt_eval/" + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name
    output_file_path = "/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result_summary/gt_eval/" + dialogue_A_file_name[:-6] + "_" + dialogue_B_file_name


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


# draw the gt-eval plot
# Create bar plot with specified colors
bars = plt.bar(model_name_list[:-1], Indistinguishable_rate_list, color=bar_color_list[:-1])


# Add title and axis labels
plt.title('Indistinguishable Rate (compared with real-world data)')
plt.xlabel('Models')
plt.ylabel('Indistinguishable Rate')

# Save the figure as a PDF
plt.savefig('/home/haozhu2/Human_Chatbot-Generation/Evaluation2/result_summary/gt_eval.png', format='png')
plt.close()  # Closes the current figure
# # Show the plot
# plt.show()
