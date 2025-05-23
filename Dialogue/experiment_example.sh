# dataset: oasst1_en, inquirer: GPT4oMini, responder: GPT4oMini, max_turns: 6
python ./main.py --data oasst1_en --inquirer_model GPT4oMini --responder_model GPT4oMini --max_turns 6 &> logs/GPT4oMini_GPT4oMini_oasst1_en_6turns.log &

# dataset: oasst1_en, inquirer: DeepSeek, responder: GPT4oMini, max_turns: 12
python ./main.py --data oasst1_en --inquirer_model DeepSeek --responder_model GPT4oMini --max_turns 12 &> logs/DeepSeek_GPT4oMini_oasst1_en_12turns.log &

# dataset: arena, inquirer: llama_8B, responder: GPT4oMini, max_turns: 6
python ./main.py --data arena --inquirer_model llama_8B --responder_model GPT4oMini --max_turns 6 &> logs/llama_8B_GPT4oMini_arena_6turns.log &

# dataset: arena, inquirer: llama_3B_v1 (our sft vision of llama_3B), responder: GPT4oMini, max_turns: 12
# an inference point on Hugging Face is needed to use llama_3B_v1
python ./main.py --data arena --inquirer_model llama_3B_v1 --responder_model GPT4oMini --max_turns 12 &> logs/llama_3B_v1_GPT4oMini_arena_12turns.log &
