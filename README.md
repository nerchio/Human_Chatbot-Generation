# Human-Chatbot Dialogue Generation

## Generation of Dialogues
Please contact Ruizhe Zhu (zhurui@student.ethz.ch) for questions of this part.

To learn more about the generation of dialogues, you can check the following directory:
```bash
cd Generation
```

The generation pipeline is built by `LangGraph`.

### How to run
First, you should install the dependencies by:
```bash
pip install -r requirements.txt
```
or 
```bash
conda env create -f environment.yml
```

Then, you need to set your own API keys in the `api.yaml` file.

After that, you should be able to run the generation experiments. You can refer to `experiment_example.sh` to get more details about the parameters.

Please note if you need more steps to run the fine-tuned models, please see part [Models](#Models).

### Datasets
In the `data` folder we provide the datasets we used.

`arena_model_a_summaries.jsonl` is the arena dataset and `oasst1_en_min_6_turns_summary` is the oasst dataset.

For fine-tuning, we use `sft_reformat.py` to reformat them into finetuning datasets `arena_sft.jsonl` and `oasst1_en_sft.jsonl`.

The two fine-tuning dataset are also public on Hugging Face: [oasst_sft](https://huggingface.co/datasets/SyangZhou/oasst_SFT) and [arena_sft](https://huggingface.co/datasets/SyangZhou/arena_SFT).

### Models
By checking `models.py` and `settings.yaml`, you can find the models and call chains we used in the project.

We have some fine-tuned models in them, and they are all public on Hugging Face. They are:
- [llama-3b-v1](https://huggingface.co/SyangZhou/autotrain-l3b-0520-v1)
- [llama-3b-v2](https://huggingface.co/SyangZhou/autotrain-l3b-0520-v2)
- [llama-8b-v1](https://huggingface.co/SyangZhou/autotrain-l8b-0520-v1)
- [llama-8b-v2](https://huggingface.co/SyangZhou/autotrain-l8b-0520-v2)
- [mistral-v1](https://huggingface.co/SyangZhou/autotrain-m-0520-v1)
- [mistral-v2](https://huggingface.co/SyangZhou/autotrain-m-0520-v2)

You have to start a new inference point on Hugging Face before using them for generation.

## Evaluation of of Generated Dialogues
Please contact Hao Zhu (zhurui@student.ethz.ch) for questions of this part.

### Evaluation Metrics
All the metrics are defined in `evaluation_metrics.py`. For more details, please check the project report.

Prompts of the judge LLM are defined in `prompts.py`.  

### How to run
The code to run the UniEval, PairEval, and GTEval process are defined in `uni_eval.py`, `pair_eval.py`, and `gt_eval.py`.  

Before running, you need to change three variables in the files: 

`dialogue_data_folder`: the directory where the dialogue data files to be evaluated reside.

`saved_result_folder`: the directory where you want to store the Evaluation results. 

`dialogue_data_file_names`: the dialogue data files you want to evaluate.

`index_pair_list`: For the PairEval and GTEval, you need to sepcify the conversations to be compared.  

Then, just execute: 

`python3 *_eval.py`

### Evaluation results
They are stored in `result_oasst` and `result_arena`.  

And we visualize the results. 

[oasst UniEval (6-turns)](Evaluation/result_summary_oasst/GPT4o_Evaluator/oasst_uni_eval_6_turns.pdf)

[oasst UniEval (12-turns)](Evaluation/result_summary_oasst/GPT4o_Evaluator/oasst_uni_eval_12_turns.pdf)

[arena UniEval (6-turns)](Evaluation/result_summary_arena/GPT4o_Evaluator/arena_uni_eval_6_turns.pdf)

[arena UniEval (12-turns)](Evaluation/result_summary_arena/GPT4o_Evaluator/arena_uni_eval_12_turns.pdf)

[oasst GTEval (6-turns)](Evaluation/result_summary_oasst/GPT4o_Evaluator/oasst_gt_eval_6_turns.pdf)

[oasst GTEval (12-turns)](Evaluation/result_summary_oasst/GPT4o_Evaluator/oasst_gt_eval_12_turns.pdf)

[arena GTEval (6-turns)](Evaluation/result_summary_arena/GPT4o_Evaluator/arena_gt_eval_6_turns.pdf)

[arena GTEval (12-turns)](Evaluation/result_summary_arena/GPT4o_Evaluator/arena_gt_eval_12_turns.pdf)