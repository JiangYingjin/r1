python eval.py --model_name "unsloth/llama-3.2-3b-instruct-bnb-4bit"
python eval.py --model_name "unsloth/llama-3.2-3b-instruct-bnb-4bit" --exp_name "better_reward_llama" --step 300
python eval.py --model_name "unsloth/llama-3.2-3b-instruct-bnb-4bit" --exp_name "better_reward_llama" --step 200
python eval.py --model_name "unsloth/llama-3.2-3b-instruct-bnb-4bit" --exp_name "better_reward_llama" --step 100

python eval.py --model_name "unsloth/qwen2.5-1.5b-instruct-bnb-4bit" --exp_name "better_reward_qwen2.5_1.5b" --step 200
python eval.py --model_name "unsloth/qwen2.5-1.5b-instruct-bnb-4bit" --exp_name "better_reward_qwen2.5_1.5b" --step 100
python eval.py --model_name "unsloth/qwen2.5-1.5b-instruct-bnb-4bit" --exp_name "better_reward_qwen2.5_1.5b" --step 50

# python eval.py --model_name "unsloth/Phi-3.5-mini-instruct-bnb-4bit" --exp_name "better_reward_phi3.5" --step 300
# python eval.py --model_name "unsloth/Phi-3.5-mini-instruct-bnb-4bit" --exp_name "better_reward_phi3.5" --step 200