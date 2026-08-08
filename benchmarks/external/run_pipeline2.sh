#!/bin/sh
pip install -q aiohttp aiolimiter openai python-dotenv PyYAML rich 2>&1 | tail -1
cd /emb
DS=dataset/004/dialogue.json; QA=dataset/004/qa_004.json
CLI="python -m eval.cli --dataset $DS --qa $QA --system mazemaker --top-k 10"
oll(){ printf 'LLM_API_KEY=ollama\nLLM_BASE_URL=http://host.containers.internal:11434/v1\n' > .env; }
nim(){ printf 'LLM_API_KEY=%s\nLLM_BASE_URL=https://integrate.api.nvidia.com/v1\n' "$NVKEY" > .env; }
echo '########## FLOOR (evaluate reused answers, glm judge) ##########'
nim; $CLI --user-id floor2 --stages evaluate 2>&1 | grep -E 'Accuracy|MC:|OE:|Total:' || true
echo '########## DREAM (fixed supersedes) ##########'
python3 /bench/dream_evermembench.py 2>&1 | grep -viE 'embed-server|embed\]|Embedding backend' | grep -E 'cycle|edges before|DONE|prefix' || true
echo '########## COMPLETE ##########'
oll; $CLI --user-id complete2 --stages search answer 2>&1 | grep -E 'Search Stage|Answer|completed' || true
nim; $CLI --user-id complete2 --stages evaluate 2>&1 | grep -E 'Accuracy|MC:|OE:|Total:' || true
echo '########## DONE ##########'
