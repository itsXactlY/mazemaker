#!/bin/sh
pip install -q aiohttp aiolimiter openai python-dotenv PyYAML rich 2>&1 | tail -1
cd /emb
printf 'LLM_API_KEY=ollama\nLLM_BASE_URL=http://host.containers.internal:11434/v1\n' > .env
CLI="python -m eval.cli --dataset dataset/004/dialogue.json --qa dataset/004/qa_004.json --system mazemaker --top-k 10"
echo '########## FLOOR (minicpm5 judge, reused answers) ##########'
$CLI --user-id floor2 --stages evaluate 2>&1 | grep -E 'Accuracy|MC:|OE:|Total:' || true
echo '########## DREAM (fixed supersedes) ##########'
python3 /bench/dream_evermembench.py 2>&1 | grep -viE 'embed-server|embed\]|Embedding backend' | grep -E 'cycle|edges before|DONE|prefix|supersedes' || true
echo '########## COMPLETE ##########'
$CLI --user-id complete2 --stages search answer evaluate 2>&1 | grep -E 'Accuracy|MC:|OE:|Total:|Search Stage|Answer Stage' || true
echo '########## DONE ##########'
