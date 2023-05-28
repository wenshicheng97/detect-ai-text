# Detect-AI-Text

## Data Generation

Rename data file as `real.jsonl` and place the file under path `./data/{DATA_NAME}/`, e.g. `./data/{roc}/real.jsonl`

Place the hugging face format model weights under path `./weights/{MODEL_NAME}/`, e.g. `./weights/llama-7b/`

Run the following script

```
python -m utils.generate
```

