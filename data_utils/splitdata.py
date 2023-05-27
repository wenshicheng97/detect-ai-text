from transformers import GPT2Tokenizer
import jsonlines
import os
from tqdm import tqdm

truncate_length = 50
split_batch = 100
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
origin = jsonlines.open("cnn-original.jsonl", mode='r')
origin_split = jsonlines.open('data/original_split_%i/cnn_stories_0.jsonl'%split_batch, mode='a')
truncate_split = jsonlines.open('data/truncate_split_%i/cnn_stories_0.jsonl'%split_batch, mode='a')
for idx, line in enumerate(tqdm(origin)):
    if idx % split_batch == 0:
        origin_split.close()
        truncate_split.close()
        origin_split = jsonlines.open('data/original_split_%i/cnn_stories_%i.jsonl' % (split_batch, idx // split_batch), mode='a')
        truncate_split = jsonlines.open('data/truncate_split_%i/cnn_stories_%i.jsonl' % (split_batch, idx // split_batch), mode='a')
    text = ''
    if type(line) == dict:
        text = line['text']
    elif type(line) == str:
        text = line
    origin_split.write(text)
    tokens = tokenizer.encode(text, truncation=True, max_length=truncate_length)
    truncate_text = tokenizer.decode(tokens)
    truncate_split.write(truncate_text)