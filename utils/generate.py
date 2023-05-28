import torch
import time
import json
import numpy as np
import argparse
import os
import nltk

from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.process import *

DATASETS = ['roc', 'arxiv']
MODELS = ['llama-7b', 'llama-13b', 'alpaca-7b', 'vicuna-7b', 'vicuna-13b']


def set_seed(seed):
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)


def generate(**kwargs):
    current = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path_to_model = kwargs['path_to_model'] if kwargs['path_to_model'] else os.path.join('weights', kwargs['model'])

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = AutoModelForCausalLM.from_pretrained(path_to_model)
    model = model.to(device)

    # Load data
    fdata = open(os.path.join(kwargs['data_dir'], kwargs['data_name'], 'real.jsonl'), 'r')

    match kwargs['data_name']:
        case 'roc':
            processor = RocProcessor(kwargs['model'], 'continue')
        case 'arxiv':
            processor = ArxivProcessor(kwargs['model'], 'continue')

    for idx, line in enumerate(fdata):
        if kwargs['time_limit'] is not None:
            if time.time() - current >= kwargs['time_limit'] * 60:
                break

        if kwargs['generation_count'] == 0:
            break

        sample = json.loads(line)
        if idx >= kwargs['start']:
            print('Generating:', idx)
            text = sample['text']

            text_length = len(tokenizer.encode(text))
         
            inputs = processor.get_inputs(tokenizer, text).to(device)


            generate_ids = model.generate(inputs.input_ids,
                                          min_length=text_length - kwargs['window'],
                                          max_length=text_length + kwargs['window'],
                                          do_sample=kwargs['do_sample'],
                                          num_beams=kwargs['num_beams'],
                                          temperature=kwargs['temperature'],
                                          top_k=kwargs['top_k'],
                                          top_p=kwargs['top_p']
                                          )

            result = tokenizer.batch_decode(generate_ids,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]

            json_result = {'id': idx, 'text': result}

            output_file = os.path.join(kwargs['data_dir'], 'test', kwargs['data_name'], 'fake.jsonl')
            with open(output_file, 'a') as f:
                f.write(json.dumps(json_result) + "\n")
            kwargs['generation_count'] -= 1


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama-7b", choices=MODELS)
    parser.add_argument("--path-to-model", type=str, default=None)
    parser.add_argument("--data-name", type=str, default='roc', choices=DATASETS)
    parser.add_argument("--data-dir", type=str, default='data')
    parser.add_argument("--generation-count", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=None, help='in miniutes')
    parser.add_argument("--start", type=int, default=0, help='starting index of the data')
    parser.add_argument("--output-dir", type=str, default='data/fakes')
    parser.add_argument("--window", type=int, default=5, help='flucuation window of the orignal length')
    parser.add_argument("--do-sample", type=bool, default=True)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
  
    args = parser.parse_args()

    set_seed(args.seed)

    return vars(args)


if __name__ == '__main__':
    generate(**get_args())
