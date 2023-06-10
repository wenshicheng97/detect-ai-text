import torch
import time
import json
import numpy as np
import argparse
import os
import nltk

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from utils.process import *

DATASETS = ['roc', 'arxiv', 'webtext']
MODELS = ['llama', 'alpaca', 'vicuna']
SIZES = ['7b', '13b']


def set_seed(seed):
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)


def generate(**kwargs):
    current = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = kwargs['model'] + '-' + kwargs['model_size']

    path_to_model = kwargs['path_to_model'] if kwargs['path_to_model'] else os.path.join('weights', model_name)

    # Load model
    tokenizer = LlamaTokenizer.from_pretrained(path_to_model)
    model = LlamaForCausalLM.from_pretrained(path_to_model, torch_dtype=torch.float16)
    model = model.to(device)
    # os.system("nvidia-smi")
    print("Load model time:", time.time() - current)
    start_time = current
    current = time.time()
    taskid = kwargs['taskid']
    # Load data
    input_file = os.path.join(kwargs['data_dir'], kwargs['data_name'], f'real{taskid}.jsonl') if kwargs['input_file'] is None else kwargs['input_file']
    fdata = open(input_file, 'r') 

    match kwargs['data_name']:
        case 'roc':
            processor = RocProcessor(kwargs['model'], 'continue')
        case 'arxiv':
            processor = ArxivProcessor(kwargs['model'], 'rewrite')
        case 'webtext':
            processor = WebtextProcessor(kwargs['model'], 'rewrite')

    for idx, line in enumerate(fdata):
        if kwargs['time_limit'] is not None:
            if time.time() - start_time >= kwargs['time_limit'] * 60:
                break

        if kwargs['generation_count'] == 0:
            break

        sample = json.loads(line)
        if idx >= kwargs['start']:
            # print('Generating:', idx)
            os.system(f"echo Generating: {idx}")
            text = processor.process_text(processor.get_text(sample))
            text_length = len(tokenizer.encode(text))
            input_text = processor.get_inputs(processor.process_text(text))
            inputs = tokenizer(input_text, return_tensors='pt').to(device)
            generate_ids = model.generate(inputs.input_ids,
                                          min_new_tokens=text_length - kwargs['window'],
                                          max_new_tokens=text_length + kwargs['window'],
                                          do_sample=kwargs['do_sample'],
                                          num_beams=kwargs['num_beams'],
                                          temperature=kwargs['temperature'],
                                          top_k=kwargs['top_k'],
                                          top_p=kwargs['top_p']
                                          )

            result = tokenizer.batch_decode(generate_ids,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]

            json_result = {'id': idx, 'text': result[len(input_text):], 'model': model_name}
            output_dir = os.path.join(kwargs['data_dir'], 'test', kwargs['data_name'])
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            output_file = os.path.join(output_dir, f'fake{taskid}.jsonl') if kwargs['output_file'] is None else kwargs['output_file']
            with open(output_file, 'a') as f:
                f.write(json.dumps(json_result) + "\n")
            kwargs['generation_count'] -= 1
    print("Generation time:", time.time() - current)
    print("Total time:", time.time() - start_time)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama", choices=MODELS)
    parser.add_argument("--model-size", type=str, default='7b', choices=SIZES)
    parser.add_argument("--path-to-model", type=str, default=None)
    parser.add_argument("--data-name", type=str, default='roc', choices=DATASETS)
    parser.add_argument("--data-dir", type=str, default='data')
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--generation-count", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=None, help='in miniutes')
    parser.add_argument("--start", type=int, default=0, help='starting index of the data')
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--window", type=int, default=5, help='flucuation window of the orignal length')
    parser.add_argument("--do-sample", type=bool, default=True)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--taskid", type=str, default='')
  
    args = parser.parse_args()

    set_seed(args.seed)

    return vars(args)


if __name__ == '__main__':
    generate(**get_args())
