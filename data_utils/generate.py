from transformers import AutoTokenizer, LlamaForCausalLM
from preprocess import process
import torch
import time
import json
import numpy as np
import argparse

DATASETS = ['roc']

MODELS = ['llama-7B']

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def construct_generation_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="llama-7B", choices=MODELS)
    parser.add_argument("--data_name", type=str, default='roc', choices=DATASETS)
    parser.add_argument("--time_limit", type=int, default=5, help='in miniutes')
    parser.add_argument("--start", type=int, default=0, help='starting index of the data')
    parser.add_argument("--output_dir", type=str, default='data/fakes')
    parser.add_argument("--window", type=int, default=5, help='flucuation window of the orignal length')
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
  
    args = parser.parse_args()

    set_seed(args)

    return args


def load_model(args):
    if args.model == 'llama-7B':
        model_path = '/home/tangyimi/ai_detection/7B_converted'
        model = LlamaForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def get_inputs(args, text, tokenizer):
    if args.data_name == 'roc':
        prompt = text.split('.')[0] + '.' # take the first sentence
    return tokenizer(prompt, return_tensors="pt")


def get_outputs(args, model, tokenizer, inputs, length):
    generate_ids = model.generate(inputs.input_ids, min_length=length - args.window, max_length=length + args.window, 
                                  do_sample=args.do_sample, num_beams=args.num_beams, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return result


def generate(args):
    current = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model, tokenizer = load_model(args)
    model = model.to(device)

    # load data
    data = process(args)
    fake_results = []

    for idx, line in enumerate(data):
        if time.time() - current >= args.time_limit * 60:
            break

        if idx >= args.start:
            id = line['id']
            text = line['text']
            length = tokenizer(text, return_tensors="pt")['input_ids'].shape[-1]
         
            inputs = get_inputs(args, text, tokenizer).to(device)

            print('Generating:', id)
            
            result = get_outputs(args, model, tokenizer, inputs, length)

            json_result = {'id': id, 'text': result}
            fake_results.append(json_result)

    output_file = args.output_dir.rstrip('/') + '/'  + '{}Fake.jsonl'.format(args.data_name)
    with open(output_file, 'a') as f:
        for fake in fake_results:
            f.write(json.dumps(fake) + "\n")


def main():
    args = construct_generation_args()
    generate(args)

if __name__ == '__main__':
    main()
