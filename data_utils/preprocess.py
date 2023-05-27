import json


def process(args, path = 'data/real/'):
    if args.data_name == 'roc':
        data_path = path + 'roc/roc2017real.txt'
        with open(data_path, 'r') as f:
            data = f.readlines()
        data = [json.loads(line) for line in data]

    return data

#if __name__ == '__main__':
#    data = process('roc')