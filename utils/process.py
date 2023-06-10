import json
import nltk
import transformers

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

TEMPLATES = {
    'alpaca': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\r\n\r\n### Instruction: {} {}\r\n\r\n### Response:',
    'vicuna': 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: {} {} ASSISTANT:',
    'llama': '{} {}'
}


class Processor(object):
    def __init__(self, model_name, generation_type, instruction):
        self.model_name = model_name
        self.generation_type = generation_type
        self.instruction = instruction
        self.template = TEMPLATES[model_name]

    def get_inputs(self, text):
        return self.template.format(self.instruction, text)


class RocProcessor(Processor):
    def __init__(self, model_name, generation_type):
        instruction = ''
        super(RocProcessor, self).__init__(model_name, generation_type, instruction)

    @staticmethod
    def get_text(sample):
        return sample['text']

    @staticmethod
    def process_text(text):
        return nltk.tokenize.sent_tokenize(text, language='english')[0]


class ArxivProcessor(Processor):
    def __init__(self, model_name, generation_type):
        instruction = 'Rephrase the following text as an abstract of paper:'
        super(ArxivProcessor, self).__init__(model_name, generation_type, instruction)

    @staticmethod
    def get_text(sample):
        return sample['abstract']

    @staticmethod
    def process_text(text):
        return text.replace('\n', ' ')


class WebtextProcessor(Processor):
    def __init__(self, model_name, generation_type):
        if generation_type == 'rewrite':
            instruction = 'Rephrase the following text:'
        else:
            instruction = ''
        super(WebtextProcessor, self).__init__(model_name, generation_type, instruction)

    @staticmethod
    def get_text(sample):
        return sample['text']

    def process_text(self, text):
        if self.generation_type == 'rewrite':
            return text
        else:
            return text
