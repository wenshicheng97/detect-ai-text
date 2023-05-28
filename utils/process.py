import json
import nltk
import transformers

nltk.download('punkt')


class Processor(object):
    def __init__(self, model_name, generation_type, instruction):
        self.model_name = model_name
        self.generation_type = generation_type
        self.instruction = instruction
        match model_name:
            case 'alpaca-7b':
                self.template = "{}\r\n\r\n### {}\r\n\r\n### Response:"
            case 'vicuna-7b':
                self.template = ''
            case _:
                self.template = ''

    @staticmethod
    def process_text(text):
        return text

    def get_inputs(self, tokenizer, text):
        input_text = self.template.format(self.instruction, text)
        return tokenizer(input_text, return_tensors='pt')


class RocProcessor(Processor):
    def __init__(self, model_name, generation_type):
        match model_name:
            case 'alpaca-7b':
                instruction = ''
            case 'vicuna-7b':
                instruction = ''
            case _:
                instruction = ''

        super(RocProcessor, self).__init__(model_name, generation_type, instruction)

    def process_text(self, text):
        return nltk.tokenize.sent_tokenize(text, language='english')[0]


class ArxivProcessor(Processor):
    def __init__(self, model_name, generation_type):
        match model_name:
            case 'alpaca-7b':
                instruction = 'Below is an instruction that describes a task. ' \
                              'Rewrite the following text as an abstract of paper.'
            case 'vicuna-7b':
                instruction = ''
            case _:
                instruction = ''

        super(ArxivProcessorr, self).__init__(model_name, generation_type, instruction)

    def process_text(self, text):
        return text.replace('\n', ' ')

