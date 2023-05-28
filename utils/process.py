import json
import nltk
import transformers

nltk.download('punkt')


class Processor(object):
    def __init__(self, data_name, generation_type, template, instruction):
        self.data_name = data_name
        self.generation_type = generation_type
        self.template = template
        self.instruction = instruction

    @staticmethod
    def process_text(text):
        return text

    def get_inputs(self, tokenizer, text):
        input_text = self.template.format(self.instruction, text)
        print(input_text)
        return tokenizer(input_text, return_tensors='pt')


class AlpacaProcessor(Processor):
    def __init__(self, data_name, generation_type):
        template = "{}\r\n\r\n### {}\r\n\r\n### Response:"
        instruction = ""
        if data_name == 'arxiv':
            instruction = (
                "Below is an instruction that describes a task. "
                "Rewrite the following text as an abstract of paper."
            )

        super(AlpacaProcessor, self).__init__(data_name, generation_type, template, instruction)
        
    def process_text(self, text):
        if self.generation_type == 'continue':
            return nltk.tokenize.sent_tokenize(text, language='english')[0]
        return text


class LLaMaProcessor(Processor):
    def __init__(self, data_name, generation_type):
        super(LLaMaProcessor, self).__init__(data_name, generation_type, '', '')


class VicunaProcessor(Processor):
    def __init__(self, data_name, generation_type):
        super(VicunaProcessor, self).__init__(data_name, generation_type, '', '')