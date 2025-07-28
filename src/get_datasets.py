from datasets import load_dataset

def load_aqua(split='test'):
    return load_dataset('aqua_rat', 'raw', split=split)

def load_trivia(split='test'):
    return load_dataset("mandarjoshi/trivia_qa", "rc", split=split)
