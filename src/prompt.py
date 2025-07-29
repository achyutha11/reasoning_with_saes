COT_PROMPT = r'Please reason step by step, and put your final answer within \boxed{}.'
MCQ_ANSWER_PROMPT = 'The correct answer is ('

def format_prompt_aqua(query, reasoning=True, include_options=True):
    question, options = query['question'], query['options']
    joined_options = "\n".join(options) if include_options else ""
    joined_options = ""
    if reasoning:
        return f'<s>[INST] {question}{joined_options}\n{COT_PROMPT} [/INST] \n<think>\n'
    else:
        return f'<s>[INST] {question}{joined_options}\n [/INST] \n'

def format_prompt_trivia(question, reasoning=True):
    if reasoning:
        return f'<s>[INST] {question}\n{COT_PROMPT} [/INST] \n<think>\n'
    else:
        return f'<s>[INST] {question}\n [/INST] \n'
