from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


def summarize(
    text, n_words=None, compression=None,
    max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0, 
    **kwargs
):
    """
    Summarize the text
    The following parameters are mutually exclusive:
    - n_words (int) is an approximate number of words to generate.
    - compression (float) is an approximate length ratio of summary and original text.
    """
    MODEL_NAME = 'cointegrated/rut5-base-absum'

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    if n_words:
        text = '[{}] '.format(n_words) + text
    elif compression:
        text = '[{0:.1g}] '.format(compression) + text

    x = tokenizer(text, return_tensors='pt', padding=True)
    with torch.inference_mode():
        out = model.generate(
            **x, 
            max_length=max_length, num_beams=num_beams, 
            do_sample=do_sample, repetition_penalty=repetition_penalty, 
            **kwargs
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def get_cluster_summaries(top_texts_list):
    
    summ_list = {}
    for cluster_id, top in enumerate(top_texts_list):
        summ_list[cluster_id] = summarize(' '.join(list(top)), n_words=50, compression=0.75)
    
    return summ_list

