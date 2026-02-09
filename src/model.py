from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return model, tokenizer
