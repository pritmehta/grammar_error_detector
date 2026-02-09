"""
It uses a trained T5-based model. 
It loads the saved model and tokenizer, 
accepts a user-input sentence, 
and generates a grammatically corrected version.

"""
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("../saved_model")
model = T5ForConditionalGeneration.from_pretrained("../saved_model")
model.to(device)
model.eval()

sentence = input("Enter sentence with grammatical errors: ")

inputs = tokenizer(
    sentence,
    return_tensors="pt",
    truncation=True,
    max_length=64
).to(device)

# Generate with scores
outputs = model.generate(
    **inputs,
    max_length=64,
    num_beams=4,
    return_dict_in_generate=True,
    output_scores=True
)

generated_ids = outputs.sequences
decoded_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ----- Confidence calculation -----
scores = outputs.scores  # list of logits for each generated token
token_ids = generated_ids[0][1:]  # skip <pad>/<s> token

token_probs = []

for score, token_id in zip(scores, token_ids):
    probs = F.softmax(score[0], dim=-1)
    token_prob = probs[token_id].item()
    token_probs.append(token_prob)

# Average probability
confidence = sum(token_probs) / len(token_probs)
confidence_percent = confidence * 100

print("\nCorrected sentence:")
print(decoded_sentence)

print(f"\nConfidence score: {confidence_percent:.2f}%")


