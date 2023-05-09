import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def predict_sentiment(text):
    encoded_text = tokenizer.encode_plus(
        text,
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids']
    attention_mask = encoded_text['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs[0]

    probs = torch.softmax(logits, dim=-1)
    prob_neg = probs[0][0].item()
    prob_pos = probs[0][1].item()

    if prob_pos > prob_neg:
        return "Positive"
    else:
        return "Negative"
    
test_sentences = ["테스트 문장입니다.", ""]
for sentence in test_sentences:
    predict_sentiment(sentence)