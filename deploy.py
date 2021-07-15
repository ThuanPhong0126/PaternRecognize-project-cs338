from pathlib import Path

import numpy
import pandas
import streamlit
import torch
from torch.nn import functional
from transformers import BertForSequenceClassification, BertTokenizerFast
from sentence_transformers import SentenceTransformer, util

select = streamlit.sidebar.selectbox("Choose the model", ('SBERT', 'BERT large'))

#@streamlit.cache(allow_output_mutation=True)
def load_model_tokenizer():
    model = BertForSequenceClassification.from_pretrained('/content/drive/MyDrive/[]Nhan Dang/Question-Similarity/bert_large')
    tokenizer = BertTokenizerFast.from_pretrained('/content/drive/MyDrive/[]Nhan Dang/Question-Similarity/bert_large',
        do_lower_case=True)
    return model, tokenizer


def predict(model, encoded_dict_questions):
    model.eval()
    with torch.no_grad():
        logits = model(encoded_dict_questions["input_ids"],
                       token_type_ids=encoded_dict_questions["token_type_ids"],
                       attention_mask=encoded_dict_questions["attention_mask"])
    label = numpy.argmax(logits[0].numpy(), axis=1).flatten()
    return label, pandas.DataFrame(functional.softmax(logits[0], dim=1).detach().numpy(), columns=["False", "True"])

def calculate_similarity(a, b):
    return util.pytorch_cos_sim(a,b)[0][0]

streamlit.title("Detecting Pairs of Similar Questions")
streamlit.markdown("Here is the implementation site with Streamlit, Pytorch, Transformers and Sentence Transformers.")

streamlit.markdown("## How to use the App?")
streamlit.markdown("Very simple. Fill out ``First question`` and ``Second question`` and click "
                   " the button ``Check if duplicates``.")


bert_model, bert_tokenizer_fast = load_model_tokenizer()
SBERT_model = SentenceTransformer('distilbert-base-nli-stsb-quora-ranking')
print("Load done!")

question_1 = streamlit.text_input("First question:", max_chars=512)
question_2 = streamlit.text_input("Second question:", max_chars=512)

if streamlit.button("Check if duplicates"):
    if not question_1 and not question_2:
        streamlit.text("empty questions")
    else:
        if select=='SBERT':
          model_apply_state = streamlit.text("Predicting ...")
          question1_encode = SBERT_model.encode([question_1])[0]
          question2_encode = SBERT_model.encode([question_2])[0]
          similarity = calculate_similarity(question1_encode, question2_encode)
          model_apply_state.text(f"Is duplicate: {True if similarity>=0.85 else False}")
        else:
          encoded_dict = bert_tokenizer_fast.encode_plus(question_1, question_2,
                                                       max_length=310,
                                                       pad_to_max_length=True, return_attention_mask=True,
                                                       return_tensors="pt", truncation=True)

          model_apply_state = streamlit.text("Predicting ...")
          y_pred, predict_proba = predict(bert_model, encoded_dict)
          model_apply_state.text(f"Is duplicate: {True if y_pred == 1 else False}")

