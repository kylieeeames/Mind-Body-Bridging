from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, AutoConfig
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_average_sentiment(sentences):
    sentiments = []
    for sentence in sentences:

        input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]
            prob = F.softmax(logits, dim=-1).numpy()

        sentiments.append(prob)

    average_sentiment = np.mean(sentiments, axis=0)
    return average_sentiment

def predict_sentiment(text):
    sentiment = sentiment_pipeline(text)[0]
    return sentiment['label'], sentiment['score']

# Streamlit
st.title('Sentiment Analysis')

# Text boxes
before_text = st.text_area("Before", "Enter your text before here")
after_text = st.text_area("After", "Enter your text after here")

def split_sentences(text):
    sentences = []
    current_sentence = []
    punctuation = {".", "?", "!"}

    for character in text:
        current_sentence.append(character)
        if character in punctuation:
            sentences.append("".join(current_sentence).strip())
            current_sentence = []

    if current_sentence:
        sentences.append("".join(current_sentence).strip())

    return sentences

# Split paragraphs into sentences
before_sentences = split_sentences(before_text)
after_sentences = split_sentences(after_text)

# Calculate average sentiment for before and after text
if st.button('Compare'):
    average_sentiment_before = get_average_sentiment(before_sentences)
    average_sentiment_after = get_average_sentiment(after_sentences)

    # Get average emotion labels
    emotion_before, score_before = predict_sentiment(before_text)
    emotion_after, score_after = predict_sentiment(after_text)

    # Results
    st.write(f'Average Emotion Before: {emotion_before}')
    st.write(f'Average Emotion After: {emotion_after}')

    st.subheader(f'Results')
    st.write(f'Negative')
    st.write(f' - Before: {average_sentiment_before[0]:.4f}')
    st.write(f' - After: {average_sentiment_after[0]:.4f}')
    st.write(f'Neutral')
    st.write(f' - Before: {average_sentiment_before[1]:.4f}')
    st.write(f' - After: {average_sentiment_after[1]:.4f}')
    st.write(f'Positive')
    st.write(f' - Before: {average_sentiment_before[2]:.4f}')
    st.write(f' - After: {average_sentiment_after[2]:.4f}')
