import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Hugging Face model details
MODEL_NAME = "google/flan-t5-base"  # Changed to a smaller, seq2seq model for demonstration

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def generate_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.title("NoCap AI Chat Interface")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's good?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Load the model (uses st.cache_resource to only load once)
    tokenizer, model = load_model()
    
    # Generate response
    response = generate_response(prompt, model, tokenizer)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.title("About")
st.sidebar.info("NoCap AI chatbot. It uses a model hosted on Hugging Face to generate responses.")
