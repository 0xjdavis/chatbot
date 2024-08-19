import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import torch
import openai
from openai import OpenAI
import groq

# Hugging Face model details
HF_MODEL_FLAN_T5 = "google/flan-t5-base"
HF_MODEL_BERT = "bert-base-uncased"
HF_MODEL_DISTILBERT = "distilbert-base-uncased"

@st.cache_resource
def load_hf_model_seq2seq():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_FLAN_T5)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_FLAN_T5)
    return tokenizer, model

@st.cache_resource
def load_hf_model_causal(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_hf_response_seq2seq(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_hf_response_causal(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_openai_response(prompt, client):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_groq_response(prompt, client):
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
    )
    return chat_completion.choices[0].message.content

st.title("NoCap AI Chat Interface")

# Model selection
model_option = st.sidebar.selectbox(
    "Choose a model",
    ("Hugging Face - google/flan-t5-base", "Hugging Face - bert-base-uncased", "Hugging Face - distilbert-base-uncased", "OpenAI GPT-3.5", "Groq llama3-8b-8192")
)

# API key input for OpenAI and Groq
if model_option == "OpenAI GPT-3.5":
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
elif model_option == "Groq llama3-8b-8192":
    groq_api_key = st.sidebar.text_input("Enter your Groq API key", type="password")
    if groq_api_key:
        groq_client = groq.Groq(api_key=groq_api_key)

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
    
    try:
        if model_option == "Hugging Face - google/flan-t5-base":
            tokenizer, model = load_hf_model_seq2seq()
            response = generate_hf_response_seq2seq(prompt, model, tokenizer)
        elif model_option == "Hugging Face - bert-base-uncased":
            tokenizer, model = load_hf_model_causal(HF_MODEL_BERT)
            response = generate_hf_response_causal(prompt, model, tokenizer)
        elif model_option == "Hugging Face - distilbert-base-uncased":
            tokenizer, model = load_hf_model_causal(HF_MODEL_DISTILBERT)
            response = generate_hf_response_causal(prompt, model, tokenizer)
        elif model_option == "OpenAI GPT-3.5":
            if not openai_api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
                st.stop()
            response = generate_openai_response(prompt, openai_client)
        elif model_option == "Groq llama3-8b-8192":
            if not groq_api_key:
                st.error("Please enter your Groq API key in the sidebar.")
                st.stop()
            response = generate_groq_response(prompt, groq_client)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.sidebar.title("About")
st.sidebar.info("NoCap AI chatbot. It uses various models to generate responses.")
