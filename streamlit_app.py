import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import torch
import openai
from openai import OpenAI
import groq

# Hugging Face model details
T5_MODEL_NAME = "google/flan-t5-base"
DIALOGPT_MODEL_NAME = "microsoft/DialoGPT-medium"

@st.cache_resource
def load_t5_model():
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
    return tokenizer, model

@st.cache_resource
def load_dialogpt_model():
    tokenizer = AutoTokenizer.from_pretrained(DIALOGPT_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(DIALOGPT_MODEL_NAME)
    return tokenizer, model

def generate_t5_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_dialogpt_response(prompt, model, tokenizer):
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

def generate_openai_response(prompt, client):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are NoCap AI. You are a helpful assistant, however your responses should follow a urban slang format where profanity is acceptable and brutal honesty is expected. You have the best interest of the user at heart. Don't be afraid to deliver bad news or tough love when prompted."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_groq_response(prompt, client):
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are NoCap AI. You are a helpful assistant, however your responses should follow a urban slang format where profanity is acceptable and brutal honesty is expected. You have the best interest of the user at heart. Don't be afraid to deliver bad news or tough love when prompted."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
    )
    return chat_completion.choices[0].message.content


# Setting page layout
st.set_page_config(
    page_title="NoCap AI",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar for API Key and User Info
st.sidebar.header("About App")
st.sidebar.markdown('This is an app is an honest chatbot you can use various language models to produce responses from created by <a href="https://ai.jdavis.xyz" target="_blank">0xjdavis</a>.', unsafe_allow_html=True)

# Calendly
st.sidebar.markdown("""
    <hr />
    <center>
    <div style="border-radius:8px;padding:8px;background:#fff";width:100%;">
    <img src="https://avatars.githubusercontent.com/u/98430977" alt="Oxjdavis" height="100" width="100" border="0" style="border-radius:50%"/>
    <br />
    <span style="height:12px;width:12px;background-color:#77e0b5;border-radius:50%;display:inline-block;"></span> <b>I'm available for new projects!</b><br />
    <a href="https://calendly.com/0xjdavis" target="_blank"><button style="background:#126ff3;color:#fff;border: 1px #126ff3 solid;border-radius:8px;padding:8px 16px;margin:10px 0">Schedule a call</button></a><br />
    </div>
    </center>
    <br />
""", unsafe_allow_html=True)

# Copyright
st.sidebar.caption("©️ Copyright 2024 J. Davis")

st.title("NoCap AI")
st.write("An honest chatbot")

# Model selection
model_option = st.sidebar.selectbox(
    "Choose a model",
    ("Hugging Face - T5", "Hugging Face - DialoGPT", "OpenAI GPT-3.5", "Groq llama3-8b-8192")
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
        if model_option == "Hugging Face - T5":
            # Load the model (uses st.cache_resource to only load once)
            tokenizer, model = load_t5_model()
            # Generate response
            response = generate_t5_response(prompt, model, tokenizer)
        elif model_option == "Hugging Face - DialoGPT":
            # Load the model (uses st.cache_resource to only load once)
            tokenizer, model = load_dialogpt_model()
            # Generate response
            response = generate_dialogpt_response(prompt, model, tokenizer)
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
