import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import openai
import groq

# Hugging Face model details
HF_MODEL_NAME = "google/flan-t5-base"

@st.cache_resource
def load_hf_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
    return tokenizer, model

def generate_hf_response(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def generate_groq_response(prompt):
    client = groq.Groq()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

st.title("NoCap AI Chat Interface")

# Model selection
model_option = st.sidebar.selectbox(
    "Choose a model",
    ("Hugging Face", "OpenAI GPT-3.5", "Groq llama3-8b-8192")
)

# API key input for OpenAI and Groq
if model_option == "OpenAI GPT-3.5":
    openai.api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
elif model_option == "Groq llama3-8b-8192":
    groq.api_key = st.sidebar.text_input("Enter your Groq API key", type="password")

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
    
    # Generate response based on selected model
    if model_option == "Hugging Face":
        tokenizer, model = load_hf_model()
        response = generate_hf_response(prompt, model, tokenizer)
    elif model_option == "OpenAI GPT-3.5":
        if not openai.api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            st.stop()
        response = generate_openai_response(prompt)
    elif model_option == "Groq llama3-8b-8192":
        if not groq.api_key:
            st.error("Please enter your Groq API key in the sidebar.")
            st.stop()
        response = generate_groq_response(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.title("About")
st.sidebar.info("NoCap AI chatbot. It uses various models to generate responses.")
