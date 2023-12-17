"""
File name: oracle_bot.py
Author: Luigi Saetta
Date created: 2023-12-17
Date last modified: 2023-12-17
Python Version: 3.9

Description:
    This module provides the chatbot UI for the RAG demo 

Usage:
    run with: streamlit run oracle_bot.py

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to buil a RAG solution,
    where all the data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

import streamlit as st

from prepare_chain import create_query_engine

#
# Configs
#


def reset_conversation():
    st.session_state.messages = []


#
# Main
#
st.title("OCI Bot powered by Generative AI")

# Added reset button
st.button("Clear Chat History", on_click=reset_conversation)

# Initialize chat history
if "messages" not in st.session_state:
    reset_conversation()

# init RAG
with st.spinner("Initializing RAG chain..."):
    query_engine = create_query_engine()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("Hello, how can I help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # here we call OCI genai...

    try:
        print("Calling RAG chain..")

        with st.spinner("Waiting for answer from AI services..."):
            response = query_engine.query(question)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error("An error occurred: " + str(e))
