"""
File name: oracle_bot.py
Author: Luigi Saetta
Date created: 2023-12-17
Date last modified: 2023-12-29
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

import logging
import time
import streamlit as st

# to use the create_query_engine
import prepare_chain

import ads
from oci_utils import load_oci_config
from oci_translator import OCITranslator

#
# Configs
#
from config import ADD_REFERENCES, ADD_OCI_TRANSLATOR, GEN_MODEL, WORD_TO_TRIGGER_TRANS


def reset_conversation():
    st.session_state.messages = []


# defined here to avoid import of streamlit in other module
# cause we need here to use @cache
@st.cache_resource
def create_query_engine(verbose=False):
    query_engine, token_counter = prepare_chain.create_query_engine(verbose=verbose)

    # token_counter keeps track of the num. of tokens
    return query_engine, token_counter


@st.cache_resource
def create_translator():
    oci_config = load_oci_config()

    oci_trans = OCITranslator(oci_config=oci_config)

    return oci_trans


# to format output with references
def format_output(response):
    output = response.response

    if ADD_REFERENCES and len(response.source_nodes) > 0:
        output += "\n\n Ref.:\n\n"

        for node in response.source_nodes:
            output += str(node.metadata).replace("{", "").replace("}", "") + "  \n"

    return output


#
# Main
#

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

st.title("OCI Bot powered by Generative AI")

# Added reset button
st.button("Clear Chat History", on_click=reset_conversation)

# Initialize chat history
if "messages" not in st.session_state:
    reset_conversation()

# init RAG
with st.spinner("Initializing RAG chain..."):
    # I have added the token counter to count token
    # I've done this way because I marked the function with @cache
    # but there was a problem with the counter. It works if it is created in the other module
    # and returned here where I print the results for each query

    # here we create the query engine
    query_engine, token_counter = create_query_engine(verbose=False)

    # adding translation in Italian?
    if ADD_OCI_TRANSLATOR and GEN_MODEL == "OCI":
        logging.info("Adding OCI Translator...")

        oci_trans = create_translator()


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
        logging.info("Calling RAG chain..")

        with st.spinner("Waiting..."):
            tStart = time.time()

            # Here we call the entire chain !!!
            response = query_engine.query(question)

            # should we translate?
            if ADD_OCI_TRANSLATOR and GEN_MODEL == "OCI":
                # check if the question ask to translate in italian
                if WORD_TO_TRIGGER_TRANS.lower() in question.lower():
                    logging.info("Translating in it...")
                    # remember you have to pass a batch!
                    response.response = (
                        oci_trans.translate([response.response])
                        .documents[0]
                        .translated_text
                    )

            tEla = time.time() - tStart
            logging.info(f"Elapsed time: {round(tEla, 1)} sec.")

        # display num. of input/output token
        # count are incrementals
        str_token1 = f"LLM Prompt Tokens: {token_counter.prompt_llm_token_count}"
        str_token2 = (
            f"LLM Completion Tokens: {token_counter.completion_llm_token_count}"
        )

        logging.info(str_token1)
        logging.info(str_token2)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            if ADD_REFERENCES:
                # add reerences
                output = format_output(response)
            else:
                output = response

            st.markdown(output)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": output})

    except Exception as e:
        logging.error("An error occurred: " + str(e))
        st.error("An error occurred: " + str(e))
