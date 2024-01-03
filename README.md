# Integrate Oracle Vector DB and OCI GenAI with llama-index
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![screenshot](./screenshot.png)

This repo contains all the work done on the development of RAG applications using:
* Oracle Vector DB
* Oracle OCI GenAI Service
* Oracle OCI Embeddings
* Cohere Reranking
* OCI [ADS 2.9.1](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/large_language_model/langchain_models.html) (with support for OCI GenAI)
* llama-index

## What is RAG?
A very good introduction to what **Retrieval Augmented Generation** (RAG) is can be found [here](https://www.oracle.com/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/)

## Features
* basic (12/2023) integration between **Oracle DB Vector Store (23c)** and **llama-index**
* All documents stored in an Oracle DB
* Oracle DB used for semantic search
* Reranking to improve retrieval
* How to show references (documents used for the response generation)
* (30/12) Added reranker implemented as OCI Model Deployment

## Demos
* [demo1](./custom_vector_store_demo1.ipynb) This NB shows a demo where you get answers to questions on Oracle Database and new features in 23c
* [Bot](./oracle_bot.py) powered by **Oracle Vector DB** and **OCI GenAI**
* [demo3](./custom_vector_store_demo3.ipynb) shows how to add a Reranker to the RAG chain; I have used **Cohere**
* [demo5] (./rag_chain_demo5.ipynb) shows a full rag chain where rerranker is deployed on OCI DS

## Setup
See the wiki pages.

## Loading data
* You can use [create_save_embeddings](./create_save_embeddings.py) to load all the data in the Oracle DB schema.
* You can launch it using the script [load_books](./load_books.sh).
* The list of files to be loaded is specified in config.py

You need to have pdf files in the same directory.

## Limited Availability
OCI GenAI and Oracle Vector DB are new features both in **Limited Availability**. 

Customers can easily enter in the LA programs.
To test these functionalities you need to enrol in the LA programs and install the proper versions of software libraries.

Code and functionalities can change, as a result of the feedbacks from customers.

## Releases used for the demo (WIP)
* OCI ADS 2.9.1
* Oracle Database 23c (23.4) free edition with Vector DB

## Libraries
* OCI ADS
* Streamlit
* Llama-index
