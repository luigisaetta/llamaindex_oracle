# Integrate Oracle Vector DB and OCI GenAI with llama-index
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![screenshot](./screenshot.png)

This repo contains all the work done on the development of RAG applications using:
* Oracle Vector DB
* Oracle OCI GenAI Service
* Oracle OCI Embeddings
* OCI ADS 2.9.1 (with support for OCI GenAI)
* llama-index

## What is RAG?
A very good introduction to what **Retrieval Augmented Generation** (RAG) is can be found [here](https://www.oracle.com/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/)

## Features
* basic (12/2023) integration between **Oracle DB Vector Store (23c)** and **llama-index**

## Demos
* [custom_vector_store_demo1](./custom_vector_store_demo1.ipynb) This NB shows a demo where you get answers to questions on Oracle Database and new features in 23c
* Bot powered by Vector DB and OCI GenAI

## Setup
See the wiki pages.

## Loading data
You can use the program [create_save_embeddings](./create_save_embeddings.py) to load all the data in the Oracle DB schema.

You can launch it using the script [load_data](./load_data.sh).

The list of files is psecified in config.py

You need to have pdf files in the same directory.

## Limited Availability
OCI GenAI and Oracle Vector DB are new features both in **Limited Availability**. 

Customers can easily enter in the LA programs.
To test these functionalities you need to enrol in the LA programs and install the proper versions of software libraries.

Code and functionalities can change, as a result of the feedbacks from customers.

## Release used for the demo (WIP)
* OCI ADS 2.9.1
* Oracle Database 23c free edition with Vector DB

## Libraries
* Streamlit
* Llama-index
