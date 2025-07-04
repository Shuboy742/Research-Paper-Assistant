# Research-Paper-Assistant

**A Streamlit web app that summarizes and explains research papers using Hugging Face language models via LangChain.
Enter a paper title, select your preferred style and length, and get clear, AI-generated summaries or beginner-friendly explanations instantly.**

Features

Summarize research papers in various styles and lengths

Beginner-friendly explanations for complex topics

Simple and interactive Streamlit interfaceSummarize research papers in various styles and lengths

Beginner-friendly explanations for complex topics

Developed it by using Hugging Face models (e.g., Llama, Zephyr)

Simple and interactive Streamlit interface

1.Create and activate a virtual environment (recommended)

 python3 -m venv venv
 source venv/bin/activate

2.Install dependencies
  pip install streamlit
  pip install streamlit langchain-huggingface python-dotenv

3.Set your Hugging Face API token

  Get your token from https://huggingface.co/settings/tokens

  Create a .env file in the project root:

  HUGGINGFACEHUB_API_TOKEN = your_huggingface_token_here

4.Add your prompt template
  Place your template.json file in the project directory.

5.Run the app
  streamlit run promptdrop.py
