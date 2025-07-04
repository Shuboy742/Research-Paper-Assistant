from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

#model
llm = HuggingFaceEndpoint(
model="meta-llama/Llama-3.1-8B-Instruct",
task="text-generation",
)

model = ChatHuggingFace(llm=llm)
#style
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f7ff;
    }
    div.stButton > button:first-child {
        background-color: #0052cc;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        height: 3em;
        width: 15em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("Reasearch Paper Assistant")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis","A Comprehensive Survey on Graph Neural Networks","mageNet Classification with Deep Convolutional Neural Networks (AlexNet)","A Neural Algorithm of Artistic Style","Deep Residual Learning for Image Recognition","Deep Learning","Deep Learning with PyTorch","Generative Adversarial Nets (GANs)","Playing Atari with Deep Reinforcement Learning (DQN)","YOLO9000: Better, Faster, Stronger","ResNet: Deep Residual Learning for Image Recognition","AlphaGo: Mastering the game of Go with deep neural networks and tree search","Word2Vec: Efficient Estimation of Word Representations in Vector Space","DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

#placeholders
if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)