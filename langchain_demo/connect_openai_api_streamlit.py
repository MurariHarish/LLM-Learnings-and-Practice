import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key
st.title('Langchain with OpenAI API demo')
input_text = st.text_input('Ask me Anything')

first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about celebrity {name}"
)
llm = OpenAI(temperature = 0.8)
chain = LLMChain(llm=llm, prompt = first_input_prompt, verbose=True, output_key='person')

second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "When was {person} born"
)

chain2 = LLMChain(llm=llm, prompt = second_input_prompt, verbose=True, output_key='dob')

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention major events happened on {dob}"
)

chain3 = LLMChain(llm=llm, prompt = third_input_prompt, verbose=True, output_key='events')

parent_chain = SequentialChain(chains=[chain,chain2,chain3], input_variables = ['name'], output_variables = ['person','dob','events'], verbose=True)

#Combining Multiple Prompt Templates
if input_text:
    st.write(parent_chain({'name' : input_text}))






