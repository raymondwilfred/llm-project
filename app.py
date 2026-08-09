import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="LangChain + Groq + Streamlit")
st.title("LangChain + Groq + Streamlit Demo")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, api_key=os.getenv("GROQ_API_KEY"))
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise, helpful assistant."),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state.history.append(("user", user_input))
    st.chat_message("user").write(user_input)
    response = st.write_stream(chain.stream({"question": user_input}))
    st.session_state.history.append(("assistant", response))