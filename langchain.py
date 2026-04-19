from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import gradio as gr

# Initialize model
llm = ChatOllama(
    base_url="http://localhost:11434",   # ✅ use base_url instead of url
    model="gemma3:270m",
    temperature=0.7,
)

# Create prompt template ONCE
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor. Explain the topic in simple language."),
    ("human", "Explain {topic} in 5 bullet points.")
])

# Create chain correctly
chain = prompt | llm

# Function for Gradio
def result_prompt(user_input):
    result = chain.invoke({"topic": user_input})
    return result.content   # ✅ MUST return something

# Test in terminal
result = chain.invoke({"topic": "LangChain prompt templates"})
print(result.content)

# Gradio UI
iface = gr.Interface(
    fn=result_prompt,
    inputs="text",
    outputs="text",
    title="My First Chatbot"
)

iface.launch()