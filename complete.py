from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import gradio as gr

# Initialize LLM
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="gemma3:270m",
    temperature=0.7,
)

# Prompts
math_prompt = PromptTemplate.from_template(
    "You are a math expert. Solve this clearly: {input}"
)

history_prompt = PromptTemplate.from_template(
    "You are a history expert. Answer clearly: {input}"
)

coding_prompt = PromptTemplate.from_template(
    "You are a programming expert. Explain clearly: {input}"
)

# Chains (new style)
math_chain = math_prompt | llm
history_chain = history_prompt | llm
coding_chain = coding_prompt | llm

# Routing function
def route_question(question):
    q = question.lower()

    if any(word in q for word in ["sum", "multiply", "multiplied", "divide", "equation"]):
        return math_chain.invoke({"input": question}).content

    elif any(word in q for word in ["who", "when", "history", "war"]):
        return history_chain.invoke({"input": question}).content

    else:
        return coding_chain.invoke({"input": question}).content


# Gradio UI function
def chatbot(user_input):
    response = route_question(user_input)
    return response


# Create interface
interface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask something..."),
    outputs="text",
    title="Multi-Expert AI (Math | History | Coding)",
    description="Ask any question. The AI will route it to the right expert."
)

# Launch
interface.launch()