from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

llm = ChatOllama(
    url="http://localhost:11434",
    model="gemma3:270m",
    temperature=0.7,
    api_key="ollama"
)

prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a short explanation of {topic}."
)

prompt2 = PromptTemplate(
    input_variables=["summary"],
    template="Convert this explanation into 3 interview questions: {summary}"
)

chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="summary")
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="questions")

sequential_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["topic"],
    output_variables=["summary", "questions"]
)

result = sequential_chain.invoke({"topic": "LangChain"})
print(result)