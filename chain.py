from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOllama(
    url="http://localhost:11434",
    model="gemma3:270m",
    temperature=0.7,
    api_key="ollama"
)

prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple English in 3 lines."
)

chain1 = LLMChain(llm=llm, prompt=prompt1)

result = chain1.invoke({"topic": "LangChain chains"})
print(result["text"])