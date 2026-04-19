from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

llm = ChatOllama(
    url="http://localhost:11434",
    model="gemma3:270m",
    temperature=0.0,
    api_key="ollama"
)

response_schemas = [
    ResponseSchema(name="gift", description="gift item name"),
    ResponseSchema(name="delivery_days", description="number of days for delivery"),
    ResponseSchema(name="price_value", description="price as a number")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that returns structured output only."),
    ("human", "Suggest a gift for {person}. {format_instructions}")
])

chain = prompt | llm

response = chain.invoke({
    "person": "a 10 year old child",
    "format_instructions": format_instructions
})

parsed = output_parser.parse(response.content)
print(parsed)