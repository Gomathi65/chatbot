from langchain_ollama import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize the model
llm = ChatOllama(
    url="http://localhost:11434",
    model="gemma3:270m",
    temperature=0.7,
    api_key="ollama"
)

# Initialize memory to store history
memory = ConversationBufferMemory()

# Create a conversation chain that uses the model and memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True # Set to True to see the prompt and history in the console
)

# Turn 1
print("--- Turn 1 ---")
response1 = conversation.predict(input="Hi, my name is Alice.")
print(f"AI: {response1}")

# Turn 2 - The model will remember the name even if not explicitly provided again
print("\n--- Turn 2 ---")
response2 = conversation.predict(input="What is my name?")
print(f"AI: {response2}")