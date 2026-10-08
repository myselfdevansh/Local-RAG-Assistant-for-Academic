import langchain
import os 
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loader\\Your paragraph text.pdf")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = init_chat_model(
    "llama-3.1-8b-instant", 
    model_provider="groq"
)

docs = data.load()

template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes text. "),
    ("human", "{data}")
])

prompt = template.format_messages(data = docs[0].page_content)


result = model.invoke(prompt)

print(result.content)