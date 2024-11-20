from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os


load_dotenv()

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name = "llama-3.2-90b-text-preview")


if __name__ == "__main__":
    res = llm.invoke('''What are the ingredients to make the luchi paratha in pakistan''')
    print(res.content)