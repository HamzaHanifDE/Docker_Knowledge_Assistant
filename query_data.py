import argparse
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from llm_helper import llm


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

No preambles please. 
"""

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text

def embedding(query_text):
    CHROMA_PATH = './chroma'
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text)
    return results

def create_prompt(results, query_text):
    """Format the retrieved data and user query into a prompt."""
    if not results or len(results) == 0:
        return None, "No context found."
    
    try:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt, None
    except Exception as e:
        return None, f"Error creating prompt: {str(e)}"

def main():
    query_text = arg_parser()    
    results = embedding(query_text)
    formatted_prompt = create_prompt(results, query_text)

    if formatted_prompt:
        response = llm.invoke(input=formatted_prompt)
        response_text = response.content
        
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\nSources: {sources}"
        print(formatted_response)
    else:
        print("No valid prompt generated.")


if __name__ == "__main__":
    main()
