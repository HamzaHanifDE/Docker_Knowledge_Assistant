import streamlit as st
from query_data import embedding, create_prompt
from llm_helper import llm

def main():
    st.title("Docker Knowledge Assistant")
    st.markdown("""
                  Welcome to the **Docker Knowledge Assistant**!  
                Ask any question about Docker.
                """
                )

    query_text = st.text_input("Enter your query:", placeholder="Type your question here...")

    if st.button("Get Response"):
        if not query_text.strip():
            st.error("Please enter a valid query.")
            return
        

        with st.spinner("Searching for context..."):
            results = embedding(query_text)
        generated_prompt, error = create_prompt(results, query_text)

        if error:
            st.warning(error)
        elif generated_prompt:
            with st.spinner("Generating response..."):
                response = llm.invoke(input=generated_prompt)
                response_text = response.content
                st.markdown("### Response:")
                st.write(response_text)
            
            sources = [doc.metadata.get("source", "Unknown source") for doc, _score in results]
        else:
            st.error("Failed to generate a prompt.")

if __name__ == "__main__":
    main()
