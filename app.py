import streamlit as st
import os
from rag_chatbot import load_docs_from_path, create_vector_store, chat_groq

# Use relative path to the PDF file now in your repo
PDF_PATH = os.path.join("Dell_data.pdf")

def main():
    st.set_page_config(page_title='RAG Chatbot')
    st.title("RAG Chatbot")

    if "vectorstore" not in st.session_state:
        with st.spinner("Loading PDF and creating vector store..."):
            docs = load_docs_from_path(PDF_PATH)
            st.session_state.vectorstore = create_vector_store(docs)
        st.success("Chatbot is ready!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.text_area("Ask a question about the PDF:", key="user_input")

    def submit_query():
        query = st.session_state.user_input
        if query:
            retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
            context = retriever.invoke(query)
            prompt = f"""
            Use the context below to answer the user's question. If the answer isn't in the context, just say "I am sorry, I don't know".

            Context: {context}
            Previous Chat: {st.session_state.chat_history}
            User Question: {query}
            """
            messages = [
   
                {'role': 'system', 'content': 'You are a helpful assistant. Avoid being funny or sarcastic in your response and always look for an answer from the given pdf file that you are trained on'},
                {'role': 'user', 'content': prompt}
            ]


            response = chat_groq(messages)

            st.session_state.chat_history.extend([
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ])
            st.session_state.user_input = ""

    st.button("Submit", on_click=submit_query)

    if st.session_state.chat_history:
        with st.expander("Chat History", expanded=True):
            for chat in st.session_state.chat_history[::-1]:
                st.markdown(f"**{chat['role'].capitalize()}:** {chat['content']}")

if __name__ == "__main__":
    main()
