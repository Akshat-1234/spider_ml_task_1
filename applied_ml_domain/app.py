import streamlit as st #for making the website/chatbot
#we made the backend in rag.py
from rag import answer_question

#Title
st.title(
    "Research Paper RAG Chatbot"
)

#Input
question = st.text_input(
    "Ask a question about the papers"
)

#If Button Clicked and question asked
if st.button("Ask") and question:
    with st.spinner("Searching papers and generating answer..."):
        answer, sources, docs = answer_question(question)

    #Answer
    st.subheader("Answer")
    st.write(answer)

    #Sources cited
    st.subheader("Sources")
    for source in sources:
        st.markdown(f"- {source}")
    
    #Number of Sources used
    st.subheader(
    f"Retrieved Papers ({len(sources)})")
    
    with st.expander("Context"):
        for doc in docs:
            st.markdown(
                f"**{doc.metadata['source_paper']}**"
)
            st.code(doc.page_content[:500], language=None)


