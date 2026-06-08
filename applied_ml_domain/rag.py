import os #Directory Management
from dotenv import load_dotenv 
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

#To safely load the APIKEY without leaking it
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

#Load embeddings(created in build_vectorstore)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#Load FAISS
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True #To prevent hacking
)
#We use Gemini FLASH as the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key
)

def answer_question(question):
    # example question = "Compare BERT and GPT-3"
    
    #using similarity search we find the closest vector to the question
    #vectorstone stores all the documents
    start = time.time()

    docs = vectorstore.similarity_search(
        question,
        k=20 # we use k=20 to find data in more depth by checking 20 nearest neighbours
            # this is useful for comparision queries or answers from multiple papers
    )
    print("Retrieval:", time.time() - start)
    #Adding the source papers metadata
    sources = sorted(
        list(
            set(
                doc.metadata["source_paper"]
                for doc in docs
            )
        )
    )

    #context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    #prompt to llm to avoid hallucination
    prompt = f"""
Answer using ONLY the provided context.

If the answer is not contained in the context,
say that the information is unavailable.
Do not hallucinate.
Organize answers clearly.
Use bullet points when listing multiple items.
Provide a concise explanation for each item.

Mention source papers when relevant.


    Context:
    {context}

    Question:
    {question}
    """
    #sending prompt to llm and printing
    start = time.time()

    try:
        response = llm.invoke(prompt)
    #Gemini daily limit being 10 made it quite difficult to continue with one API key
    except Exception:
        return (
            "Gemini API quota exceeded. Please try again later.",
            sources,
            docs
        )

    print("Gemini:", time.time() - start)
    return response.content, sources, docs

if __name__ == "__main__":

    answer, sources = answer_question(
        "What is self attention?"
    )

    print(answer)

    print("\nSources:")

    for source in sources:
        print(source)