import fitz  #pdf reader 
import os    #to manage directories

#Langchain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter #to make chunks
from langchain_community.vectorstores import FAISS #for search
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

#Paper Extraction from folder
all_documents = []


#Here we first use os module to find a given file in the papers folder,
#Then we extract text from each page and combine them to get full text
#Then we add the text of each pdf to all_documents along with file name
#We add file name because we are going to need to display which paper was the content fetched from while answering 
pdf_dir = "./applied_ml_domain/papers"


for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_dir,file)
        doc = fitz.open(path)
        full_text = ""
        for page in doc:
            full_text+=page.get_text()
        all_documents.append({
            "source_paper" : file,
            "content"      : full_text
        })


#Now the text is split into chunks if size 1000
#we use chunk overlap to divide text in a way that 2 adjacent segments contain common parts
#Here also we save the source paper name with each chunk to be able to later cite it
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []

for doc in all_documents:
    split_chunks=splitter.split_text(
        doc["content"]
    )
    
    for chunk in split_chunks:
        chunks.append({
            "text"         : chunk,
            "source_paper" : doc["source_paper"]
        })
        
# Now we need to convert the chunks to Langchain document because FAISS works perfectly with langchain documents
documents = []
for chunk in chunks:
    documents.append(
        Document(
            page_content=chunk["text"],
            metadata={
                "source_paper" : chunk["source_paper"]
            }
        )
    )

# Building the embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Building FAISS

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)
#creating vectorstore folder to save vectors
vectorstore.save_local(
    "applied_ml_domain/vectorstore"
)

#Verification of working of the code using a normal question
#here the answer is 5 nearest chunks of text 
results = vectorstore.similarity_search(
    "Explain the self-attention mechanism in transformers",
    k=5
)

#the chunks are sliced to 300 words for efficient verification
for doc in results:

    print("\nSOURCE:")
    print(doc.metadata["source_paper"])

    print(doc.page_content[:300])
    