!pip install openai==1.27
!pip install pinecone-client==4.0.0
!pip install langchain==0.1.19
!pip install langchain-openai==0.1.6
!pip install langchain-pinecone==0.1.0
!pip install tiktoken==0.7.0
!pip install typing_extensions==4.11.0


import pandas as pd
movies_raw = pd.read_csv("IMDB.csv")

movies_raw.head()

movies = movies_raw.rename(columns = {
    "primaryTitle": "movie_title",
    "Description": "movie_description",
})

movies["source"] = "title/" + movies["tconst"]

movies = movies.loc[movies["titleType"] == "movie"]

movies = movies[["movie_title", "movie_description", "source", "genres"]]

movies.head()

from langchain.document_loaders import DataFrameLoader

movies["page_content"] = "Title: " + movies["movie_title"] + "\n" + \
                         "Genre: " + movies["genres"] + "\n" + \
                         "Description: " +movies["movie_description"]

movies = movies[["page_content", "source"]]

docs = DataFrameLoader(
    movies,
    page_content_column="page_content",
).load()

print(f"First 3 documents: {docs[:3]}")
print(f"Number of documents: {len(docs)}") 

import tiktoken
encoder = tiktoken.get_encoding("cl100k_base")

tokens_per_doc = [len(encoder.encode(doc.page_content)) for doc in docs]

total_tokens = sum(tokens_per_doc)
cost_per_1000_tokens = 0.0001
cost =  (total_tokens / 1000) * cost_per_1000_tokens
cost

import os
import pinecone

api_key=os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=api_key)

index_name = "imdb-movies"

existing_index_names = [idx.name for idx in pc.list_indexes().indexes]

if index_name not in existing_index_names:
    pc.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536,
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
    )

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings()

index = pc.Index(index_name)

n_vectors = index.describe_index_stats()['total_vector_count']
print(f"There are {n_vectors} vectors in the index already.")

if n_vectors > 0:
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
else:
    docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

question = "What's a good movie about viking?"
    
print("These are the documents most relevant to the question:")
docsearch.as_retriever().invoke(question)

from langchain.prompts import PromptTemplate

DOCUMENT_PROMPT = """{page_content}
IMDB link: {source}
========="""

QUESTION_PROMPT = """Given the following extracted parts of a movie database and a question, create a final answer with the IMDB link as source ("SOURCE").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCE" part in your answer.

QUESTION: What's a good movie about a robot to watch with my family?
=========
Title: A.I. Artificial Intelligence
Genre: Drama,Sci-Fi
Description: A robotic boy, the first programmed to love, David (Haley Joel Osment) is adopted as a test case by a Cybertronics employee (Sam Robards) and his wife (Frances O'Connor). Though he gradually becomes their child, a series of unexpected circumstances make this life impossible for David. Without final acceptance by humans or machines, David embarks on a journey to discover where he truly belongs, uncovering a world in which the line between robot and machine is both vast and profoundly thin.
IMDB link: 
=========
Title: I, Robot
Genre: Action,Mystery,Sci-Fi
Description: In 2035, highly intelligent robots fill public service positions throughout the world, operating under three rules to keep humans safe. Despite his dark history with robotics, Detective Del Spooner (Will Smith) investigates the alleged suicide of U.S. Robotics founder Alfred Lanning (James Cromwell) and believes that a human-like robot (Alan Tudyk) murdered him. With the help of a robot expert (Bridget Moynahan), Spooner discovers a conspiracy that may enslave the human race.
IMDB link: 
=========
Title: The Iron Giant
Genre: Action,Adventure,Animation
Description: In this animated adaptation of Ted Hughes' Cold War fable, a giant alien robot (Vin Diesel) crash-lands near the small town of Rockwell, Maine, in 1957. Exploring the area, a local 9-year-old boy, Hogarth, discovers the robot, and soon forms an unlikely friendship with him. When a paranoid government agent, Kent Mansley, becomes determined to destroy the robot, Hogarth and beatnik Dean McCoppin (Harry Connick Jr.) must do what they can to save the misunderstood machine.
IMDB link: 
=========
FINAL ANSWER: 'The Iron Giant' is an animated movie about a friendship between a robot and a kid. It would be a good movie to watch with a kid.
SOURCE: 

QUESTION: {question}
=========
{summaries}
FINAL ANSWER:"""

document_prompt = PromptTemplate.from_template(DOCUMENT_PROMPT)
question_prompt = PromptTemplate.from_template(QUESTION_PROMPT)


from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    chain_type="stuff",
    llm=llm,
    chain_type_kwargs={
        "document_prompt": document_prompt,
        "prompt": question_prompt,
    },
    retriever=docsearch.as_retriever(),
)

qa_with_sources.invoke(question)


import langchain
langchain.debug = True
qa_with_sources.invoke(question)

