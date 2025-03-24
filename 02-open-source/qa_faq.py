import streamlit as st
import time  # Replace with actual RAG processing time

from openai import OpenAI
from elasticsearch import Elasticsearch

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

es_client = Elasticsearch('http://localhost:9200')

def elastic_search(query, index_name = "course-questions"):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs

def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: {context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
        print(context)
    
    prompt = prompt_template.format(question=query, context=context).strip()
    print("=== CONTEXT ===")  # <-- Ajoutez ceci
    print(context)
    return prompt

def llm(prompt):
    response = client.chat.completions.create(
        model='llama3.2:1b',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# App layout
st.title("RAG Application")

# Initialize session state for storing response
if 'response' not in st.session_state:
    st.session_state.response = None

# Input box
user_input = st.text_input("Enter your question:", key="input")

# Ask button
if st.button("Ask"):
    if user_input.strip():  # Check if input is not empty
        with st.spinner("Processing..."):
            # Call RAG function and store response in session state
            st.session_state.response = rag(user_input)
    else:
        st.warning("Please enter a question")

# Display response if available
if st.session_state.response is not None:
    st.subheader("Response:")
    st.write(st.session_state.response)