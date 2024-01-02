from llama_index.llms import Replicate
from dotenv import load_dotenv
import os

load_dotenv()

replicate_token = os.getenv("REPLICATE_API_TOKEN")

llm = Replicate(
    model="mistralai/mistral-7b-instruct-v0.1:5fe0a3d7ac2852264a25279d1dfb798acbc4d49711d126646594e212cb821749"
)
# resp = llm.complete("Who is Paul Graham?")
# print(resp)

from llama_index.llms import ChatMessage

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content=""),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")



from llama_index import ServiceContext
from llama_index.llms import YourOpenSourceLLM  # replace with your LLM

# Create a ServiceContext with your LLM
service_context = ServiceContext.from_defaults(llm=YourOpenSourceLLM())

# Pass the service_context to the VectorStoreIndex
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context, storage_context=storage_context
)

# Continue with the rest of your code
retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
response_synthesizer = get_response_synthesizer()
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_mode='default',
    response_synthesizer=response_synthesizer,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)]
)


from llama_index.prompts.base import PromptTemplate

# Create a prompt template
qa_tmpl_str = """\
Context information is below. 
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, \
answer the query.
{emotion_str}
Query: {query_str}
Answer: \
"""
qa_tmpl = PromptTemplate(qa_tmpl_str)

# Add the prompt template to the query engine
query_engine.update_prompts({1234: qa_tmpl})
