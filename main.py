import os
import textwrap
from dotenv import load_dotenv
from llama_index import download_loader
from llama_hub.github_repo import GithubRepositoryReader, GithubClient
from llama_index import VectorStoreIndex
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index import get_response_synthesizer
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.query_engine import RetrieverQueryEngine
import deeplake
import re

# Load environment variables
load_dotenv()

replicate_token = os.getenv("REPLICATE_API_TOKEN")
active_loop_token = os.getenv("ACTIVELOOP_TOKEN")
dataset_path = os.getenv("DATASET_PATH")
github_token = os.getenv("GITHUB_TOKEN")

def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

def initialize_github_client():
    github_token = os.getenv("GITHUB_TOKEN")
    return GithubClient(github_token)

def main():
    if not replicate_token:
        raise EnvironmentError("Replicate token not found in environment variables")

    # Check for GitHub Token
    if not github_token:
        raise EnvironmentError("GitHub token not found in environment variables")
    
    # Check for Activeloop Token
    if not active_loop_token:
        raise EnvironmentError("Activeloop token not found in environment variables")

    github_client = initialize_github_client()
    download_loader("GithubRepositoryReader")

    github_url = input("Please enter the GitHub repository URL: ")
    # owner, repo = parse_github_url(github_url)

    while True:
        owner, repo = parse_github_url(github_url)
        if validate_owner_repo(owner, repo):
            loader = GithubRepositoryReader(
                github_client,
                owner = owner,
                repo = repo,
                filter_file_extensions=(
                    [".py", ".js", ".ts", ".md"],
                    GithubRepositoryReader.FilterType.INCLUDE,
                ),
                verbose=False,
                concurrent_requests=5,
            )
            print(f"Loading {repo} repository by {owner}")
            docs = loader.load_data(branch="main")
            print("Documents uploaded: ")
            for doc in docs:
                print(doc.metadata)
            break # Exit the loop once the valid URL is processed
        else:
            print("Invalid GitHub URL. Please try again.")
            github_url = input("Please enter the GitHub repository URL: ")
    
    print("Uploading to vector store... ")

    # Create vector store and upload data
    try:
        exists = deeplake.exists(dataset_path)
        if exists:
            vector_store = DeepLakeVectorStore(
                dataset_path=dataset_path,
                read_only=True,
                runtime={"tensor_db": True},
            )
        else:
            vector_store = DeepLakeVectorStore(
                dataset_path=dataset_path,
                overwrite=True,
                runtime={"tensor_db": True},
            )
    except Exception as e:
        print(f"An unexpected error occurred while creating or fetching the vector store: {str(e)}")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
    # query_engine = index.as_query_engine()
    response_synthesizer = get_response_synthesizer()
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_mode='default',
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)]
    )

    # intro_question = "What is the repository about?"
    # print(f"Test question: {intro_question}")
    # print("=" * 50)
    # answer = query_engine.query(intro_question)

    # print(f"Answer: {textwrap.fill(str(answer), 100)} \n")

    while True:
        user_question = input("Please enter your question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            print("Exiting, thanks for chatting! :) ")
            break

        print(f"Your question: {user_question}")
        print("=" * 50)

        answer = query_engine.query(user_question)
        print(f"Answer: {textwrap.fill(str(answer), 100)} \n")


if __name__ == "__main__":
    main()