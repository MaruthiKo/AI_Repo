
# AI_REPO

This repository contains code for chatting with a GitHub Repo that leverages an Open Source large language model (LLM) to provide informative responses to user queries.

## Demo


https://github.com/MaruthiKo/AI_Repo/assets/63769209/babadbb9-9095-4e58-9de2-ba43c64cdf21


## Prerequisites

Before running the code, please ensure you have the following installed:

- Python 3.7 or later (3.10 was used in the project)
  
<strong> API tokens for: </strong>
- Replicate (LLM provider)
- Activeloop
- GitHub

## Setup

Clone this repository:

```bash
git clone https://github.com/<your-username>/AI_Repo.git
```

Navigate to the repository directory:

```bash
cd <repo-name>
```

Create a .env file:

In the root of the repository, create a file named .env.<br>Add the following lines to the .env file, replacing the placeholders with your actual API tokens:
```
REPLICATE_API_TOKEN=<your_replicate_token>
ACTIVELOOP_TOKEN=<your_activeloop_token>
GITHUB_TOKEN=<your_github_token>
DATASET_PATH=<path_to_store_vector_store>
```

## Running the Code

Install required libraries:

```bash
pip install -r requirements.txt
```

Launch the Gradio interface:

```bash
python assignment.py
```

## Interact with the system
- A Gradio interface will open in your web browser.
- Type your question in the chat box and press Enter.
- The system will process your query and provide a response using the LLM.
## Additional Notes

The code is currently configured to use the segment-anything GitHub repository as the data source. If you want to use a different repository, modify the `github_url` variable in the main function.

You can adjust the `LLM model`, `vector store` settings, and other parameters in the code as needed.

