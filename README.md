
# RAG Chatbot

This project is a **retrieval-augmented generation (RAG)** chatbot that leverages language models to generate responses based on relevant information from documents. It uses **Ollama** for hosting the language models and provides a simple interface using **Streamlit**.

## Setup Instructions

Follow the steps below to set up the project on your local machine.

### 1. Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/mrpvi/rag-chatbot.git
cd rag-chatbot
```

### 2. Create and Activate a Virtual Environment

Next, create and activate a virtual environment to manage the project’s dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # For Windows, use: .venv\Scripts\activate
```

### 3. Install Required Dependencies

Install the necessary Python dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Pull the Ollama Model

To pull the required Ollama model, run the following command:

```bash
ollama pull deepseek-r1:1.5b
```

#### Optional: Use a Larger Model

The project defaults to the **DeepSeek-R1 1.5B** parameter model. If you prefer to use a larger model for potentially better performance, follow these steps:

1. Pull a different model by running:

    ```bash
    ollama pull <model_name>
    ```

    Example for a larger model:

    ```bash
    ollama pull deepseek-r1:14b
    ```

2. Update the `MODEL_NAME` constant in `app.py` to reflect your chosen model if you want. For example:

    ```python
    MODEL_NAME = "deepseek-r1:14b"  # Example of a larger model
    ```

### 5. Run the Streamlit Application

Finally, start the Streamlit application using **Uvicorn**:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The application will be available at `http://localhost:8000`.

---

## Notes

- If you are using a Windows machine, be sure to use `.env\Scripts\activate` for activating the virtual environment.
- Ensure that you have **Ollama** installed to pull the models.
- If you choose a larger model, be aware that it might require more system resources (memory, processing power).

---

This setup should allow you to run and test the chatbot with the model of your choice. Feel free to customize the application according to your needs!


## Usage

Once the application is running, clients can interact with the chatbot by sending a POST request to the `/ask_question` endpoint. You can use the following curl command to ask a question:

```bash
curl -X 'POST' \
  http://server_ip:8000/ask_question \
  -H 'Content-Type: application/json' \
  -d '{"question": "who is Ali?"}'
```

Replace `server_ip` with the actual IP address or domain of the server where the chatbot is hosted. The response will contain the answer to the question.
