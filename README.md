# 📄 PDF Assistant 🤖

Welcome to the PDF Assistant! This tool allows you to interact with the content of your PDF documents through a chat interface powered by language models. Here's how you can make the most of it:


https://github.com/EvelynLopesSS/PDf_Assistant/assets/113462824/074f7a26-b1ac-45e9-8438-a417a342aa04



## 👉 Resources Used

![image](https://github.com/EvelynLopesSS/PDF_Assistant_Ollama/assets/113462824/36b7307b-4898-430b-b624-d817bf42f3cf)


### 📚 Langchain 🦜🖇️

Langchain is a library that offers a variety of functionalities for natural language processing (NLP), including language modeling, search, and information retrieval. In the PDF Assistant, Langchain is used to create a question and answer (QA) model to interact with the content of PDF documents.

### 🧠 Ollama🦙

Ollama is an artificial intelligence platform that provides advanced language models for various NLP tasks. In the PDF Assistant, we use Ollama to integrate powerful language models, such as Mistral, which is used to understand and respond to user questions.

To use Ollama, follow the instructions below:
1. **Installation**: After installing Ollama, execute the following commands in the terminal to download and configure the Mistral model:
```cmd
ollama run mistral
```

2. **Execution**: To pull the Mistral model, use the following command:
```cmd
ollama pull mistral
```

You can find more information and download Ollama at [https://ollama.com](https://ollama.com).

### 🔍 Chroma

Chroma is a library for efficient storage and retrieval of document vectors. It is used in the PDF Assistant to index the content of PDF documents and facilitate the retrieval of relevant information during interactions with users.


## How to Use

After configuring Ollama, you can run the PDF Assistant as follows:

1. Clone this repository to your local environment.
2. In the terminal, navigate to the project directory.
3. Execute the command `streamlit run filename.py` to start the application.

Once the application is running, you can upload PDF documents and start interacting with the content through the chat interface.
### 🚀 Getting Started

1. **Upload Your PDF**: Drag and drop your PDF file into the designated area or use the upload button below.

### 💬 How to Interact
2. **Ask Questions**: Once your document has been processed, start asking questions in the chat input to interact with the PDF content.

The PDF Assistant uses advanced language processing and retrieval techniques to understand your queries and provide accurate responses based on the content of your PDF document.



