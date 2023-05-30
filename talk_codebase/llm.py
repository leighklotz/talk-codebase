import os

from langchain import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory

from talk_codebase.utils import StreamStdOut
from talk_codebase.vectorstore import create_vector_store


class Session:
    """Represents a conversation session.
    
    Attributes:
        vector_store (FAISS): A FAISS object that creates and manages the vector store.
        openai_api_key (str): A string that contains the OpenAI API key.
        model_name (str): A string that contains the name of the model.
        chat_history (list): A list that contains the chat history.
    """

    def __init__(self, root_dir:str, openai_api_key:str, model_name:str):
        self.root_dir = root_dir
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.chat_history = []
        self.create_pipeline()

    def create_pipeline(self):
        """
        Creates a pipeline for the conversational agent.
        Creates a vector store, a conversation buffer memory and a chat model.
        """
        self.vector_store = create_vector_store(self.root_dir, self.openai_api_key, self.model_name)
        self.memory = ConversationBufferMemory(input_key="question", output_key="answer", memory_key="chat_history", return_messages=True)
        self.model = ChatOpenAI(model_name=self.model_name, openai_api_key=self.openai_api_key, streaming=True, callback_manager=CallbackManager([StreamStdOut()]))
```
        self.vector_store = create_vector_store(self.root_dir, self.openai_api_key, self.model_name)
        self.memory = ConversationBufferMemory(input_key="question", output_key="answer", memory_key="chat_history", return_messages=True)
        self.model = ChatOpenAI(model_name=self.model_name, openai_api_key=self.openai_api_key, streaming=True, callback_manager=CallbackManager([StreamStdOut()]))
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        self.qa = ConversationalRetrievalChain.from_llm(self.model, retriever=self.retriever, return_source_documents=True, memory=self.memory)


        

    def print_sources(self, result):
        """
        Prints the metadata of the source documents used to answer the question.

        Args:
        - result: A dictionary object representing the result of a question-answering task.

        Returns:
        - None
        """
        print('\nSources:\n' + '\n -'.join(
            [f'📄 {s.metadata["source"]} in {os.path.abspath(s.metadata["source"])}:' for s in result["source_documents"]]))

    def send_question(self, question: str, show_sources: bool=True) -> dict:
        """
        Send a question to the conversational retrieval model and return the result.

        Args:
        - question (str): The question to be sent to the model.
        - show_sources (bool, optional): Whether to print the sources of the retrieved documents. Defaults to True.

        Returns:
        - result (dict): A dictionary containing the answer to the question and other metadata.

        """

        result = self.qa({"question": question, "chat_history": self.chat_history})

        if show_sources:
            self.print_sources(result)

        self.chat_history.append((question, result["answer"]))

        return result
