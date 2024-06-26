import streamlit as st
from dotenv import load_dotenv
import tempfile
from torch import cuda, bfloat16
#import transformers
from langchain.llms import HuggingFacePipeline
#from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
#from transformers import StoppingCriteria, StoppingCriteriaList
from frontPageTemplate import css, bot_template, user_template
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

import os

load_dotenv()

llmtemplate = """[INST]
As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
- Answer the question based on the provided documents.
- Be direct and factual, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
- Do not fabricate information or include questions in your responses.
- do not prompt to select answers. do not ask me questions
{question}
[/INST]
"""
#device_name:f'cuda:{cuda.default_stream()}'
#print(cuda)
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device) 

#model_dir = "/content/drive/MyDrive/Llama-2-7b-chat-hf"
#model_dir = "NousResearch/Llama-2-7b-chat-hf"
#tokenizer = AutoTokenizer.from_pretrained(model_dir)

#stop_list = ['\nHuman:', '\n```\n']
#stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
#stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

#class StopOnTokens(StoppingCriteria):
#    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#        for stop_ids in stop_token_ids:
#            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
#                return True
#        return False

#stopping_criteria = StoppingCriteriaList([StopOnTokens()])

class MaxLengthCriteria(StoppingCriteria):
    """
    Custom stopping criteria to stop generation when the output length exceeds a maximum length.
    """
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[1] >= self.max_length

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(llmtemplate)

def load_model():
    print("entering load_model")
    model_dir = "NousResearch/Llama-2-7b-chat-hf"

    #if "tokenizer" not in st.session_state:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
     #   st.session_state.tokenizer = tokenizer
    
    #tokenizer=st.session_state.tokenizer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    
    #if "model" not in st.session_state:
    model = AutoModelForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config,
                                                            torch_dtype=torch.bfloat16, device_map="auto", )
    #    st.session_state.model = model

    #model=st.session_state.model

    print("AutoModelForCausalLM.from_pretrained")
    model.eval()
    print("model eval")
    generate_text = pipeline(
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        # quantization_config=bnb_config,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        do_sample=True,
        max_length=10000,
        # we pass model parameters here too
        #stopping_criteria=stopping_criteria,  # without this model rambles during chat
        max_new_tokens=10000,
        temperature=.001,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
 #       max_new_tokens=8096,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    print("pipeline")
    model_kwargs = {'temperature': 0}
    llm = HuggingFacePipeline(pipeline=generate_text)
    print("HuggingFacePipeline")
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(llmtemplate)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')
    print("ConversationBufferMemory")
    print("exiting load_model")
    return llm, memory

def ingest_into_vectordb(split_docs):
    print("entering ingest_into_vectordb")
    print(split_docs)
#
#    if "embeddings" not in st.session_state:
#        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
#                                        model_kwargs={'device': 'cuda'})
#        st.session_state.embeddings=embeddings

    #loader = CSVLoader(split_docs)
    # Load data from the csv file using the load command
    #csv_data = loader.load()
 #   loader = PyPDFLoader(split_docs)
 #   pages = loader.load()
 #   text_splitter = CharacterTextSplitter(
 #       separator="\n",
 #       chunk_size=1000,
 #       chunk_overlap=150
 #   )
#  docs = text_splitter.split_documents(pages)

### NEW CODE ####
    print("before embedding")
    # Create embeddings
   # if "embeddings" not in st.session_state.embeddings:
    embeddings = HuggingFaceInstructEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={"device": "cuda:0"},
    )
    #    st.session_state.embeddings=embeddings
     #   print("after embedding")

    print("one more after embedding")

    loader = TextLoader(split_docs)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)

    print(db.index.ntotal)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    print("exiting ingest_into_vectordb")
    return db

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(llmtemplate)

def get_conversation_chain(vectordb, llm, memory):
    print("entering get_conversation_chain")
    print("memory",memory)
    #retrieves top 2 serach results.
    retriever = vectordb.as_retriever(search_kwargs={'k': 2})

    conversation_chain = (ConversationalRetrievalChain.from_llm
                          (llm=llm,
                           retriever=retriever,
                           condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                           memory=memory,
                           return_source_documents=False))
    print("Conversational Chain created for the LLM using the vector store")
    print("exiting get_conversation_chain")
    return conversation_chain

def handle_userinput(user_question):
    print("entering handle_userinput")
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        print(i)
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print(message.content)
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    print("exiting handle_userinput")
            
def main():
    load_dotenv()

    #if "tokenizer" not in st.session_state:
    #    st.session_state.tokenizer = None

    #if "model" not in st.session_state:
    #    st.session_state.model = None

    #if "embeddings" not in st.session_state:
    #    st.session_state.embeddings = None

    llm, memory = load_model()

    st.set_page_config(page_title="Chat with your pdf",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDF :")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'")  # , accept_multiple_files=True)
        if pdf_docs:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(pdf_docs.getvalue())
                tmp_file_path = tmp_file.name
        if st.button("Process"):
            with st.spinner("Processing"):
                # create vector store
                vectorstore = ingest_into_vectordb(tmp_file_path)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore,llm, memory)

if __name__ == '__main__':
    main()
