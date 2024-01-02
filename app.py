import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from sentence_transformers import SentenceTransformer
from streamlit_chat import message
import google.generativeai as genai
import os

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your file😊", page_icon="✨", layout="wide")

    # Add emojis and styling to the header
    st.header("🌟 Chat with your Document and Images 📄📸")
    st.subheader('By Abdul Rehman Zahid 😊')

    selected_option = st.sidebar.radio("Select option:", ["Chat_with_Documents", "Chat_with_Images_using_Gemini"])

    if selected_option == "Chat_with_Documents":
        document_gpt_code()
    elif selected_option == "Chat_with_Images_using_Gemini":
        gemini_ai_pro_vision_code()

def document_gpt_code():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "user_text_input" not in st.session_state:
        st.session_state.user_text_input = None
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None  # Added line for OpenAI API Key

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'txt'], accept_multiple_files=True)
        st.session_state.user_text_input = st.text_area("Enter your text here:")
        st.session_state.openai_api_key = st.text_input("Enter your OpenAI API Key:")
        embadings_button = st.button("Generate Embeddings")
        process_button = st.button("Process")

    if embadings_button:
        if not st.session_state.user_text_input:
            st.warning("Please enter some text before generating embeddings.")
            st.stop()

        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API Key.")
            st.stop()

        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")
        text_chunks = get_text_chunks(files_text + "\n" + st.session_state.user_text_input)
        st.write("File chunks created...")
        vetorestore = get_vectorstore(text_chunks)
        st.write("Vector Store Created...")
        st.session_state.conversation = get_conversation_chain(vetorestore, st.session_state.openai_api_key)
        st.session_state.processComplete = True

    if process_button:
        if not uploaded_files:
            st.warning("Please upload a file before processing.")
            st.stop()

        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API Key.")
            st.stop()

        files_text = get_files_text(uploaded_files)
        st.write("File loaded...")
        text_chunks = get_text_chunks(files_text)
        st.write("File chunks created...")
        vetorestore = get_vectorstore(text_chunks)
        st.write("Vector Store Created...")
        st.session_state.conversation = get_conversation_chain(vetorestore, st.session_state.openai_api_key)
        st.session_state.processComplete = True

    # Move the conversation history container outside the 'if' block
    response_container = st.container()

    # Retrieve user input outside the 'if process:' block
    user_question_key = "user_question_key"
    user_question = st.text_input("Ask Question about your files.", key=user_question_key)

    if st.session_state.processComplete == True and user_question:
        handle_user_input(user_question)
        # Clear the input field after handling user input
        st.session_state.user_question = None

    with response_container:
        if st.session_state.chat_history:
            for i, messages in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    message(messages.content, is_user=True, key=str(i))
                else:
                    message(messages.content, key=str(i))

def gemini_ai_pro_vision_code():
    GOOGLE_API_KEY = "AIzaSyA839oSQIG0-kbEhfXvGLzm9i8jZ7Y5xTw"
    genai.configure(api_key=GOOGLE_API_KEY)

    openai_key = st.secrets["OPENAI_API_KEY"]
    
    st.title("Gemini AI Pro Vision")

    # Your Streamlit code for uploading an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Prompt for generating content
        prompt = st.text_area("Enter prompt for gemini-pro-vision model")

        if st.button("Generate Content"):
            # Check if both image and prompt are provided
            if image is not None and prompt:
                # Combine prompt and image for input to the model
                input_data = [prompt, image]

                # Process the combined input using gemini-pro-vision model
                model_vision = genai.GenerativeModel("gemini-pro-vision")
                response_vision = model_vision.generate_content(input_data)

                # Display the response

                st.text_area("Gemini Pro Vision Response:", value= response_vision.text)
            else:
                st.warning("Please provide both an image and a prompt.")

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        elif file_extension == ".txt":
            text += get_txt_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_txt_text(file):
    text = ""
    try:
        text = file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading the text file: {str(e)}")
    return text

def get_csv_text(file):
    return "a"

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

if __name__ == '__main__':
    main()
