import streamlit as st
import time
from main import extract_pdf_data,split_text,create_vectorstore,fetch_results,model_api_call

st.title("DocuMuse: Your PDF Chatbot")
st.subheader("Chat with your PDF documents using Groq's fast inference!")

# print("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")
# st.subheader("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

@st.cache_resource
def load_vector_store(uploaded_file):
    # Save the uploaded file to a temporary location
    temp_file = "./temp.pdf"

    with open(temp_file, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = uploaded_file.name
        doc = extract_pdf_data(temp_file)
        time.sleep(5)
    st.write(f"PDF file loaded successfully! Number of pages:{len(doc)}")
    # Initiate vectore store
    # create chunks
    chunks = split_text(doc)
    # create vectorstore
    vectorstore = create_vectorstore(chunks)
    return vectorstore

if uploaded_file :
    with st.spinner("Analyszing your docuemnt📖...", show_time=True):
        vectorstore = load_vector_store(uploaded_file)
    st.success("Vectorstore📔 created successfully!👍")
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I am here to answer your questions from your PDF."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display bot message in chat message container
    with st.chat_message("assistant"):
        # Display a spinner while waiting for the response
        with st.spinner("Thinking..."):
            # Fetch results from vectorstore
            vectorstore_results = fetch_results(prompt,vectorstore)
            # Call the model API with the query and vectorstore results
            response = model_api_call(prompt,vectorstore_results)
            # Simulate a delay for the bot's response
            time.sleep(2)
            # Display the results in the chat message container
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Display the response in the chat message container
            st.markdown(response)

st.markdown(
    """
    <style>
    .bottom-right {
        position: fixed;
        bottom: 10px;
        right: 10px;
        z-index: 9999;
    }
    </style>

    <div class="bottom-right">
      <a href="https://groq.com" target="_blank" rel="noopener noreferrer">
        <img
          src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg"
          alt="Powered by Groq for fast inference."
          width="100"
        />
      </a>
    </div>
    """,
    unsafe_allow_html=True
)