import os
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from dotenv import load_dotenv
from google.cloud import vision
from pdf2image import convert_from_bytes
import time 
import datetime 
# For IP address (best effort)
from streamlit.web.server.server import Server 
# For rate limiting state
from collections import defaultdict

from vector_store import VectorStoreManager
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

# --- Rate Limiting Configuration ---
OCR_PAGE_LIMIT_PER_SESSION = 25 # Max OCR pages per user session
# Global dictionary to store page counts per IP/session_id (in-memory, resets on app restart)
# For a more robust solution, use Redis or a database
if 'ocr_page_counts' not in st.session_state:
    st.session_state.ocr_page_counts = defaultdict(lambda: {'count': 0, 'last_reset': datetime.date.today()})

def get_client_ip():
    """Tries to get the client's IP address. Highly dependent on deployment."""
    try:
        session_info = Server.get_current()._get_session_info()
        if session_info and hasattr(session_info, "ws") and hasattr(session_info.ws, "request"):
             # This path might change with Streamlit versions or not be available
            request = session_info.ws.request
            # Check for common headers used by proxies
            if "X-Forwarded-For" in request.headers:
                return request.headers["X-Forwarded-For"].split(',')[0].strip()
            if "X-Real-IP" in request.headers:
                return request.headers["X-Real-IP"].strip()
            if hasattr(request, "remote_ip"): # Fallback for direct connection (dev)
                return request.remote_ip
    except Exception:
        pass # Silently fail if IP cannot be retrieved
    return "unknown_ip" # Fallback

def check_and_update_ocr_limit(ip_or_session_id, pages_to_process):
    """Checks if the OCR page limit has been reached for the given ID and updates it."""
    today = datetime.date.today()
    
    # Reset count if it's a new day
    if st.session_state.ocr_page_counts[ip_or_session_id]['last_reset'] != today:
        st.session_state.ocr_page_counts[ip_or_session_id]['count'] = 0
        st.session_state.ocr_page_counts[ip_or_session_id]['last_reset'] = today

    current_count = st.session_state.ocr_page_counts[ip_or_session_id]['count']
    
    if current_count + pages_to_process > OCR_PAGE_LIMIT_PER_SESSION:
        st.error(f"OCR page limit ({OCR_PAGE_LIMIT_PER_SESSION} pages per session/day) reached. You have processed {current_count} pages. Please try again later or with fewer pages.")
        return False # Limit reached
    
    # If not reached, allow and update (this update happens *after* successful OCR in the calling function)
    return True # Limit not reached

def increment_ocr_count(ip_or_session_id, pages_processed):
    """Increments the OCR page count for the given ID."""
    # Ensure daily reset logic is applied before incrementing
    today = datetime.date.today()
    if st.session_state.ocr_page_counts[ip_or_session_id]['last_reset'] != today:
        st.session_state.ocr_page_counts[ip_or_session_id]['count'] = 0
        st.session_state.ocr_page_counts[ip_or_session_id]['last_reset'] = today
    
    st.session_state.ocr_page_counts[ip_or_session_id]['count'] += pages_processed


# --- Custom CSS for Chat UI (remains the same) ---
def load_css():
    st.markdown("""
    <style>
        .chat-container {
            display: flex; flex-direction: column; height: auto; max-height: 70vh;      
            overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px;
            padding: 15px; background-color: #f9f9f9; margin-bottom: 10px; 
        }
        .chat-message { display: flex; margin-bottom: 15px; align-items: flex-start; }
        .user-message { justify-content: flex-end; }
        .assistant-message { justify-content: flex-start; }
        .message-bubble { max-width: 70%; padding: 10px 15px; border-radius: 18px;
            word-wrap: break-word; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .user-message .message-bubble { background-color: #007bff; color: white;
            border-bottom-right-radius: 5px; margin-left: auto; 
        }
        .assistant-message .message-bubble { background-color: #e9ecef; color: #333;
            border-bottom-left-radius: 5px; 
        }
        .avatar { width: 40px; height: 40px; border-radius: 50%; background-color: #ccc;
            display: flex; align-items: center; justify-content: center;
            font-weight: bold; color: white; margin-right: 10px; 
        }
        .user-message .avatar { margin-left: 10px; margin-right: 0; order: 1; }
        .user-message .message-content { order: 0; }
        .message-timestamp { font-size: 0.75em; color: #888; margin-top: 5px; }
        .user-message .message-timestamp { text-align: right; }
        .assistant-message .message-timestamp { text-align: left; }
        .source-expander .stExpander { border: 1px solid #ddd; border-radius: 5px; margin-top: 8px; }
        .source-expander summary { font-size: 0.9em; font-weight: bold; }
        .source-document { background-color: #f8f9fa; border: 1px solid #eee; border-radius: 4px;
            padding: 8px; margin-bottom: 5px; font-size: 0.85em;
        }
        .source-document strong { color: #007bff; }
        .source-document p { color: #333333; margin-top: 5px; margin-bottom: 0; line-height: 1.4; }
        .stChatInputContainer > div { border-top: 1px solid #e0e0e0; padding-top: 10px; }
        .processed-docs-container { padding: 10px; margin-bottom: 15px; }
        .processed-docs-container h4 { color: #333; margin-bottom: 10px; font-size: 1.1em; }
        .processed-doc-item { background-color: #ffffff; border: 1px solid #e0e0e0;
            border-radius: 6px; padding: 10px 15px; margin-bottom: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-size: 0.9em;
            display: flex; align-items: center;
        }
        .processed-doc-item strong { color: #333; margin-left: 8px; }
        .processed-doc-item-type { color: #555; font-weight: normal; margin-left: 5px; font-size: 0.9em;}
        .chat-start-prompt { text-align:center; margin-top:20px; color: #777; font-style: italic;}
    </style>
    """, unsafe_allow_html=True)


# Initialize Google Vision client
@st.cache_resource
def get_vision_client():
    try:
        client = vision.ImageAnnotatorClient()
        return client
    except Exception as e:
        st.error(f"Failed to initialize Google Vision client: {e}")
        st.warning("OCR functionality will be unavailable. Ensure GOOGLE_APPLICATION_CREDENTIALS is set correctly.")
        return None

# --- OCR Functions with Rate Limiting ---
def perform_ocr_on_image_bytes(image_bytes, filename_for_log="image"):
    client_ip = get_client_ip() # Or use st.session_state.session_id if IP is unreliable
    
    # Check limit for 1 page (since it's one image)
    if not check_and_update_ocr_limit(client_ip, 1):
        return None # Limit reached

    client = get_vision_client()
    if not client: return None
    try:
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        if response.error.message:
            st.error(f"OCR Error for '{filename_for_log}': {response.error.message}")
            return None
        text = response.text_annotations[0].description if response.text_annotations else None
        if text:
            increment_ocr_count(client_ip, 1) # Increment count on successful OCR
        return text
    except Exception as e:
        st.error(f"OCR processing error for '{filename_for_log}': {e}")
        return None

def perform_ocr_on_pdf_bytes(pdf_bytes, filename_for_log=""):
    client_ip = get_client_ip() # Or use st.session_state.session_id
    client = get_vision_client()
    if not client: return None
    
    try:
        # Determine number of pages to process (up to 10 as before)
        # This is a bit tricky as convert_from_bytes loads all images first.
        # We'll convert, then check against remaining quota.
        # For a very large PDF, this might still load many images into memory before stopping.
        temp_images_for_page_count = convert_from_bytes(pdf_bytes, last_page=10, thread_count=1, fmt='jpeg', size=(100,None)) # Minimal conversion for count
        num_pages_to_ocr = len(temp_images_for_page_count)
        del temp_images_for_page_count # Free memory

        if not check_and_update_ocr_limit(client_ip, num_pages_to_ocr):
            return None # Limit reached before processing

        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=num_pages_to_ocr) # Actual conversion
        full_text = ""
        pages_successfully_ocred = 0
        
        for i, image in enumerate(images):
            # No need to check limit per page here again, already checked for the batch
            byte_arr = BytesIO()
            image.save(byte_arr, format='PNG')
            text_from_page = perform_ocr_on_image_bytes_internal(byte_arr.getvalue(), client, f"{filename_for_log} page {i+1}") # Internal call without re-checking limit
            if text_from_page:
                full_text += f"\n--- Page {i + 1} (OCR from {filename_for_log}) ---\n{text_from_page}"
                pages_successfully_ocred += 1
        
        if pages_successfully_ocred > 0:
            increment_ocr_count(client_ip, pages_successfully_ocred) # Increment count for successfully OCR'd pages
        
        return full_text if full_text.strip() else None
    except Exception as e:
        st.error(f"PDF OCR Error for '{filename_for_log}': {e}")
        return None

def perform_ocr_on_image_bytes_internal(image_bytes, vision_client, filename_for_log="image"):
    """Internal OCR function that doesn't do rate limit checks itself."""
    # This function is called by perform_ocr_on_pdf_bytes after the batch limit check.
    try:
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image) # Use passed client
        if response.error.message:
            # st.error(f"Internal OCR Error for '{filename_for_log}': {response.error.message}") # Avoid too many errors
            return None
        return response.text_annotations[0].description if response.text_annotations else None
    except Exception:
        # st.error(f"Internal OCR processing error for '{filename_for_log}': {e}")
        return None


# ... (setup_conversation_chain, handle_user_query, other extract_... functions, save_uploaded_file remain the same)
def setup_conversation_chain(vector_store):
    llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192", api_key=os.environ.get("GROQ_API_KEY"))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    return ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 3}), memory=memory, verbose=False, return_source_documents=True)

def handle_user_query(query):
    if 'conversation_chain' not in st.session_state or st.session_state.conversation_chain is None:
        st.warning("Conversation chain not initialized. Please process documents first."); return None, None
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.conversation_chain.invoke({"question": query, "chat_history": st.session_state.get("chat_history_tuples", [])})
            st.session_state.chat_history_tuples = st.session_state.get("chat_history_tuples", []) + [(query, result["answer"])]
            sources = result.get("source_documents", []); 
            if not isinstance(sources, list): sources = []
            return result["answer"], sources
        except Exception as e: st.error(f"Error generating response: {str(e)}"); return None, []

def extract_text_from_pdf_path(file_path, filename_for_log=""):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f); text = ""
            for page_num in range(min(10, len(pdf_reader.pages))):
                page_text = pdf_reader.pages[page_num].extract_text() or ""
                text += f"\n--- Page {page_num + 1} (from {filename_for_log}) ---\n{page_text}"
            return text if text.strip() else None
    except Exception as e: st.error(f"Error in PDF text extraction for '{filename_for_log}': {e}"); return None

def extract_text_from_csv_path(file_path):
    try: df = pd.read_csv(file_path); return df.to_string()
    except Exception as e: st.error(f"Error extracting text from CSV '{os.path.basename(file_path)}': {e}"); return None

def extract_text_from_json_path(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e: st.error(f"Error extracting text from JSON '{os.path.basename(file_path)}': {e}"); return None

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}; response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status(); soup = BeautifulSoup(response.text, 'html.parser')
        for s in soup(["script", "style"]): s.extract()
        text = soup.get_text(); lines = (l.strip() for l in text.splitlines())
        chunks = (p.strip() for l in lines for p in l.split("  "))
        return '\n'.join(c for c in chunks if c)
    except Exception as e: st.error(f"Error extracting text from URL '{url}': {e}"); return None

def save_uploaded_file(uploaded_file):
    file_path = os.path.join("data", uploaded_file.name)
    try:
        with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e: st.error(f"Error saving file '{uploaded_file.name}': {e}"); return None


# --- process_documents_for_rag (calls the rate-limited OCR functions) ---
# This function's core logic (looping through files and calling extraction) remains largely the same,
# but now the OCR calls within it are rate-limited.
def process_documents_for_rag():
    if 'uploaded_files' not in st.session_state or not st.session_state.uploaded_files:
        st.warning("No documents uploaded yet to process for RAG!"); return None

    st.session_state.vector_store = None; st.session_state.conversation_chain = None
    st.session_state.messages = []; st.session_state.chat_history_tuples = []
    
    unique_collection_name = f"rag_collection_{int(time.time())}"
    st.session_state.current_collection_name = unique_collection_name

    all_extracted_documents = []
    progress_bar = st.progress(0)
    total_files = len(st.session_state.uploaded_files)
    status_area = st.empty()

    for i, file_info in enumerate(st.session_state.uploaded_files):
        file_path, file_name, file_type = file_info['path'], file_info['name'], file_info['type']
        text = None
        status_area.info(f"Processing: {file_name} ({file_type}) [{i+1}/{total_files}]")
        try:
            if file_type == 'PDF': # For PDF, direct extraction first, then OCR if needed (OCR is rate-limited)
                text = extract_text_from_pdf_path(file_path, file_name) # Not rate-limited
                if not text or len(text.strip()) < 50:
                    status_area.info(f"Direct PDF for '{file_name}' insufficient. OCR attempt... [{i+1}/{total_files}]")
                    with open(file_path, 'rb') as f_bytes: text = perform_ocr_on_pdf_bytes(f_bytes.read(), file_name) # Rate-limited
            elif file_type == 'handwritten': # Directly uses rate-limited OCR
                with open(file_path, 'rb') as f_bytes: file_bytes = f_bytes.read()
                if file_path.lower().endswith('.pdf'):
                    status_area.info(f"OCR handwritten PDF: {file_name}... [{i+1}/{total_files}]")
                    text = perform_ocr_on_pdf_bytes(file_bytes, file_name) # Rate-limited
                else:
                    status_area.info(f"OCR handwritten image: {file_name}... [{i+1}/{total_files}]")
                    text = perform_ocr_on_image_bytes(file_bytes, file_name) # Rate-limited
            elif file_type == 'Image': # Directly uses rate-limited OCR
                status_area.info(f"OCR image: {file_name}... [{i+1}/{total_files}]")
                with open(file_path, 'rb') as f_bytes: text = perform_ocr_on_image_bytes(f_bytes.read(), file_name) # Rate-limited
            elif file_type == 'CSV': text = extract_text_from_csv_path(file_path)
            elif file_type == 'JSON': text = extract_text_from_json_path(file_path)
            elif file_type == 'website':
                with open(file_path, 'r', encoding='utf-8') as f_text: text = f_text.read()
            
            if text and text.strip(): all_extracted_documents.append({'name': file_name, 'type': file_type, 'text': text})
            elif text is None and (file_type == 'handwritten' or file_type == 'Image' or (file_type == 'PDF' and (not text or len(text.strip()) < 50))):
                # This means OCR was attempted but failed, possibly due to rate limit
                st.warning(f"OCR for '{file_name}' might have been skipped or failed (e.g., rate limit).")
            else: st.warning(f"No usable text from: {file_name}")
        except Exception as e: st.error(f"Error processing {file_name}: {e}")
        progress_bar.progress((i + 1) / total_files)
        
    status_area.empty(); progress_bar.empty()
    if not all_extracted_documents:
        st.error("No text extracted. Vector store cannot be built."); return None
    
    st.info(f"Text extraction complete. Building vector store: {unique_collection_name}...")
    with st.spinner("Generating vector embeddings..."):
        try:
            vector_mgr = VectorStoreManager(collection_name=unique_collection_name)
            vector_store = vector_mgr.create_or_update_vector_store(all_extracted_documents)
            st.session_state.vector_store = vector_store
            st.session_state.conversation_chain = setup_conversation_chain(vector_store)
            st.success(f"Vector store '{unique_collection_name}' ready with {len(all_extracted_documents)} docs!")
            return vector_store
        except Exception as e:
            st.error(f"Failed to create vector store: {e}"); return None

# ... (display_chat_message and main function remain the same as your last provided version)
def display_chat_message(role, content, sources=None, timestamp=None):
    avatar_char = "U" if role == "user" else "A"; message_class = "user-message" if role == "user" else "assistant-message"
    avatar_html = f"<div class='avatar'>{avatar_char}</div>"
    formatted_timestamp = timestamp.strftime("%I:%M %p, %b %d") if timestamp else ""
    if role == "user":
        st.markdown(f"<div class='chat-message {message_class}'><div class='message-content'><div class='message-bubble'>{content}</div><div class='message-timestamp'>{formatted_timestamp}</div></div>{avatar_html}</div>", unsafe_allow_html=True)
    else: 
        st.markdown(f"<div class='chat-message {message_class}'>{avatar_html}<div class='message-content'><div class='message-bubble'>{content}</div><div class='message-timestamp'>{formatted_timestamp}</div></div></div>", unsafe_allow_html=True)
    if role == "assistant" and sources and isinstance(sources, list) and len(sources) > 0:
        with st.container(): 
            st.markdown("<div class='source-expander'>", unsafe_allow_html=True)
            with st.expander("📚 View Sources", expanded=False):
                for i, source_doc in enumerate(sources):
                    doc_name = "Unknown Source"; page_content_snippet = "No content available."
                    if hasattr(source_doc, 'metadata') and source_doc.metadata: doc_name = source_doc.metadata.get('name', source_doc.metadata.get('source', 'Unknown Source'))
                    if hasattr(source_doc, 'page_content') and source_doc.page_content: page_content_snippet = source_doc.page_content[:250] + "..."
                    st.markdown(f"<div class='source-document'><strong>Source {i+1}: {doc_name}</strong><p>{page_content_snippet}</p></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="Document RAG Chatbot"); load_css() 
    st.title("📄 Intelligent Document Assistant")
    st.markdown("Upload documents, websites, or handwritten notes to build a knowledge base, then chat with it. Each 'Process' action creates a fresh knowledge base.")

    if 'uploaded_files' not in st.session_state: st.session_state.uploaded_files = []
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'chat_history_tuples' not in st.session_state: st.session_state.chat_history_tuples = []
    if 'vector_store' not in st.session_state: st.session_state.vector_store = None
    if 'conversation_chain' not in st.session_state: st.session_state.conversation_chain = None
    if 'current_collection_name' not in st.session_state: st.session_state.current_collection_name = None
    # Initialize session_id if not present (alternative to IP if IP is unreliable)
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(time.time()) + "_" + str(os.urandom(4).hex())


    sidebar = st.sidebar
    with sidebar:
        st.header("⚙️ Controls & Uploads"); st.subheader("1. Add Sources")
        with st.container():
            tab_files, tab_website, tab_handwritten = st.tabs(["📁 Files", "🌐 Website", "✍️ Handwritten"])
            with tab_files: # File Uploader Logic
                regular_uploaded_files = st.file_uploader("Upload PDF, CSV, JSON, or Image files", type=["pdf", "csv", "json", "png", "jpg", "jpeg"], accept_multiple_files=True, key="file_uploader_sidebar")
                if regular_uploaded_files:
                    new_files_added=False
                    for uploaded_file in regular_uploaded_files:
                        if not any(f['name'] == uploaded_file.name for f in st.session_state.uploaded_files):
                            file_path = save_uploaded_file(uploaded_file)
                            if file_path:
                                file_type_str = {"application/pdf": "PDF", "text/csv": "CSV", "application/json": "JSON"}.get(uploaded_file.type, "Image" if uploaded_file.type.startswith('image/') else "Unknown")
                                st.session_state.uploaded_files.append({"name": uploaded_file.name, "type": file_type_str, "path": file_path, "size": f"{uploaded_file.size / (1024*1024):.2f} MB"})
                                new_files_added=True
                    if new_files_added: st.experimental_rerun()
            with tab_website: # Website URL Logic
                website_url = st.text_input("Enter website URL", key="website_url_input_sidebar")
                if st.button("Add Website URL", key="add_website_button_sidebar"):
                    if website_url and not any(f['name'] == website_url for f in st.session_state.uploaded_files):
                        with st.spinner(f"Fetching {website_url}..."): text = extract_text_from_url(website_url)
                        if text:
                            safe_fn = "".join(c if c.isalnum() else"_" for c in website_url.split('//')[-1])[:50]; file_name = f"website_{safe_fn}.txt"; fp = os.path.join("data",file_name)
                            try:
                                with open(fp,"w",encoding="utf-8") as f: f.write(text)
                                st.session_state.uploaded_files.append({"name":website_url,"type":"website","path":fp,"size":f"{len(text)/(1024*1024):.2f} MB"})
                                st.experimental_rerun()
                            except Exception as e: st.error(f"Could not save website text: {e}")
                        else: st.error(f"Could not extract text from {website_url}")
                    elif not website_url: st.warning("Please enter a URL.")
                    else: st.warning(f"Website {website_url} already in list.")
            with tab_handwritten: # Handwritten File Logic
                handwritten_file = st.file_uploader("Upload handwritten notes (image/PDF)", type=["png","jpg","jpeg","pdf"], key="handwritten_uploader_sidebar")
                if handwritten_file and st.button("Add Handwritten Document", key="add_handwritten_button_sidebar"):
                    if not any(f['name'] == handwritten_file.name for f in st.session_state.uploaded_files):
                        file_path = save_uploaded_file(handwritten_file)
                        if file_path:
                            st.session_state.uploaded_files.append({"name":handwritten_file.name,"type":"handwritten","path":file_path,"size":f"{handwritten_file.size/(1024*1024):.2f} MB"})
                            st.experimental_rerun()
                    else: st.warning(f"File {handwritten_file.name} already in list.")
        st.subheader("2. Manage Source List")
        if st.session_state.uploaded_files: # Manage Source List Display
            st.caption(f"Current sources: {len(st.session_state.uploaded_files)}")
            files_to_remove_indices = []
            for i, file_info in enumerate(st.session_state.uploaded_files[:5]):
                c1,c2=st.columns([0.8,0.2]); emoji={"PDF":"🔴","CSV":"📊","JSON":"🧱","Image":"🖼️","website":"🌐","handwritten":"✍️"}.get(file_info['type'],"📄")
                disp_name=file_info['name'][:20]+"..." if len(file_info['name'])>23 else file_info['name']
                c1.markdown(f"<small>{emoji} {disp_name}</small>",unsafe_allow_html=True)
                if c2.button("➖",key=f"remove_sidebar_{i}_{file_info['name']}",help=f"Remove {file_info['name']}"): files_to_remove_indices.append(st.session_state.uploaded_files.index(file_info))
            if len(st.session_state.uploaded_files)>5: st.caption(f"... and {len(st.session_state.uploaded_files)-5} more.")
            if files_to_remove_indices:
                for idx in sorted(files_to_remove_indices,reverse=True):
                    rm_file=st.session_state.uploaded_files.pop(idx)
                    try: 
                        if os.path.exists(rm_file['path']): os.remove(rm_file['path'])
                    except Exception as e: st.error(f"Error removing file '{rm_file['name']}': {e}")
                st.experimental_rerun()
            st.markdown("---") # Process Button
            if st.button("⚙️ Process Sources for Chat",type="primary",use_container_width=True,disabled=not st.session_state.uploaded_files): process_documents_for_rag()
        else: st.info("Upload documents via tabs above to begin.")
    
    with st.container(): # Main Chat Area
        st.header("💬 Chat Interface")
        if not st.session_state.vector_store or not st.session_state.conversation_chain:
            st.info("Welcome! Please upload and process documents using the sidebar to enable chat.")
            if st.session_state.current_collection_name: st.caption(f"Last attempted knowledge base: {st.session_state.current_collection_name}")
        else:
            st.success(f"Knowledge Base Ready: **{st.session_state.current_collection_name}** ({len(st.session_state.uploaded_files)} source files)")
            st.markdown("<div class='chat-container' id='chat-container'>",unsafe_allow_html=True)
            if not st.session_state.messages: # Display Processed Docs or Welcome
                if st.session_state.uploaded_files:
                    st.markdown("<div class='processed-docs-container'><h4>Ready to Chat About:</h4>",unsafe_allow_html=True)
                    for doc_info in st.session_state.uploaded_files:
                        emoji={"PDF":"🔴","CSV":"📊","JSON":"🧱","Image":"🖼️","website":"🌐","handwritten":"✍️"}.get(doc_info['type'],"📄")
                        disp_name=doc_info['name'][:50]+"..." if len(doc_info['name'])>53 else doc_info['name']
                        st.markdown(f"<div class='processed-doc-item'>{emoji} <strong>{disp_name}</strong><span class='processed-doc-item-type'>({doc_info['type']})</span></div>",unsafe_allow_html=True)
                    st.markdown("<p class='chat-start-prompt'>Ask a question below to start chatting.</p></div>",unsafe_allow_html=True)
                else: display_chat_message("assistant","Knowledge base is ready. How can I help?",timestamp=datetime.datetime.now())
            else: # Display Chat Messages
                for msg_data in st.session_state.messages: display_chat_message(msg_data["role"],msg_data["content"],msg_data.get("sources"),msg_data.get("timestamp"))
            st.markdown("</div>",unsafe_allow_html=True)
            
    if prompt := st.chat_input("Ask a question...", key=f"chat_input_{st.session_state.current_collection_name}"): # Chat Input Logic
        if not st.session_state.vector_store or not st.session_state.conversation_chain: st.warning("Please process documents first.")
        else:
            user_ts=datetime.datetime.now(); st.session_state.messages.append({"role":"user","content":prompt,"timestamp":user_ts})
            answer,sources=handle_user_query(prompt)
            assistant_ts=datetime.datetime.now()
            if answer is not None: st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources,"timestamp":assistant_ts})
            else: st.session_state.messages.append({"role":"assistant","content":"Sorry, I encountered an issue. Please try again.","timestamp":assistant_ts})
            st.experimental_rerun()

if __name__ == "__main__":
    main()