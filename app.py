import streamlit as st
from streamlit import session_state as ss
import asyncio
import os
import tempfile
import boto3
from mistralai import Mistral
import base64
import json
from pathlib import Path
from mistralai import DocumentURLChunk

# Import the document loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_community.document_loaders import MathpixPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_docling import DoclingLoader
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from llama_parse import LlamaParse
import nest_asyncio
nest_asyncio.apply()

# Function to display PDF using HTML embed
def display_pdf(file_path=None, file_bytes=None):
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    elif file_bytes:
        base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    else:
        return st.error("No valid PDF source provided")
    
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="600" type="application/pdf"></iframe>'
    
    # Display the PDF
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to embed PDF files in base64 format
def get_binary_file_downloader_html(bin_file, file_label):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

# Function to generate PDF preview
def get_pdf_preview(file_path):
    try:
        # First page only
        from pdf2image import convert_from_path
        
        if os.path.exists(file_path):
            with st.spinner("Generating preview..."):
                # Wrap in try-except as pdf2image might not be installed
                try:
                    images = convert_from_path(file_path, first_page=1, last_page=1)
                    if images:
                        return images[0]
                except Exception as e:
                    st.warning(f"Could not generate preview: {str(e)}")
                    return None
        else:
            st.error(f"File not found: {file_path}")
            return None
    except ImportError:
        st.info("Install pdf2image package for PDF previews")
        return None

# Define the document processing functions
async def file_PyPDFLoader(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    
    return "\n\n".join(page.page_content for page in pages)

async def file_UnstructuredLoader(file_path):
    loader = UnstructuredLoader(file_path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

async def file_AmazonTextractPDFLoader(file_path, aws_access_key_id=None, aws_secret_access_key=None, aws_region=None): 
    client = boto3.client('textract',
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key,
                      region_name=aws_region)
    
    loader = AmazonTextractPDFLoader(
        file_path=file_path,
        client=client,
    )
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

async def file_MathpixPDFLoader(file_path, api_key=None):
    loader = MathpixPDFLoader(file_path, api_key=api_key)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

async def file_PDFPlumberLoader(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

async def file_PyPDFium2Loader(file_path):
    loader = PyPDFium2Loader(file_path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

async def file_PyMuPDFLoader(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

async def file_PyMuPDF4LLMLoader(file_path):
    loader = PyMuPDF4LLMLoader(file_path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

async def file_PDFMinerLoader(file_path):
    loader = PDFMinerLoader(file_path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

async def file_DoclingLoader(file_path):
    loader = DoclingLoader(file_path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)

async def file_AzureAIDocumentIntelligenceLoader(file_path, endpoint=None, key=None):
    loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=endpoint, api_key=key, file_path=file_path, api_model="prebuilt-layout")
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


async def file_llamaparse(file_path, llama_api_key=None):
    parser = LlamaParse(
        api_key=llama_api_key,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=1,  # using 1 for single file processing
        verbose=True,
        language="en",  # Optionally you can define a language, default=en
    )
    
    # Process the file
    documents = parser.load_data(file_path)
    
    # Join the content from all documents
    if documents:
        return "\n\n".join(doc.text for doc in documents)
    

async def file_vision_llm(file_path, openai_key=None):
    from vision_llm import DocumentProcessor
    processor = DocumentProcessor(max_concurrent_requests=5, openai_key=openai_key)
    return await processor.process_document(file_path, os.path.basename(file_path))

async def file_mistral_ocr(file_path, mistral_key=None):
    client = Mistral(api_key=mistral_key)
    # Verify PDF file exists
    pdf_file = Path(file_path)
    assert pdf_file.is_file()

    # Upload PDF file to Mistral's OCR service
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_file.stem,
            "content": pdf_file.read_bytes(),
        },
        purpose="ocr",
    )

    # Get URL for the uploaded file
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    # Process PDF with OCR, including embedded images
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=True
    )

    # Convert response to JSON format
    response_dict = json.loads(pdf_response.model_dump_json())
    return "\n\n".join(page["markdown"] for page in response_dict["pages"])

# Setup the Streamlit app
st.set_page_config(page_title="PDF-Parser Playground", page_icon="📄", layout="wide")

# Initialize session state variables
if 'pdf_ref' not in ss:
    ss.pdf_ref = None
if 'selected_file_path' not in ss:
    ss.selected_file_path = None
if 'file_name' not in ss:
    ss.file_name = None

st.title("PDF-Parser Playground")
st.write("Select a sample PDF document or upload your own, then select processing methods to compare extraction results.")

# Define paths for default PDF files from the pdf_files folder
# Using a relative path instead of __file__
pdf_files_dir = "pdf_files"  # Relative path to the pdf_files directory

# Check if directory exists
if not os.path.exists(pdf_files_dir):
    st.warning(f"The pdf_files directory ({pdf_files_dir}) does not exist. Please create it and add sample PDF files.")
    # Create a fallback empty directory
    os.makedirs(pdf_files_dir, exist_ok=True)

# Get available PDF files from the directory
available_pdfs = [f for f in os.listdir(pdf_files_dir) if f.lower().endswith('.pdf')]

# Create a dictionary of sample files
sample_files = {
    pdf_file: os.path.join(pdf_files_dir, pdf_file) for pdf_file in available_pdfs
}

# If no PDFs found, add placeholder message
if not sample_files:
    st.sidebar.warning("No PDF files found in the pdf_files directory. Add some PDF files to enable sample selection.")
    # Add some defaults for UI rendering purposes
    sample_files = {
        "invoice.pdf": None,
        "document.pdf": None
    }

# Add sample files to the left sidebar
st.sidebar.header("Sample PDF Files")

if available_pdfs:
    # Radio buttons for sample document selection with "Upload my own document" as default
    sample_choice = st.sidebar.radio(
        "Select a sample PDF:", 
        ["Upload my own document"] + available_pdfs,
        index=0  # Make "Upload my own document" the default
    )
else:
    st.sidebar.warning("No PDF files found in the pdf_files folder.")
    sample_choice = "Upload my own document"

# File selector based on choice - we'll use session state to track these

# Add processing methods to the sidebar
st.sidebar.markdown("---")
st.sidebar.header("Processing Methods")

available_methods = [
    "PyPDFLoader (Basic)",
    "UnstructuredLoader",
    "PDFPlumberLoader",
    "PyPDFium2Loader",
    "PyMuPDFLoader",
    "PyMuPDF4LLMLoader",
    "DoclingLoader",
    "AmazonTextractPDFLoader (AWS)",
    "MathpixPDFLoader (API Key)",
    "AzureAIDocumentIntelligenceLoader (Azure)",
    "VisionLLM (OpenAI)",
    "MistralOCR (Mistral)",
    "LlamaParse (API Key)"  # Add this line
]

# Add a "Select All" checkbox
select_all = st.sidebar.checkbox("Select All Methods")

# Conditional multiselect - if "Select All" is checked, select all methods
if select_all:
    selected_methods = available_methods.copy()
    # Display the methods as disabled multiselect
    st.sidebar.multiselect(
        "Selected Methods:",
        available_methods,
        default=available_methods,
        disabled=True
    )
else:
    # Create a multiselect without any limit
    selected_methods = st.sidebar.multiselect(
        "Choose processing methods to compare:",
        available_methods
    )

# Conditional inputs based on selected methods
if any("AmazonTextractPDFLoader" in method for method in selected_methods):
    st.sidebar.subheader("AWS Credentials")
    aws_access_key_id = st.sidebar.text_input("AWS Access Key ID", type="password")
    aws_secret_access_key = st.sidebar.text_input("AWS Secret Access Key", type="password")
    aws_region = st.sidebar.text_input("AWS Region", value="us-east-1")

if any("MathpixPDFLoader" in method for method in selected_methods):
    st.sidebar.subheader("Mathpix Credentials")
    mathpix_api_key = st.sidebar.text_input("Mathpix API Key", type="password")

if any("AzureAIDocumentIntelligenceLoader" in method for method in selected_methods):
    st.sidebar.subheader("Azure AI Document Intelligence Credentials")
    azure_endpoint = st.sidebar.text_input("Azure Endpoint URL")
    azure_key = st.sidebar.text_input("Azure API Key", type="password")

if any("VisionLLM" in method for method in selected_methods):
    st.sidebar.subheader("OpenAI Credentials")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

if any("MistralOCR" in method for method in selected_methods):
    st.sidebar.subheader("Mistral Credentials")
    mistral_key = st.sidebar.text_input("Mistral API Key", type="password")

# Add this to the credential input section (around line ~450):
if any("LlamaParse" in method for method in selected_methods):
    st.sidebar.subheader("LlamaParse Credentials")
    llama_api_key = st.sidebar.text_input("LlamaParse API Key", type="password")

# Process button is now in the right column with the results

# Function to run the selected processing method
def run_processing_method(file_path, method):
    if method == "PyPDFLoader (Basic)":
        return asyncio.run(file_PyPDFLoader(file_path))
    elif method == "UnstructuredLoader":
        return asyncio.run(file_UnstructuredLoader(file_path))
    elif method == "PDFPlumberLoader":
        return asyncio.run(file_PDFPlumberLoader(file_path))
    elif method == "PyPDFium2Loader":
        return asyncio.run(file_PyPDFium2Loader(file_path))
    elif method == "PyMuPDFLoader":
        return asyncio.run(file_PyMuPDFLoader(file_path))
    elif method == "PyMuPDF4LLMLoader":
        return asyncio.run(file_PyMuPDF4LLMLoader(file_path))
    elif method == "PDFMinerLoader":
        return asyncio.run(file_PDFMinerLoader(file_path))
    elif method == "DoclingLoader":
        return asyncio.run(file_DoclingLoader(file_path))
    elif method == "AmazonTextractPDFLoader (AWS)":
        return asyncio.run(file_AmazonTextractPDFLoader(
            file_path, 
            aws_access_key_id, 
            aws_secret_access_key, 
            aws_region
        ))
    elif method == "MathpixPDFLoader (API Key)":
        return asyncio.run(file_MathpixPDFLoader(file_path, mathpix_api_key))
    elif method == "AzureAIDocumentIntelligenceLoader (Azure)":
        return asyncio.run(file_AzureAIDocumentIntelligenceLoader(
            file_path, 
            azure_endpoint, 
            azure_key
        ))
    elif method == "VisionLLM (OpenAI)":
        return asyncio.run(file_vision_llm(file_path, openai_key))
    elif method == "MistralOCR (Mistral)":
        return asyncio.run(file_mistral_ocr(file_path, mistral_key))
    elif method == "LlamaParse (API Key)":
        return asyncio.run(file_llamaparse(file_path, llama_api_key))
    else:
        return "Unknown processing method"

# Main area for uploading files and showing preview/results
st.markdown("## Document Upload & Preview")

# Add a clean container for the main content
main_container = st.container()

# Create two equal columns with proper spacing
col1, col2 = main_container.columns([1, 1], gap="large")

with col1:
    # Upload area or selected PDF display
    if sample_choice == "Upload my own document":
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="pdf")
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                ss.selected_file_path = tmp_file.name
                ss.file_name = uploaded_file.name
                ss.pdf_ref = uploaded_file
                st.success(f"Successfully uploaded: {ss.file_name}")
    else:
        # Use the selected sample file
        ss.selected_file_path = sample_files[sample_choice]
        ss.file_name = sample_choice
        # For sample files, we need to read the file into memory for the PDF viewer
        if os.path.exists(ss.selected_file_path):
            with open(ss.selected_file_path, 'rb') as f:
                binary_data = f.read()
                # Create a mock file object for session state
                from io import BytesIO
                file_bytes = BytesIO(binary_data)
                file_bytes.name = ss.file_name
                ss.pdf_ref = file_bytes
            st.info(f"Using sample document: {sample_choice}")
    
    # Display PDF preview in a clean container
    if ss.pdf_ref:
        st.markdown("### PDF Preview")
        try:
            # Get the binary data
            pdf_bytes = ss.pdf_ref.getvalue()
            
            # Show a download button for the PDF
            st.download_button(
                "📥 Download PDF for viewing",
                data=pdf_bytes,
                file_name=ss.file_name,
                mime="application/pdf"
            )
            
            # Try using PyMuPDF (which should already be installed for PyMuPDFLoader)
            try:
                import io
                import fitz  # PyMuPDF
                import PIL.Image
                
                # Open the PDF from memory
                with st.spinner("Generating PDF preview..."):
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                    page_count = len(pdf_document)
                    
                    # Show info about the PDF
                    st.info(f"PDF has {page_count} page{'s' if page_count > 1 else ''}.")
                    
                    # Create tabs for page navigation
                    if page_count > 1:
                        # Create page selectors in 5-page groups
                        page_groups = [list(range(i, min(i+5, page_count))) for i in range(0, page_count, 5)]
                        
                        # Let user select page group if many pages
                        if len(page_groups) > 1:
                            group_labels = [f"Pages {g[0]+1}-{g[-1]+1}" for g in page_groups]
                            selected_group_idx = st.radio("Select page range:", options=range(len(group_labels)), 
                                                         format_func=lambda i: group_labels[i], horizontal=True)
                            pages_to_show = page_groups[selected_group_idx]
                        else:
                            pages_to_show = page_groups[0]
                        
                        # Create tabs for the selected pages
                        page_tabs = st.tabs([f"Page {i+1}" for i in pages_to_show])
                        
                        # Display each page in its tab
                        for tab_idx, page_num in enumerate(pages_to_show):
                            with page_tabs[tab_idx]:
                                page = pdf_document.load_page(page_num)
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
                                img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
                                st.image(img, caption=f"Page {page_num+1} of {page_count}", use_container_width=True)
                    else:
                        # Just show the single page without tabs
                        page = pdf_document.load_page(0)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
                        st.image(img, caption=f"Page 1 of 1", use_container_width=True)
            
            except ImportError:
                st.warning("Could not generate detailed PDF preview. Using simpler preview.")
                # Fall back to a simpler method
                if ss.selected_file_path:
                    try:
                        from PIL import Image
                        import subprocess
                        import os
                        
                        # Save PDF to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                            tmp_pdf.write(pdf_bytes)
                            pdf_path = tmp_pdf.name
                        
                        st.markdown("Preview not available directly in the browser. Please download the PDF to view it.")
                        st.image("https://via.placeholder.com/700x900?text=Download+PDF+to+view", width=600)
                    except Exception as inner_e:
                        st.error(f"Error generating preview: {str(inner_e)}")
                        st.image("https://via.placeholder.com/700x900?text=PDF+Preview+Unavailable", width=600)
                else:
                    st.markdown("PDF preview not available. Please download the file to view it.")
                    
        except Exception as e:
            st.error(f"Error displaying PDF: {str(e)}")
            st.markdown("PDF preview could not be generated. Please download the file to view it.")
            # Still provide the download button
            st.download_button(
                "📥 Download PDF",
                data=ss.pdf_ref.getvalue() if ss.pdf_ref else b"",
                file_name=ss.file_name if ss.file_name else "document.pdf",
                mime="application/pdf",
                key="download_fallback"
            )

with col2:
    # Processing options and methods tabs
    st.markdown("### Processing Results")
    
    # Place the process button directly under the Processing Results header
    # Create a centered process button with some styling
    
    process_button = st.button("Process Document", key="process_main")
    
    # Process the document when the button is clicked
    if 'process_main' in st.session_state and st.session_state.process_main:
        process_requested = True
    else:
        process_requested = False
        
    if process_button or process_requested:
        if not selected_methods:
            st.warning("Please select at least one processing method in the sidebar.")
        elif ss.selected_file_path is None:
            st.warning("Please select a sample document or upload your own PDF document.")
        else:
            results = {}
            
            try:
                # First process all methods
                with st.spinner(f"Processing document '{ss.file_name}' with selected methods..."):
                    for method in selected_methods:  # Limit to first 4
                        try:
                            # Process the document and get the extracted text
                            extracted_text = run_processing_method(ss.selected_file_path, method)
                            results[method] = extracted_text
                        except Exception as e:
                            st.error(f"Error processing with {method}: {str(e)}")
                            results[method] = f"Error: {str(e)}"
                
                # Then display results in tabs that match the width of the PDF viewer
                if results:
                    # Create tabs for each method
                    if len(selected_methods) > 0:
                        tabs = st.tabs([method.split(' ')[0] for method in selected_methods])
                        
                        for i, (method, text) in enumerate(results.items()):
                            with tabs[i]:
                                if not text.startswith("Error:"):
                                    # Use a fixed height text area to match PDF viewer height
                                    st.text_area(
                                        f"Extracted Text", 
                                        text, 
                                        height=650,
                                        key=f"text_{method}"
                                    )
                                    
                                    # Add metrics in a row
                                    metric_cols = st.columns(3)
                                    with metric_cols[0]:
                                        st.metric("Characters", len(text))
                                    with metric_cols[1]:
                                        st.metric("Words", len(text.split()))
                                    with metric_cols[2]:
                                        st.metric("Lines", len(text.splitlines()))
                                    
                                    # Add download button
                                    st.download_button(
                                        label=f"Download Text",
                                        data=text,
                                        file_name=f"{ss.file_name.split('.')[0]}_{method.split(' ')[0]}.txt",
                                        mime="text/plain",
                                        key=f"download_{method}"
                                    )
                                else:
                                    st.error(text)
                        
                        # Add a button to download all results in a single file
                        combined_text = ""
                        for method, text in results.items():
                            combined_text += f"=== {method} ===\n\n{text}\n\n"
                        
                        st.download_button(
                            label="Download All Results",
                            data=combined_text,
                            file_name=f"{ss.file_name.split('.')[0]}_all_methods.txt",
                            mime="text/plain",
                            key="download_all"
                        )
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                # Do not clean up the temporary file here to keep it available for comparison
                pass

# Add comparison summary in its own section below the main columns
if 'results' in locals() and len(selected_methods) > 1:
    st.markdown("---")
    st.markdown("## Comparison Summary")
    
    # Calculate statistics for comparison
    comparison_data = []
    
    for method, text in results.items():
        if isinstance(text, str) and not text.startswith("Error:"):
            char_count = len(text)
            word_count = len(text.split())
            line_count = len(text.splitlines())
            
            comparison_data.append({
                "Method": method,
                "Characters": char_count,
                "Words": word_count,
                "Lines": line_count
            })
    
    if comparison_data:
        # Use columns for the table and the notes
        stat_col1, stat_col2 = st.columns([2, 1])
        
        with stat_col1:
            st.subheader("Text extraction statistics")
            st.table(comparison_data)
            
        with stat_col2:
            st.subheader("Notes")
            st.markdown("""
            - **Character count**: Higher counts may indicate better text extraction or possible duplication
            - **Word count**: Compare with expected document content
            - **Line count**: Can help identify if paragraph structure is preserved
            """)
        
        st.markdown("### Visualization")
        
        # Create a bar chart for visual comparison
        import pandas as pd
        import altair as alt
        
        df = pd.DataFrame(comparison_data)
        
        # Create charts with consistent styling
        base_chart = alt.Chart(df).encode(
            x=alt.X('Method:N', title='Method', sort=None),
            color=alt.Color('Method:N', legend=None)
        )
        
        # Character count chart
        char_chart = base_chart.mark_bar().encode(
            y=alt.Y('Characters:Q', title='Character Count')
        ).properties(title='Character Count Comparison', height=300)
        
        # Word count chart
        word_chart = base_chart.mark_bar().encode(
            y=alt.Y('Words:Q', title='Word Count')
        ).properties(title='Word Count Comparison', height=300)
        
        # Line count chart
        line_chart = base_chart.mark_bar().encode(
            y=alt.Y('Lines:Q', title='Line Count')
        ).properties(title='Line Count Comparison', height=300)
        
        # Display charts in a grid
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.altair_chart(char_chart, use_container_width=True)
            st.altair_chart(line_chart, use_container_width=True)
            
        with chart_col2:
            st.altair_chart(word_chart, use_container_width=True)
            # Add blank space to align with the other column
            st.write("")
            st.write("")
            st.markdown("""
            Remember that raw statistics don't tell the full story - review the actual extracted text 
            to evaluate quality, especially for specific document features like tables, formatting, 
            and specialized content.
            """)

# Remove the comparison section from outside the columns
# All comparison code is now inside the right column

# Cleanup code - we'll move this to a button
if st.sidebar.button("Clear All"):
    if sample_choice == "Upload my own document" and ss.selected_file_path and os.path.exists(ss.selected_file_path):
        try:
            os.unlink(ss.selected_file_path)
        except:
            pass
    ss.selected_file_path = None
    ss.file_name = None
    ss.pdf_ref = None
    st.experimental_rerun()

# Add information about sample files in an expander
with st.sidebar.expander("About Sample Documents"):
    st.markdown("""
    ### Sample Documents
    
    The application uses sample PDF documents from the `pdf_files` folder to test different extraction methods.
    
    These documents are designed to showcase the strengths and weaknesses of different 
    PDF extraction methods across various document types.
    
    #### Available Sample Documents:
    """)
    
    # List available PDF files dynamically
    if available_pdfs:
        for pdf_file in available_pdfs:
            st.markdown(f"- **{pdf_file}**")
    else:
        st.warning("No sample PDF files found in the pdf_files directory.")
    
    st.markdown("""
    You can also upload your own PDF documents to compare extraction results.
    """)

# Add some information about each method in the sidebar
with st.sidebar.expander("About Processing Methods"):
    st.markdown("""
    - **PyPDFLoader**: Basic PDF text extraction
    - **UnstructuredLoader**: Extracts text with layout awareness
    - **PDFPlumberLoader**: Good for PDFs with tables
    - **PyPDFium2Loader**: Fast PDF processing
    - **PyMuPDFLoader**: High-quality extraction with formatting
    - **PyMuPDF4LLMLoader**: Optimized for language models
    - **AmazonTextractPDFLoader**: AWS OCR service
    - **MathpixPDFLoader**: Good for mathematical content
    - **AzureAIDocumentIntelligenceLoader**: Azure Document Analysis
    - **Docling** :IBM Docling loader
    - **VisionLLM**: Uses OpenAI vision models
    - **MistralOCR**: Uses Mistral OCR capabilities
    - **LlamaParse**: Anthropic's high-quality PDF parser optimized for LLMs
    """)

# Add comparison information
with st.sidebar.expander("Comparison Tips"):
    st.markdown("""
    **When comparing methods, consider:**
    
    - **Text accuracy**: Check if all text is correctly extracted
    - **Layout preservation**: How well the document structure is maintained
    - **Table handling**: Ability to properly extract tabular data
    - **Processing speed**: Time taken to extract text
    - **Special character handling**: How well special characters are preserved
    - **Image text extraction**: Ability to extract text from images within PDFs
    
    Different methods excel at different document types:
    - For **text-heavy documents**, PyMuPDFLoader or PDFMinerLoader often work well
    - For **tables**, PDFPlumberLoader is typically best
    - For **scanned documents**, OCR methods (Textract, Azure, Mistral) perform better
    - For **mathematical content**, MathpixPDFLoader is specialized
    """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.info("PDF Document Processor v1.0")
