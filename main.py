import streamlit as st
from PIL import Image
import io
import time 
import numpy as np
import easyocr


# Cache the model loading to prevent Streamlit from reloading it on every rerun
@st.cache_resource
def get_ocr_reader():
    """Initializes and caches the EasyOCR reader model (English language).
        Returns: easyocr Reader instance
    """

    return easyocr.Reader(['en'])


def get_ocr_result(uploaded_file):
    """
    Uses the EasyOCR library to extract text from the uploaded image.
    Returns: string containing extracted text 
    """

    st.session_state['is_processing'] = True
    st.session_state['ocr_result'] = None
    st.toast("Starting EasyOCR extraction...", icon="🔍")

    try:
        # Get the cached EasyOCR reader instance
        reader = get_ocr_reader()
        image_np = np.array(Image.open(uploaded_file))
        results = reader.readtext(image_np, detail=0) # detail=0 returns only the text strings      
        extracted_text = "\n".join(results)           # Combine the list of text strings into a single block
        final_output = f"""
## 📝 Extracted Text (EasyOCR Result)

**Source File:** `{uploaded_file.name}`

---

**OCR Content:**

```
{extracted_text}
```

Processed successfully on: {time.ctime()}
"""
        st.session_state['ocr_result'] = final_output
        st.toast("OCR completed! Text extracted successfully.", icon="✅")

    except Exception as e:
        st.error(f"An error occurred during OCR processing with EasyOCR: {e}")
        st.session_state['ocr_result'] = f"Error: Could not process image. {e}"

    finally:
        # Reset session state processing flag
        st.session_state['is_processing'] = False


# Page Configuration
st.set_page_config(
    page_title="Fresh-Gens",
    page_icon="⚡",
    layout="wide"
)
 
# State Management
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
    

if 'ocr_result' not in st.session_state:
    st.session_state['ocr_result'] = None


if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False


# UI Components

st.title("🖼️ EasyOCR Streamlit App")
st.subheader("Upload an image and extract text using open-source vision.")

st.markdown("""
Upload a clear image containing text. This version uses **EasyOCR**, an open-source library, 
to recognize and display the text content.
""")
st.markdown("---")

# File Uploader 
uploaded_file = st.file_uploader(
    "Choose an image file to upload:",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    key="image_uploader"
)


if uploaded_file is not None and uploaded_file != st.session_state['uploaded_file']:
    st.session_state['uploaded_file'] = uploaded_file
    st.session_state['ocr_result'] = None # Clear old result on new upload
    st.toast(f"File uploaded successfully: {uploaded_file.name}", icon="✅")

elif uploaded_file is None and st.session_state['uploaded_file'] is not None:
    st.session_state['uploaded_file'] = None
    st.session_state['ocr_result'] = None
    st.toast("File cleared from session.", icon="🗑️")


# Display 

file_to_display = st.session_state['uploaded_file']

if file_to_display:
    st.success(f"File '{file_to_display.name}' is ready for OCR processing.")
    
    # Use columns to display image, controls, and metadata
    img_col, meta_col = st.columns([2, 1])

    with img_col:
        image_data = Image.open(file_to_display)
        st.image(image_data, caption=file_to_display.name, use_container_width=True) 

    with meta_col:
        st.subheader("Controls")
        

        if st.button("Extract Text (EasyOCR)", type="primary", disabled=st.session_state['is_processing']):
            get_ocr_result(file_to_display)

        st.markdown("---")
        st.subheader("File Details")
        st.markdown(f"**Name:** `{file_to_display.name}`")
        st.markdown(f"**Type:** `{file_to_display.type}`")
        st.markdown(f"**Size:** `{file_to_display.size / 1024:.2f} KB`")

    st.markdown("## Extracted Text Output")


    if st.session_state['is_processing']:
        st.info("EasyOCR is analyzing the image for text content...")
        # Show a spinner instead of a simple progress bar while processing
        with st.spinner('Processing image with deep learning model...'):
            time.sleep(0.1) # Small delay to allow spinner to show up
    elif st.session_state['ocr_result']:
        # Display the extracted text using markdown
        st.markdown(st.session_state['ocr_result'])
    else:
        st.info("Press 'Extract Text (EasyOCR)' to begin the analysis.")


else:
    st.info("No image file has been uploaded yet. Upload one to begin the OCR process.")


# Footer 
st.sidebar.markdown("### Application Information")
st.sidebar.info("This application uses Streamlit's file handling and the open-source EasyOCR library.")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit and 📖")
