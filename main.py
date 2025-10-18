import streamlit as st
from PIL import Image
import io
import time 
import numpy as np
import easyocr 
from gtts import gTTS 
from io import BytesIO 


@st.cache_resource
def get_ocr_reader():
    """Initializes and caches the EasyOCR reader model (English language)."""
    return easyocr.Reader(['en'])


def extract_raw_text(uploaded_file):
    """
    Performs OCR and returns only the raw extracted text string.

    Args:
        uploaded_file: The file object from Streamlit's st.file_uploader.
    
    Returns:
        str: The extracted text, combined by newlines.
    """
    reader = get_ocr_reader()
    # Open the image using PIL and convert to NumPy array
    image_pil = Image.open(uploaded_file)
    image_np = np.array(image_pil)
    
    # Perform OCR, returning only the text strings (detail=0)
    results = reader.readtext(image_np, detail=0) 
    return "\n".join(results) 


def generate_audio(text_to_read):
    """
    Generates audio bytes from text using gTTS.
    """
    if not text_to_read:
        return None
        
    st.toast("Generating speech...", icon="🔊")
    try:
        tts = gTTS(text=text_to_read, lang='en')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception as e:
        st.error(f"Error generating TTS audio: {e}")
        return None


def run_ocr_pipeline(uploaded_file):
    """
    Orchestrates the OCR extraction, handles state, formatting, and errors.
    """

    st.session_state['is_processing'] = True
    st.session_state['ocr_result'] = None
    st.session_state['raw_ocr_text'] = None 
    st.toast("Starting EasyOCR extraction...", icon="🔍")

    try:
        extracted_text = extract_raw_text(uploaded_file)
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
        st.session_state['raw_ocr_text'] = extracted_text # Store clean text for TTS
        st.session_state['ocr_result'] = final_output     # Store formatted text for display
        st.toast("OCR completed! Text extracted successfully.", icon="✅")

    except Exception as e:
        st.error(f"An error occurred during OCR processing with EasyOCR: {e}")
        st.session_state['ocr_result'] = f"Error: Could not process image. {e}"

    finally:
        st.session_state['is_processing'] = False


st.set_page_config(
    page_title="Fresh-Gens",
    page_icon="⚡",
    layout="wide"
)


def tts_callback():
    """
    Callback function for the TTS button.
    Now uses the clean 'raw_ocr_text' state directly.
    """
    raw_text = st.session_state.get('raw_ocr_text')
    
    if raw_text:
        audio_data = generate_audio(raw_text)
        if audio_data:
            st.session_state['audio_bytes'] = audio_data
    else:
        st.error("Cannot read text: No raw OCR result found in state.")

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'ocr_result' not in st.session_state:
    st.session_state['ocr_result'] = None
if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False
if 'audio_bytes' not in st.session_state:
    st.session_state['audio_bytes'] = None
if 'raw_ocr_text' not in st.session_state:
    st.session_state['raw_ocr_text'] = None



# UI Components ---

st.title("🖼️ EasyOCR & 🔊 Text-to-Speech App")
st.subheader("Extract text from an image and listen to it using open-source libraries.")

st.markdown("---")

# File Uploader 
uploaded_file = st.file_uploader(
    "Choose an image file to upload:",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    key="image_uploader"
)

# Handle Upload and State Management (Check for new upload/clear)
if uploaded_file is not None and uploaded_file != st.session_state['uploaded_file']:
    st.session_state['uploaded_file'] = uploaded_file
    st.session_state['ocr_result'] = None
    st.session_state['raw_ocr_text'] = None
    st.session_state['audio_bytes'] = None
    st.toast(f"File uploaded successfully: {uploaded_file.name}", icon="✅")

elif uploaded_file is None and st.session_state['uploaded_file'] is not None:
    st.session_state['uploaded_file'] = None
    st.session_state['ocr_result'] = None
    st.session_state['raw_ocr_text'] = None # Clear new state
    st.session_state['audio_bytes'] = None 
    st.toast("File cleared from session.", icon="🗑️")


# Display 
file_to_display = st.session_state['uploaded_file']

if file_to_display:
    st.success(f"File '{file_to_display.name}' is ready for processing.")
    
    img_col, meta_col = st.columns([2, 1])

    with img_col:
        image_data = Image.open(file_to_display)
        st.image(image_data, caption=file_to_display.name, use_container_width=True) 

    with meta_col:
        st.subheader("Controls")
        
        # OCR Button - Now calls the refined pipeline
        if st.button("Extract Text (EasyOCR)", type="primary", disabled=st.session_state['is_processing']):
            run_ocr_pipeline(file_to_display)
        
        # TTS Button - Enabled only if OCR is not processing AND we have raw text
        tts_disabled = st.session_state['is_processing'] or not st.session_state['raw_ocr_text']

        if st.button("Read Text Aloud (TTS)", disabled=tts_disabled, on_click=tts_callback):
            pass 
            
        st.markdown("---")
        st.subheader("File Details")
        st.markdown(f"**Name:** `{file_to_display.name}`")
        st.markdown(f"**Type:** `{file_to_display.type}`")
        st.markdown(f"**Size:** `{file_to_display.size / 1024:.2f} KB`")

    # Output Display Section
    st.markdown("## Extracted Text Output")

    if st.session_state['is_processing']:
        st.info("EasyOCR is analyzing the image for text content...")
        with st.spinner('Processing image with deep learning model...'):
            time.sleep(0.1) 
    elif st.session_state['ocr_result']:
        # Display the extracted text using markdown
        st.markdown(st.session_state['ocr_result'])
        
        # Audio Player
        if st.session_state['audio_bytes']:
             st.subheader("Audio Playback")
             st.audio(st.session_state['audio_bytes'], format='audio/mp3')
             st.info("The extracted text above is ready for listening.")
        else:
             st.info("Text extracted. Press 'Read Text Aloud (TTS)' to generate audio.")

    else:
        st.info("Press 'Extract Text (EasyOCR)' to begin the analysis.")


else:
    st.info("No image file has been uploaded yet. Upload one to begin the OCR process.")


# Footer 
st.sidebar.markdown("### Application Information")
st.sidebar.info("This application uses Streamlit for UI, EasyOCR for vision, and gTTS for speech generation.")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit and 📖")
