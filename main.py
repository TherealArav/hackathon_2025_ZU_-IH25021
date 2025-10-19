import streamlit as st
from PIL import Image
import io
import time 
import numpy as np
import easyocr 
from gtts import gTTS 
from io import BytesIO 
from googletrans import Translator
# NEW IMPORT: Import the CV logic from the microservice file
from cv_logic import classify_road_sign 


# --- Helper Functions ---

@st.cache_resource
def get_ocr_reader():
    """Initializes and caches the EasyOCR reader model (English language)."""
    return easyocr.Reader(['en'])


def extract_raw_text(uploaded_file):
    """
    Performs OCR and returns only the raw extracted text string.
    """
    reader = get_ocr_reader()
    image_pil = Image.open(uploaded_file)
    image_np = np.array(image_pil)
    results = reader.readtext(image_np, detail=0) 
    return "\n".join(results) 


def generate_audio(text_to_read, language='en'): 
    """
    Generates audio bytes from text using gTTS for a specific language.
    """
    if not text_to_read:
        return None
        
    st.toast(f"Generating {language.upper()} speech...", icon="🔊")
    try:
        tts = gTTS(text=text_to_read, lang=language) 
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.read()
    except Exception as e:
        st.error(f"Error generating TTS audio in {language.upper()}: {e}")
        return None

def translate_text(text, dest_lang='ar'): 
    """Translates text using googletrans."""
    if not text:
        return ""
    try:
        # Use the global translator instance
        translation = translator.translate(text, dest=dest_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return "Translation failed."


# --- OCR, CV, & Translation Pipeline (Orchestration) ---

def run_ocr_pipeline(uploaded_file):
    """
    Orchestrates the OCR extraction, translation, CV classification, state updates, and error handling.
    """

    st.session_state['is_processing'] = True
    st.session_state['ocr_result'] = None
    st.session_state['raw_ocr_text'] = None 
    st.session_state['arabic_text'] = None 
    st.session_state['audio_bytes'] = None 
    st.session_state['cv_result'] = None       # Clear CV Result
    st.session_state['cv_image_np'] = None     # Clear CV Image NP array
    st.toast("Starting Image Analysis (OCR, Translation, and CV)...", icon="🔍")

    try:
        # Get image as PIL object and NumPy array for different libraries
        # FIX: Explicitly convert PIL image to 'RGB' to strip the 4th (Alpha) channel
        image_pil = Image.open(uploaded_file).convert('RGB')
        # Use the RGB NumPy array which is the standard output of Image.open/np.array
        image_np_rgb = np.array(image_pil)
        
        # 1. OCR Extraction (English)
        with st.spinner('Running OCR...'):
            # NOTE: We keep the old extract_raw_text as it uses the same Image.open
            # but EasyOCR is more tolerant of 4-channel images than PyTorch/YOLO.
            # To ensure consistency, we should ideally use the converted image_pil here,
            # but for simplicity, we'll rely on EasyOCR's robustness for now.
            extracted_text = extract_raw_text(uploaded_file)
        
        # 2. Translation (Arabic)
        with st.spinner('Translating Text...'):
            arabic_translation = translate_text(extracted_text, dest_lang='ar')

        # 3. Road Sign Classification (CV) - Pass the corrected 3-channel image NP array
        with st.spinner('Running Computer Vision Classification...'):
            classification_text, classified_image_np = classify_road_sign(image_np_rgb)
            
        # 4. Format English Output for Display
        final_english_output = f"""
**Source File:** `{uploaded_file.name}`

---

**OCR Content:**

```
{extracted_text}
```

Processed successfully on: {time.ctime()}
"""
        # 5. Update session states
        st.session_state['raw_ocr_text'] = extracted_text 
        st.session_state['arabic_text'] = arabic_translation 
        st.session_state['ocr_result'] = final_english_output 
        st.session_state['cv_result'] = classification_text      # Set CV Result
        st.session_state['cv_image_np'] = classified_image_np    # Set CV Image NP array
        
        st.toast("All analyses completed!", icon="✅")

    except Exception as e:
        st.error(f"An error occurred during pipeline execution: {e}")
        st.session_state['ocr_result'] = f"Error: Could not process. {e}"

    finally:
        st.session_state['is_processing'] = False


# --- Main Application Logic ---

st.set_page_config(
    page_title="Fresh-Gens",
    page_icon="⚡",
    layout="wide"
)

# --- Callbacks for TTS Button ---

def tts_callback():
    """
    Callback function for the TTS button. Uses the selected language and corresponding text.
    """
    selected_lang = st.session_state.get('tts_language_select') 
    
    if selected_lang == 'English':
        raw_text = st.session_state.get('raw_ocr_text')
        lang_code = 'en'
    elif selected_lang == 'Arabic':
        raw_text = st.session_state.get('arabic_text')
        lang_code = 'ar'
    else:
        st.error("Invalid language selected for TTS.")
        return

    # Generate and store audio only if text is available
    if raw_text and raw_text not in ["Translation failed.", ""]:
        audio_data = generate_audio(raw_text, language=lang_code)
        if audio_data:
            st.session_state['audio_bytes'] = audio_data
    else:
        st.error(f"Cannot read text: No valid {selected_lang} text found.")


# Initialize Translator instance globally
translator = Translator() 


# --- State Initialization ---

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'uploaded_file_name' not in st.session_state: # Use name to detect a change in file content
    st.session_state['uploaded_file_name'] = None 
if 'ocr_result' not in st.session_state:
    st.session_state['ocr_result'] = None
if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False
if 'audio_bytes' not in st.session_state:
    st.session_state['audio_bytes'] = None
if 'raw_ocr_text' not in st.session_state:
    st.session_state['raw_ocr_text'] = None
if 'arabic_text' not in st.session_state:
    st.session_state['arabic_text'] = None
if 'tts_language_select' not in st.session_state: 
    st.session_state['tts_language_select'] = 'English'
if 'cv_result' not in st.session_state:         # NEW STATE INIT
    st.session_state['cv_result'] = None
if 'cv_image_np' not in st.session_state:       # NEW STATE INIT
    st.session_state['cv_image_np'] = None


# --- UI Components ---

st.title("🖼️ Fresh-Gens: Multi-Modal Image Analyzer")
st.subheader("Extract Text, Translate, Speak, and Classify Road Signs using Computer Vision.")

st.markdown("---")

# File Uploader 
uploaded_file = st.file_uploader(
    "Choose an image file to upload:",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    key="image_uploader"
)

# Handle Upload and State Management (Check for new upload/clear)
if uploaded_file is not None and uploaded_file != st.session_state.get('uploaded_file'):
    # Store the file object/name to detect changes
    st.session_state['uploaded_file'] = uploaded_file
    
    # Clear previous analysis states
    st.session_state['ocr_result'] = None
    st.session_state['raw_ocr_text'] = None
    st.session_state['arabic_text'] = None 
    st.session_state['audio_bytes'] = None
    st.session_state['cv_result'] = None
    st.session_state['cv_image_np'] = None
    st.toast(f"File uploaded successfully: {uploaded_file.name}", icon="✅")

elif uploaded_file is None and st.session_state.get('uploaded_file') is not None:
    # Clear session if user removes the file
    st.session_state['uploaded_file'] = None
    st.session_state['uploaded_file_name'] = None
    st.session_state['ocr_result'] = None
    st.session_state['raw_ocr_text'] = None 
    st.session_state['arabic_text'] = None
    st.session_state['audio_bytes'] = None 
    st.session_state['cv_result'] = None
    st.session_state['cv_image_np'] = None
    st.toast("File cleared from session.", icon="🗑️")


# Display 
file_to_display = st.session_state.get('uploaded_file')

if file_to_display:
    st.success(f"File '{file_to_display.name}' is ready for processing.")
    
    img_col, meta_col = st.columns([2, 1])

    with img_col:
        image_data = Image.open(file_to_display)
        st.image(image_data, caption="Original Image", use_container_width=True) 

    with meta_col:
        st.subheader("Controls")
        
        # OCR/CV Button
        if st.button("Run Full Analysis", type="primary", disabled=st.session_state['is_processing']):
            run_ocr_pipeline(file_to_display)
        
        # TTS Language Selector
        tts_disabled = st.session_state['is_processing'] or not st.session_state['raw_ocr_text']
        
        st.selectbox(
            "Select TTS Language:",
            ('English', 'Arabic'),
            key='tts_language_select', # This key is used in the tts_callback
            disabled=tts_disabled,
        )

        # TTS Button
        if st.button("Read Text Aloud (TTS)", disabled=tts_disabled, on_click=tts_callback):
            pass 
            
        st.markdown("---")
        st.subheader("File Details")
        st.markdown(f"**Name:** `{file_to_display.name}`")
        st.markdown(f"**Type:** `{file_to_display.type}`")
        # Ensure 'file_to_display.size' is available before using it
        if hasattr(file_to_display, 'size'):
            st.markdown(f"**Size:** `{file_to_display.size / 1024:.2f} KB`")

    # Output Display Section
    st.markdown("## Analysis Results")

    if st.session_state['is_processing']:
        st.info("Processing image and performing translation...")
        with st.spinner('Running OCR, Translation, and CV...'):
            time.sleep(0.1) 
            
    elif st.session_state['ocr_result'] or st.session_state['cv_result']:
        # Use tabs for clean multi-lingual and multi-modal display
        tab1, tab2, tab3 = st.tabs(["Road Sign Classification", "English Output (OCR)", "Arabic Translation (العربية)"])
        
        # --- TAB 1: CV Classification ---
        with tab1:
            st.subheader("Computer Vision Road Sign Analysis")
            
            if st.session_state['cv_result']:
                st.markdown(f"**Classification Result:** **{st.session_state['cv_result']}**")
                
                # Display the processed image if available
                classified_image_np = st.session_state.get('cv_image_np')
                if classified_image_np is not None and isinstance(classified_image_np, np.ndarray):
                    # classified_image_np is RGB (as returned by cv.py) and ready for st.image
                    st.image(classified_image_np, caption="CV Result (Bounding Box)", use_container_width=True)
                    st.caption("A bounding box has been drawn around the detected sign based on color and shape heuristics.")
                else:
                    st.info("No processed image available (error during CV or no sign detected).")
            else:
                st.info("Run the 'Run Full Analysis' button to start classification.")


        # --- TAB 2: OCR English ---
        with tab2:
            st.subheader("English Extracted Text")
            st.markdown(st.session_state.get('ocr_result', "No OCR results yet.")) 
        
        # --- TAB 3: Arabic Translation ---
        with tab3:
            st.subheader("Translated Arabic Text")
            arabic_text = st.session_state.get('arabic_text', "")
            if arabic_text and arabic_text != "Translation failed.":
                # Ensure Arabic text displays right-to-left
                st.markdown(f"<div style='direction: rtl; text-align: right;'>{arabic_text}</div>", unsafe_allow_html=True)
            else:
                st.info("Translation not available or failed.")
            
        
        # Audio Player Section (outside tabs for consistency)
        st.markdown("---")
        if st.session_state['audio_bytes']:
            st.subheader("Audio Playback")
            st.audio(st.session_state['audio_bytes'], format='audio/mp3')
            st.info(f"The selected text ({st.session_state['tts_language_select']}) is ready for listening.")
        else:
            st.info("Text extracted and translated. Select a language and press 'Read Text Aloud (TTS)' to generate audio.")

    else:
        st.info("Press 'Run Full Analysis' to begin the full multi-modal analysis.")


else:
    st.info("No image file has been uploaded yet. Upload one to begin the OCR, translation, and CV process.")


# Footer 
st.sidebar.markdown("### Application Information")
st.sidebar.info("This application uses Streamlit for UI, EasyOCR for text extraction, **googletrans** for translation, gTTS for speech, and **OpenCV** for computer vision classification.")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit and 📖")
