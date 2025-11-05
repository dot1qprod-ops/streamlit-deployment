import streamlit as st
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import librosa
from tempfile import NamedTemporaryFile
import os
import soundfile as sf
import torch

st.set_page_config(
    page_title="Chichewa ASR & Translation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Chichewa Speech Recognition and Translation")
st.caption("Automated transcription and translation system for Chichewa audio")

@st.cache_resource
def load_models():
    """Load the Whisper transcription model and NLLB translation model"""
    with st.spinner("Loading models..."):
        # Load Whisper model and processor separately
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        whisper_model_raw = WhisperForConditionalGeneration.from_pretrained("zerolat3ncy/whisper-ch-chk500")
        
        # Create pipeline with explicit processor
        whisper_model = pipeline(
            "automatic-speech-recognition",
            model=whisper_model_raw,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=-1
        )
        
        translation_model = pipeline(
            "translation", 
            model="zerolat3ncy/nllb-financial-nya-en", 
            device=-1
        )
        
    return whisper_model, translation_model

def convert_audio(audio_bytes):
    """Convert audio to 16kHz WAV format for Whisper"""
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name
    
    audio_data, sr = librosa.load(temp_path, sr=16000)
    
    with NamedTemporaryFile(delete=False, suffix=".wav") as converted_file:
        sf.write(converted_file.name, audio_data, sr, format='WAV')
        converted_path = converted_file.name
    
    os.unlink(temp_path)
    return converted_path

def transcribe(model, audio_path):
    """Transcribe Chichewa audio using Whisper"""
    try:
        # Load audio using librosa at 16kHz (Whisper's expected sampling rate)
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # Pass the audio array directly to the pipeline
        result = model(audio_array)
        
        if isinstance(result, dict):
            return result["text"].strip()
        elif isinstance(result, list) and len(result) > 0:
            return result[0]["text"].strip()
        else:
            return str(result).strip()
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

def translate(model, text):
    """Translate Chichewa text to English using NLLB"""
    if not text or text.strip() == "":
        return ""
    
    if len(text.split()) > 200:
        text = " ".join(text.split()[:200])
    
    try:
        result = model(text)
        return result[0]['translation_text']
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return ""

def calculate_accuracy(reference, hypothesis):
    """Calculate word-level accuracy between reference and hypothesis"""
    if not reference or not hypothesis:
        return None
    
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    matches = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)
    accuracy = (matches / max(len(ref_words), len(hyp_words))) * 100
    
    return accuracy

# Load models
try:
    whisper_model, translation_model = load_models()
    st.success("Models loaded successfully")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

st.divider()

# Input method selection
input_method = st.radio("Input Method", ["Upload Audio File", "Record Audio"], horizontal=True)

audio_data = None

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Select audio file", 
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a']
    )
    if uploaded_file is not None:
        st.audio(uploaded_file)
        audio_data = uploaded_file
else:
    recorded_audio = st.audio_input("Record audio")
    if recorded_audio is not None:
        st.audio(recorded_audio)
        audio_data = recorded_audio

# Optional reference text inputs
with st.expander("Reference Text (Optional)"):
    st.caption("Provide reference text to calculate accuracy metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        reference_chichewa = st.text_area(
            "Chichewa Reference", 
            placeholder="Enter correct Chichewa transcription"
        )
    
    with col2:
        reference_english = st.text_area(
            "English Reference", 
            placeholder="Enter correct English translation"
        )

# Process button
if audio_data is not None:
    if st.button("Process Audio", type="primary", use_container_width=True):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Preparing audio...")
            progress_bar.progress(10)
            
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                if hasattr(audio_data, 'read'):
                    audio_bytes = audio_data.read()
                else:
                    audio_bytes = audio_data.getvalue()
                
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            if input_method == "Record Audio":
                status_text.text("Converting audio format...")
                progress_bar.progress(20)
                converted_path = convert_audio(audio_bytes)
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                tmp_path = converted_path
            
            status_text.text("Transcribing audio...")
            progress_bar.progress(40)
            transcription = transcribe(whisper_model, tmp_path)
            
            status_text.text("Translating to English...")
            progress_bar.progress(70)
            translation = translate(translation_model, transcription)
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            st.success("Processing complete")
            st.divider()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Chichewa Transcription")
                st.text_area(
                    "Transcription Output",
                    value=transcription if transcription else "No transcription generated",
                    height=150,
                    label_visibility="collapsed",
                    key="transcription_output"
                )
                
                if reference_chichewa.strip() and transcription:
                    accuracy = calculate_accuracy(reference_chichewa, transcription)
                    if accuracy is not None:
                        st.metric("Accuracy", f"{accuracy:.1f}%")
            
            with col_b:
                st.subheader("English Translation")
                st.text_area(
                    "Translation Output",
                    value=translation if translation else "No translation generated",
                    height=150,
                    label_visibility="collapsed",
                    key="translation_output"
                )
                
                if reference_english.strip() and translation:
                    accuracy = calculate_accuracy(reference_english, translation)
                    if accuracy is not None:
                        st.metric("Accuracy", f"{accuracy:.1f}%")
            
            st.divider()
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                st.download_button(
                    "Download Transcription", 
                    transcription, 
                    "transcription.txt", 
                    "text/plain",
                    use_container_width=True
                )
            
            with col_dl2:
                st.download_button(
                    "Download Translation", 
                    translation, 
                    "translation.txt", 
                    "text/plain",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

else:
    st.info("Please upload an audio file or record audio to begin")
