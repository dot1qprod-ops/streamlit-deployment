import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import librosa
import torch
from tempfile import NamedTemporaryFile
import os

st.set_page_config(
    page_title="Chichewa ASR & Translation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Chichewa Speech Recognition and Translation")
st.caption("Audio → Whisper ASR → NLLB Translation → English Text")

@st.cache_resource
def load_models():
    """Load Whisper ASR and NLLB translation models"""
    with st.spinner("Loading models..."):
        # Load Whisper processor from base model
        processor = WhisperProcessor.from_pretrained(
            "openai/whisper-small",
            language="ny",
            task="transcribe"
        )
        
        # Load fine-tuned Whisper model
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "zerolat3ncy/whisper-ch-chk500"
        )
        
        # Load NLLB tokenizer and model
        nllb_tokenizer = AutoTokenizer.from_pretrained(
            "zerolat3ncy/nllb-financial-nya-en",
            src_lang="nya_Latn",
            tgt_lang="eng_Latn"
        )
        
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            "zerolat3ncy/nllb-financial-nya-en"
        )
        
    return processor, whisper_model, nllb_tokenizer, nllb_model

def transcribe_audio(processor, model, audio_path):
    """Transcribe Chichewa audio to text using Whisper"""
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        
        input_features = processor(
            audio,
            return_tensors="pt",
            sampling_rate=sr
        ).input_features
        
        with torch.no_grad():
            generated_ids = model.generate(inputs=input_features)
        
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

def translate_text(tokenizer, model, chichewa_text):
    """Translate Chichewa text to English using NLLB"""
    if not chichewa_text or chichewa_text.strip() == "":
        return ""
    
    try:
        inputs = tokenizer(
            chichewa_text,
            return_tensors='pt',
            max_length=256,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                early_stopping=True
            )
        
        translation = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return translation
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
    processor, whisper_model, nllb_tokenizer, nllb_model = load_models()
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
            
            status_text.text("Transcribing audio...")
            progress_bar.progress(40)
            transcription = transcribe_audio(processor, whisper_model, tmp_path)
            
            status_text.text("Translating to English...")
            progress_bar.progress(70)
            translation = translate_text(nllb_tokenizer, nllb_model, transcription)
            
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
