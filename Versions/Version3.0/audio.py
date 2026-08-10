import io
import os

import numpy as np
import soundfile as sf
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline

# Load .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Page
st.set_page_config(
    page_title="FOMO Guardian AI",
    page_icon="🎙️",
    layout="centered",
)

# Load Whisper once
@st.cache_resource
def load_whisper():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small.en",
        token=HF_TOKEN,
    )

try:
    pipe = load_whisper()
except Exception as e:
    st.error("Unable to load the Whisper model.")
    st.exception(e)
    st.stop()

# UI
st.title("🎙️ FOMO Guardian AI")
st.write("Speak to FOMO Guardian and your voice will be converted into text.")

# IMPORTANT: only ONE audio_input widget
audio = st.audio_input(
    "Speak to FOMO Guardian",
    sample_rate=16000,
    key="fomo_audio_input",
)

if audio is not None:
    st.write("🎙️ Processing your voice...")

    try:
        # Read audio directly from memory.
        # This avoids passing a filename to Transformers/Whisper,
        # so FFmpeg is not required by this code.
        audio_bytes = audio.getvalue()

        audio_array, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
        )

        # Stereo -> mono
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        # English-only whisper-small.en:
        # Do NOT pass language="en" or task="transcribe".
        result = pipe(
            {
                "raw": audio_array,
                "sampling_rate": sample_rate,
            }
        )

        text = result.get("text", "").strip()

        if text:
            st.success("Speech recognized!")
            st.subheader("You said:")
            st.write(text)
        else:
            st.warning("No speech was detected. Please try again.")

    except Exception as e:
        st.error("Speech recognition failed.")
        st.exception(e)