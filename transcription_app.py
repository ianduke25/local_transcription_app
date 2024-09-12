import streamlit as st
import os
import tempfile
import whisper
import time

# Load the Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def transcribe_video(video_file, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        temp_video_path = temp_video.name

    try:
        result = model.transcribe(temp_video_path, verbose=False, language='English', condition_on_previous_text=False)
        return result["segments"]
    finally:
        os.unlink(temp_video_path)

def main():
    st.title("Video Transcription App")
    st.write("Upload a video file to generate a transcript.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mkv", "avi", "mov"])

    if uploaded_file is not None:
        model = load_whisper_model()

        st.write("Transcribing... This may take a while depending on the video length.")
        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        segments = transcribe_video(uploaded_file, model)

        transcript = ""
        for i, segment in enumerate(segments):
            transcript += f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n"
            progress = (i + 1) / len(segments)
            progress_bar.progress(progress)
            status_text.text(f"Processing segment {i+1}/{len(segments)}")

        end_time = time.time()
        processing_time = end_time - start_time

        st.write(f"Transcription completed in {processing_time:.2f} seconds.")
        st.text_area("Transcript", transcript, height=300)

        st.download_button(
            label="Download Transcript",
            data=transcript,
            file_name=f"{uploaded_file.name}_transcript.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()