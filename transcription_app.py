import streamlit as st
import os
import tempfile
import whisper
import time
from zipfile import ZipFile
from io import BytesIO

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
    st.title("ACLU Video Transcription App")

    # Use session state to store transcripts
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = {}

    uploaded_files = st.file_uploader("Choose video files", type=["mp4", "mkv", "avi", "mov"], accept_multiple_files=True)

    if uploaded_files is not None:
        if st.button("Start Transcription"):
            model = load_whisper_model()

            total_start_time = time.time()

            # Create a placeholder for overall progress
            overall_progress_bar = st.progress(0)
            overall_status_text = st.empty()

            for file_index, uploaded_file in enumerate(uploaded_files):
                # Only transcribe if the file hasn't been transcribed yet
                if uploaded_file.name not in st.session_state.transcripts:
                    st.write(f"Transcribing '{uploaded_file.name}'...")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    start_time = time.time()
                    segments = transcribe_video(uploaded_file, model)

                    transcript = ""
                    for i, segment in enumerate(segments):
                        transcript += f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n"
                        progress = (i + 1) / len(segments)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing segment {i+1}/{len(segments)}")

                    end_time = time.time()
                    processing_time = end_time - start_time

                    # Store the transcript in session state
                    st.session_state.transcripts[uploaded_file.name] = transcript

                    # Update overall progress
                    overall_progress = (file_index + 1) / len(uploaded_files)
                    overall_progress_bar.progress(overall_progress)
                    overall_status_text.text(f"Completed {file_index + 1}/{len(uploaded_files)} files.")

            total_end_time = time.time()

    # If transcripts are available, display and provide download option
    if st.session_state.transcripts:
        for filename, transcript in st.session_state.transcripts.items():
            with st.expander(f"Transcript for '{filename}'"):
                st.text_area(f"Transcript for {filename}", transcript, height=300)

        # Create a ZIP file of all transcripts
        if st.button("Download All Transcripts as ZIP"):
            with BytesIO() as zip_buffer:
                with ZipFile(zip_buffer, "w") as zip_file:
                    for filename, transcript in st.session_state.transcripts.items():
                        transcript_filename = f"{filename}_transcript.txt"
                        zip_file.writestr(transcript_filename, transcript)

                zip_buffer.seek(0)

                # Trigger the download directly
                st.download_button(
                    label="Download All Transcripts",
                    data=zip_buffer,
                    file_name="transcripts.zip",
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()

