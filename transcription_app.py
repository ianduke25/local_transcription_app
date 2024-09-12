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
    st.title("Video Transcription App")
    st.write("Upload multiple video files to generate transcripts.")

    uploaded_files = st.file_uploader("Choose video files", type=["mp4", "mkv", "avi", "mov"], accept_multiple_files=True)

    if uploaded_files is not None:
        model = load_whisper_model()

        transcripts = {}
        total_start_time = time.time()

        # Create a placeholder for overall progress
        overall_progress_bar = st.progress(0)
        overall_status_text = st.empty()

        for file_index, uploaded_file in enumerate(uploaded_files):
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

            transcripts[uploaded_file.name] = transcript
            st.write(f"Transcription for '{uploaded_file.name}' completed in {processing_time:.2f} seconds.")
            
            # Display transcript in a dropdown for each file
            with st.expander(f"Transcript for '{uploaded_file.name}'"):
                st.text_area(f"Transcript for {uploaded_file.name}", transcript, height=300)

            # Update overall progress
            overall_progress = (file_index + 1) / len(uploaded_files)
            overall_progress_bar.progress(overall_progress)
            overall_status_text.text(f"Completed {file_index + 1}/{len(uploaded_files)} files.")

        total_end_time = time.time()
        st.write(f"All transcriptions completed in {total_end_time - total_start_time:.2f} seconds.")

        # Create a ZIP file of all transcripts
        if st.button("Download All Transcripts as ZIP"):
            with BytesIO() as zip_buffer:
                with ZipFile(zip_buffer, "w") as zip_file:
                    for filename, transcript in transcripts.items():
                        transcript_filename = f"{filename}_transcript.txt"
                        zip_file.writestr(transcript_filename, transcript)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="Download ZIP",
                    data=zip_buffer,
                    file_name="transcripts.zip",
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()
