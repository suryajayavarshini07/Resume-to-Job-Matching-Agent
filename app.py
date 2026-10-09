
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from google.genai import types
import io
import os
import time

load_dotenv()

MODEL_NAME = "gemini-3.5-flash"

st.set_page_config(
    page_title="Resume-to-Job Matching Agent",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume-to-Job Matching Agent")
st.write("Compare your resume with a job description.")

api_key = os.getenv("GEMINI_API_KEY")


def get_gemini_client():
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Check your .env file."
        )

    return genai.Client(api_key=api_key)


def generate_with_retry(client, prompt, attempts=3):
    """Retry temporary server errors with increasing delays."""

    for attempt in range(attempts):
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

        except Exception as error:
            message = str(error).upper()

            temporary_error = any(
                code in message
                for code in [
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    "500",
                    "INTERNAL",
                    "502",
                    "504",
                    "DEADLINE_EXCEEDED"
                ]
            )

            if not temporary_error or attempt == attempts - 1:
                raise

            wait_seconds = 2 ** (attempt + 1)
            time.sleep(wait_seconds)


def show_error(error, task):
    message = str(error)
    upper_message = message.upper()

    if "503" in upper_message or "UNAVAILABLE" in upper_message:
        st.error(
            "Gemini server is temporarily unavailable. "
            "Please try again later."
        )
    elif "429" in upper_message or "RESOURCE_EXHAUSTED" in upper_message:
        st.error(
            "API quota or rate limit reached. "
            "Check your Gemini API usage and limits."
        )
    elif "404" in upper_message or "NOT_FOUND" in upper_message:
        st.error(
            f"Model '{MODEL_NAME}' was not found or is unavailable "
            "for this API. Check the supported model list."
        )
    elif "401" in upper_message or "403" in upper_message:
        st.error(
            "API authentication failed. Check your API key and permissions."
        )
    else:
        st.error(f"{task} failed: {message}")


# STEP 1: Upload resume
st.subheader("Step 1: Upload Your Resume")

uploaded_file = st.file_uploader(
    "Choose your resume PDF",
    type=["pdf"]
)

resume_text = ""

if uploaded_file is not None:
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))

        resume_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()

        if resume_text:
            st.success("Resume text extracted successfully!")

            with st.expander("View extracted resume text"):
                st.text(resume_text)
        else:
            st.warning(
                "No readable text found. "
                "This may be a scanned PDF."
            )

    except Exception as error:
        show_error(error, "Resume reading")


# STEP 2: Job description
st.subheader("Step 2: Enter Job Description")

job_description = st.text_area(
    "Paste the job description",
    height=220,
    placeholder=(
        "Example: Data Analyst Fresher. "
        "Required skills: Python, SQL, Excel, Power BI..."
    )
)


# STEP 3: Test connection
st.subheader("Step 3: Test Gemini API Connection")

if st.button("Test Gemini Connection"):
    try:
        client = get_gemini_client()

        with st.spinner("Testing connection..."):
            response = generate_with_retry(
                client,
                "Reply with exactly: Gemini connection successful"
            )

        if response.text:
            st.success("Gemini API connected successfully!")
            st.write(response.text)
        else:
            st.warning("Gemini returned an empty response.")

    except Exception as error:
        show_error(error, "Connection test")


# STEP 4: Analyze resume
st.subheader("Step 4: Analyze Resume")

if st.button("Analyze Resume", type="primary"):

    if uploaded_file is None:
        st.warning("Please upload a resume PDF.")

    elif not resume_text:
        st.warning("No readable resume text was found.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    elif not api_key:
        st.error("GEMINI_API_KEY not found in your .env file.")

    else:
        prompt = f"""
You are a Resume-to-Job Matching Agent.

Compare the resume with the job description.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Create these report sections:

## 1. Estimated Match Percentage
Estimate a score from 0 to 100 based on relevant requirements
and evidence in the resume. Explain the scoring briefly.
This is not a hiring probability.

## 2. Matching Skills
List skills explicitly supported by the resume and required by the job.

## 3. Skills Not Demonstrated
List requirements not clearly shown in the resume.
Do not assume the candidate lacks a skill just because it is not listed.

## 4. Relevant Projects and Education
Use only information supported by the resume.

## 5. Resume Improvement Suggestions
Give practical suggestions for this specific job.

## 6. Recommended Learning Steps
Suggest useful skills or projects to learn next.

Use simple English and bullet points.
Never invent skills, qualifications, projects, or experience.
Do not repeat personal phone numbers or email addresses.
"""

        try:
            client = get_gemini_client()

            with st.spinner(
                "Analyzing resume. Temporary errors will be retried..."
            ):
                response = generate_with_retry(client, prompt)

            if response.text:
                st.success("Resume analysis completed!")
                st.subheader("Resume Analysis Report")
                st.markdown(response.text)

                st.download_button(
                    "Download Analysis Report",
                    data=response.text,
                    file_name="resume_analysis_report.txt",
                    mime="text/plain"
                )
            else:
                st.warning("Gemini returned no analysis.")

        except Exception as error:
            show_error(error, "Resume analysis")
            st.info(
                "If the server remains unavailable after retries, "
                "wait and try again later."
            )