import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import re
import requests

# Load environment variables
load_dotenv()

def clean_text(text):
    text = re.sub(r'<[^>]*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\s{2,}', ' ', text)  # Remove excessive spaces
    return text.strip()

class Chain:
    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not set in environment variables.")
        try:
            self.llm = ChatGroq(
                temperature=0,
                groq_api_key=groq_key,
                model_name="deepseek-r1-distill-llama-70b"
            )
        except Exception as e:
            st.error(f"Failed to initialize LLM: {e}")
            raise

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        try:
            res = chain_extract.invoke(input={"page_data": cleaned_text})
            
            
            json_parser = JsonOutputParser()
            extracted_jobs = json_parser.parse(res.content)
            
            

            # Ensure the result is a list of jobs
            if isinstance(extracted_jobs, list):
                # Validate if each job has all required keys
                for job in extracted_jobs:
                    if not all(key in job for key in ['role', 'experience', 'skills', 'description']):
                        st.error("Error: One or more jobs are missing required keys.")
                        return []
                return extracted_jobs
            else:
                st.error("Error: Extracted jobs are not in a list format.")
                return []
        except OutputParserException as e:
            st.error("Output parser error: Context too big.")
            raise
        except Exception as e:
            st.error(f"Error during job extraction: {e}")
            raise

    def write_mail(self, job, links, user_name, user_about):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are {user_name}. {user_about}
            Your job is to write a cold email to the client regarding the job mentioned above, describing how you can contribute to fulfilling their needs.
            Also, add the most relevant ones from the following links to showcase portfolio: {link_list}
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        try:
            res = chain_email.invoke({
                "job_description": str(job),
                "link_list": links,
                "user_name": user_name,
                "user_about": user_about
            })
            return res.content
        except Exception as e:
            st.error(f"Error during email generation: {e}")
            raise

class Portfolio:
    def __init__(self):
        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = []

    def add_to_portfolio(self, skills, links):
        if skills and links:
            st.session_state['portfolio'].append({"skills": skills, "links": links})

    def query_links(self, required_skills):
        if not required_skills:
            return []
        matched_links = []
        for entry in st.session_state['portfolio']:
            portfolio_skills = entry['skills']
            if any(skill in portfolio_skills for skill in required_skills):
                matched_links.append(entry['links'])
        return matched_links[:2]

def fallback_scrape(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return clean_text(response.text)
    except Exception as e:
        st.error(f"Fallback scraping failed: {e}")
        return ""

def create_streamlit_app(llm, portfolio):
    st.set_page_config(page_title="Cold Email Generator", layout="wide")
    st.title("Cold Email Generator")
    st.write("Generate personalized cold emails for job applications.")

    user_name = st.text_input("Your Name")
    user_about = st.text_area("About You")
    url_input = st.text_input("Job Posting URL")
    skills_input = st.text_area("Your Skills (comma-separated)")
    links_input = st.text_area("Portfolio Links (comma-separated)")

    if st.button("Submit"):
        try:
            skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]
            links_list = [l.strip() for l in links_input.split(",") if l.strip()]
            portfolio.add_to_portfolio(skills_list, links_list)

            try:
                loader = WebBaseLoader([url_input])
                content = loader.load().pop().page_content
            except Exception:
                st.warning("WebBaseLoader failed, using fallback method.")
                content = fallback_scrape(url_input)

            if not content.strip():
                st.warning("No content was extracted or the content is empty.")
                return

            cleaned = clean_text(content)
            jobs = llm.extract_jobs(cleaned)

            if jobs:  # Ensure jobs list is not empty
                email_output = ""
                for job in jobs:
                    job_skills = job.get('skills', [])
                    matched_links = portfolio.query_links(job_skills)
                    email = llm.write_mail(job, matched_links, user_name, user_about)
                    email_output = email  # Only store the last generated email
                if email_output:
                    st.code(email_output)  # Only show the final email
            else:
                st.warning("No jobs found in the extracted data.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    try:
        chain = Chain()
        portfolio = Portfolio()
        create_streamlit_app(chain, portfolio)
    except Exception as e:
        st.error(f"App failed to start: {e}")
