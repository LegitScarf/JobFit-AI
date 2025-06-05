import streamlit as st
import requests
import fitz  # PyMuPDF
import os
import time
from openai import OpenAI
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="JobFit AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .score-excellent {
        color: #28a745;
        font-weight: bold;
        font-size: 2rem;
    }
    
    .score-good {
        color: #17a2b8;
        font-weight: bold;
        font-size: 2rem;
    }
    
    .score-average {
        color: #ffc107;
        font-weight: bold;
        font-size: 2rem;
    }
    
    .score-poor {
        color: #dc3545;
        font-weight: bold;
        font-size: 2rem;
    }
    
    .tips-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client with proper API key handling
@st.cache_resource
def init_openai_client():
    try:
        # Get API key from secrets or environment
        api_key = None
        
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        # Fallback to environment variable
        elif "OPENAI_API_KEY" in os.environ:
            api_key = os.environ["OPENAI_API_KEY"]
        
        if not api_key:
            st.error("❌ OpenAI API key not found. Please add it to Streamlit secrets or environment variables.")
            st.info("💡 Add your API key in Streamlit Cloud: Settings → Secrets → OPENAI_API_KEY = \"your-key-here\"")
            return None
        
        # Clean the API key (remove any whitespace, newlines, etc.)
        api_key = api_key.strip().replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Validate API key format
        if not api_key.startswith("sk-"):
            st.error("❌ Invalid API key format. OpenAI keys should start with 'sk-'")
            st.info(f"🔍 Your key starts with: {api_key[:10]}...")
            return None
        
        # Test the API key length (OpenAI keys are typically long)
        if len(api_key) < 40:
            st.error("❌ API key appears to be too short. Please check if it's complete.")
            return None
        
        return OpenAI(api_key=api_key)
        
    except Exception as e:
        st.error(f"❌ Error initializing OpenAI client: {str(e)}")
        return None

# Headers for downloading the PDF
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# PDF Reader Class
class PDFReader:
    def __init__(self, url, save_path="Resume.pdf"):
        self.url = self.convert_google_drive_url(url)
        self.save_path = save_path
        response = requests.get(self.url, headers=headers)
        if response.status_code == 200:
            with open(self.save_path, "wb") as file:
                file.write(response.content)
        else:
            raise Exception(f"Failed to download PDF. Status code: {response.status_code}")
    
    def convert_google_drive_url(self, url):
        if "drive.google.com" in url:
            if "/d/" in url:
                file_id = url.split("/d/")[1].split("/")[0]
                return f"https://drive.google.com/uc?export=download&id={file_id}"
        return url
    
    def get_file_path(self):
        return self.save_path

# Extract PDF Text
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Enhanced prompts
system_prompt = """
You are an advanced AI Applicant Tracking System (ATS) designed to evaluate resumes against job descriptions.
You assess candidate suitability based on relevance of skills, experiences, and qualifications to the job role.

CRITICAL: You must respond ONLY with valid JSON format. Do not include any text before or after the JSON.

Your response must be exactly in this JSON structure:
{
    "overall_score": 7,
    "explanation": "Brief professional explanation of the evaluation",
    "matching_skills": ["skill1", "skill2", "skill3"],
    "missing_skills": ["skill1", "skill2"],
    "experience_match": "How well experience aligns with job requirements",
    "education_match": "Education relevance to the position",
    "recommendations": ["suggestion1", "suggestion2"],
    "interview_likelihood": "High",
    "key_strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["area1", "area2"]
}

Rules:
- overall_score: Must be integer 0-10
- interview_likelihood: Must be exactly "High", "Medium", or "Low"
- All arrays can be empty [] if no items found
- All strings should be concise and professional
- Return ONLY the JSON object, no other text
"""

def user_prompt_for(job_description, resume_text):
    return f"""
Analyze this resume against the job description and provide detailed evaluation:

JOB DESCRIPTION:
{job_description}

CANDIDATE'S RESUME:
{resume_text}

Please evaluate comprehensively and return your response in the specified JSON format.
"""

def messages_for(job_description, resume_text):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(job_description, resume_text)}
    ]

# Enhanced evaluation function with better JSON parsing
def evaluate_resume(job_description, url):
    try:
        # Check if OpenAI client is initialized
        openai_client = init_openai_client()
        if openai_client is None:
            return {"error": "OpenAI client not properly initialized. Please check your API key in Streamlit secrets."}
        
        pdf_reader = PDFReader(url)
        file_path = pdf_reader.get_file_path()
        resume_text = extract_text_from_pdf(file_path)
        
        if not resume_text.strip():
            raise Exception("Could not extract text from PDF. Please ensure the PDF is readable.")
        
        messages = messages_for(job_description, resume_text)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,  # Lower temperature for more consistent output
            max_tokens=1500
        )
        
        # Clean up the PDF file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        result_text = response.choices[0].message.content
        
        # Clean the response text
        result_text = result_text.strip()
        
        # Remove any markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()
        elif result_text.startswith("```"):
            result_text = result_text.replace("```", "").strip()
        
        # Try to parse JSON
        try:
            parsed_result = json.loads(result_text)
            
            # Validate required fields and provide defaults
            validated_result = {
                "overall_score": parsed_result.get("overall_score", 0),
                "explanation": parsed_result.get("explanation", "Resume evaluated successfully"),
                "matching_skills": parsed_result.get("matching_skills", []),
                "missing_skills": parsed_result.get("missing_skills", []),
                "experience_match": parsed_result.get("experience_match", "Experience evaluation completed"),
                "education_match": parsed_result.get("education_match", "Education evaluation completed"),
                "recommendations": parsed_result.get("recommendations", []),
                "interview_likelihood": parsed_result.get("interview_likelihood", "Medium"),
                "key_strengths": parsed_result.get("key_strengths", []),
                "areas_for_improvement": parsed_result.get("areas_for_improvement", [])
            }
            
            return validated_result
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, create a structured response from the text
            st.warning("⚠️ Received non-JSON response, parsing as text...")
            
            # Extract score using regex if possible
            import re
            score_match = re.search(r'score.*?(\d+)', result_text.lower())
            score = int(score_match.group(1)) if score_match else 5
            
            return {
                "overall_score": min(10, max(0, score)),  # Ensure score is between 0-10
                "explanation": result_text[:500] + "..." if len(result_text) > 500 else result_text,
                "matching_skills": [],
                "missing_skills": [],
                "experience_match": "Please see explanation for details",
                "education_match": "Please see explanation for details", 
                "recommendations": ["Review the detailed explanation for specific recommendations"],
                "interview_likelihood": "Medium",
                "key_strengths": [],
                "areas_for_improvement": [],
                "note": "Response was parsed from text format"
            }
            
    except Exception as e:
        return {"error": str(e)}

# Score visualization function
def display_score(score):
    if score >= 8:
        return "score-excellent", "🟢 Excellent Match"
    elif score >= 6:
        return "score-good", "🔵 Good Match"
    elif score >= 4:
        return "score-average", "🟡 Average Match"
    else:
        return "score-poor", "🔴 Poor Match"

# Main UI
def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 AI-Powered ATS Scanner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent Resume Evaluation System powered by OpenAI</p>', unsafe_allow_html=True)
    
    # Check API key status at startup
    openai_client = init_openai_client()
    if openai_client is None:
        st.stop()  # Stop execution if no valid API key
    
    # Sidebar for settings and tips
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Paste the job description** in the text area
        2. **Add a public Google Drive link** to the resume PDF
        3. **Click 'Analyze Resume'** to get detailed evaluation
        
        💡 **Pro Tips:**
        - Make sure your Google Drive link is publicly accessible
        - PDF should be text-readable (not scanned image)
        - More detailed job descriptions yield better analysis
        """)
        
        # API Connection Test
        st.header("🔧 API Status")
        if st.button("🧪 Test OpenAI Connection"):
            try:
                test_response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                st.success("✅ OpenAI API connection successful!")
            except Exception as e:
                st.error(f"❌ API test failed: {str(e)}")
        
        st.header("📊 Settings") 
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Standard", "Detailed", "Comprehensive"],
            index=1
        )
        
        st.header("📈 Recent Evaluations")
        if 'evaluation_history' not in st.session_state:
            st.session_state.evaluation_history = []
        
        if st.session_state.evaluation_history:
            for i, eval_item in enumerate(st.session_state.evaluation_history[-3:]):
                with st.expander(f"Evaluation {len(st.session_state.evaluation_history)-i}"):
                    st.write(f"**Score:** {eval_item.get('overall_score', 'N/A')}/10")
                    st.write(f"**Date:** {eval_item.get('timestamp', 'N/A')}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Paste the complete job description here...",
            height=250,
            placeholder="Enter the job title, responsibilities, required skills, qualifications, and any other relevant details..."
        )
        
        st.subheader("📎 Resume Upload")
        resume_url = st.text_input(
            "Public Google Drive Resume Link",
            placeholder="https://drive.google.com/file/d/your-file-id/view?usp=sharing",
            help="Make sure your Google Drive link is set to 'Anyone with the link can view'"
        )
        
        # Validation helpers
        if resume_url and "drive.google.com" not in resume_url:
            st.warning("⚠️ Please ensure you're using a Google Drive link")
        
        if job_description and len(job_description.split()) < 20:
            st.info("💡 Consider adding more details to the job description for better analysis")
    
    with col2:
        st.subheader("🎯 Quick Stats")
        
        # Sample metrics (these would be calculated after evaluation)
        if 'last_evaluation' in st.session_state:
            last_eval = st.session_state.last_evaluation
            if not last_eval.get('error'):
                score = last_eval.get('overall_score', 0)
                score_class, score_label = display_score(score)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Overall Score</h3>
                    <div class="{score_class}">{score}/10</div>
                    <p>{score_label}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if last_eval.get('interview_likelihood'):
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Interview Likelihood</h3>
                        <h2>{last_eval.get('interview_likelihood', 'N/A')}</h2>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Analysis button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze Resume", use_container_width=True)
    
    # Analysis results
    if analyze_button:
        if not job_description or not resume_url:
            st.error("❌ Please provide both job description and resume link.")
        else:
            with st.spinner("🔍 Analyzing resume... This may take a moment..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result = evaluate_resume(job_description, resume_url)
                st.session_state.last_evaluation = result
                
                # Add to history
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.evaluation_history.append(result)
                
                progress_bar.empty()
            
            if result.get('error'):
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success("✅ Analysis Complete!")
                
                # Display results in organized sections
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    score = result.get('overall_score', 0)
                    score_class, score_label = display_score(score)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin-top: 0;">Overall Score</h3>
                        <div class="{score_class}">{score}/10</div>
                        <p>{score_label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if result.get('interview_likelihood'):
                        likelihood = result.get('interview_likelihood', 'N/A')
                        color = "#28a745" if likelihood == "High" else "#ffc107" if likelihood == "Medium" else "#dc3545"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin-top: 0;">Interview Likelihood</h3>
                            <h2 style="color: {color}; margin: 0.5rem 0;">{likelihood}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col3:
                    matching_skills = result.get('matching_skills', [])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin-top: 0;">Matching Skills</h3>
                        <h2 style="color: #17a2b8; margin: 0.5rem 0;">{len(matching_skills)}</h2>
                        <p>Skills Found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed breakdown
                st.subheader("📊 Detailed Analysis")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "✅ Strengths", "❌ Gaps", "💡 Recommendations"])
                
                with tab1:
                    st.write("**Evaluation Summary:**")
                    st.write(result.get('explanation', 'No explanation available'))
                    
                    if result.get('experience_match'):
                        st.write("**Experience Match:**")
                        st.write(result.get('experience_match'))
                    
                    if result.get('education_match'):
                        st.write("**Education Match:**")
                        st.write(result.get('education_match'))
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result.get('matching_skills'):
                            st.write("**🎯 Matching Skills:**")
                            for skill in result.get('matching_skills', []):
                                st.write(f"• {skill}")
                    
                    with col2:
                        if result.get('key_strengths'):
                            st.write("**💪 Key Strengths:**")
                            for strength in result.get('key_strengths', []):
                                st.write(f"• {strength}")
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result.get('missing_skills'):
                            st.write("**❌ Missing Skills:**")
                            for skill in result.get('missing_skills', []):
                                st.write(f"• {skill}")
                    
                    with col2:
                        if result.get('areas_for_improvement'):
                            st.write("**📈 Areas for Improvement:**")
                            for area in result.get('areas_for_improvement', []):
                                st.write(f"• {area}")
                
                with tab4:
                    if result.get('recommendations'):
                        st.write("**💡 Recommendations for Candidate:**")
                        for i, rec in enumerate(result.get('recommendations', []), 1):
                            st.markdown(f"""
                            <div class="tips-box">
                                <strong>{i}.</strong> {rec}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Download results option
                st.subheader("📥 Export Results")
                
                # Create downloadable report
                report_data = {
                    "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "overall_score": result.get('overall_score'),
                    "analysis": result
                }
                
                st.download_button(
                    label="📄 Download Evaluation Report (JSON)",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"ats_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
