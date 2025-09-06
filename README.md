
# Smart Resume Reviewer

An **AI-powered Resume Reviewer** that helps job seekers improve their resumes by analyzing them against target job roles and job descriptions. The app highlights strengths, identifies gaps, and generates an improved version of the resume in PDF format.

🚀 Built with **Streamlit**, **Python**, and **OpenAI**.

---

## ✨ Features
- 📂 Upload your resume (PDF or text).
- 🎯 Enter target job role and optional Job Description (JD).
- 🤖 AI-powered feedback:
  - Strengths of your resume
  - Keyword gaps
  - Duplicate/vague phrases
  - Section-wise feedback
- 📑 Generate an **improved resume** in PDF format.
- 🌐 Deployed on **Streamlit Cloud**.

---

## 🛠 Tech Stack
- [Streamlit](https://streamlit.io/) – Web UI
- [Python](https://www.python.org/) – Backend
- [OpenAI API](https://platform.openai.com/) – AI-powered analysis
- [PyPDF2](https://pypi.org/project/pypdf2/) – PDF text extraction
- [FPDF](https://pypi.org/project/fpdf/) – PDF generation

---

## 🚀 Deployment
The app can be deployed easily on **Streamlit Community Cloud**.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/smart-resume-reviewer.git
   cd smart-resume-reviewer
2.Install dependencies:
pip install -r requirements.txt

3.Run the app locally:
streamlit run app.py

---

💡 Future Improvements

>Support for multiple resume formats (Word, Google Docs)
>Resume scoring based on target JD
>Interactive charts for skill gaps and keyword analysis
