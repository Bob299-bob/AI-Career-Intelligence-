from groq import Groq
import os
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()
client=Groq(api_key=os.getenv("GROQ_API_KEY"))

#Text_extraction
import pdfplumber
def pdf_extract(data_path):
    pdf_text=""
    with pdfplumber.open(data_path) as file:
        for page in file.pages:
            page_text=page.extract_text()
            if page_text:
                pdf_text+=page_text+"\n"
    return pdf_text

#chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

#Embedding ,Vector DB Store
#import Embedding Model
from sentence_transformers import SentenceTransformer
model=SentenceTransformer(
    "all-MiniLM-L6-v2"
)

#Creating RAG SYSTEM
import faiss
def RAG(pdf_text):
  if pdf_text!="":
    pdf=splitter.split_text(pdf_text)
    pdf_embedding=model.encode(pdf).astype('float32')
    faiss.normalize_L2(pdf_embedding)
    index=faiss.IndexFlatIP(pdf_embedding.shape[1])
    index.add(pdf_embedding)
    return index,pdf
  else:
      st.error('Sorry cant help you at this time')
      st.stop()


#Creating Retrieval System
def Retrieve(query,index,pdf):
    chunks=[]
    query=model.encode([query]).astype('float32')
    faiss.normalize_L2(query)
    distance,indices=index.search(query,k=min(5,len(pdf)))
    for idx in indices[0]:
        chunks.append(pdf[idx])
    return chunks

#Analysis
def Output(query,chunks):
    context="\n".join(chunks)
    prompt=f"""
Answer only from the given context

Context:
{context}

Question:
{query}
"""
    response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{'role':'user','content':prompt}]
    )
    return response.choices[0].message.content
#skillExtract agent
def extract_skills(pdf, query):
    prompt = f"""
You are an ATS and skill analysis assistant.

Resume:
{pdf}

Target Role:
{query}

First check it is CV,Resume 
If yes then,
Return:
1. Matching Skills
2. Missing Skills
3. Skill Match Percentage

If not CV, Resume then tell him about the document and say sorry to him
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
#Resume agent
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def ATS(pdf,query):
    print(len(pdf))
    if len(pdf)>2000:
        pdf=pdf[:2000]
    one_pd=model.encode(pdf).astype('float32')
    two_q=model.encode([query]).astype('float32')
    score=cosine_similarity(one_pd,two_q)
    score=np.max(score)*100
    prompt = f"""
You are an expert Resume Screening AI.

Document:
{pdf}

User Query:
{query}

Instructions:

1. First determine whether the document is a Resume/CV.

2. If the document IS a Resume/CV:
   - State that the document is a Resume/CV.
   - Analyze the candidate's education, skills, projects, experience, and certifications.
   - Suggest the top 3 most suitable job roles.
   - Explain briefly why each role matches the candidate's profile.
   - If a Job Description is provided in the query, perform a Skill Gap Analysis and list:
       • Matching Skills
       • Missing Skills
       • Recommendations for improvement

3. If the document is NOT a Resume/CV:
   - State what type of document it appears to be.
   - Politely apologize.
   - Explain that job-role recommendation and skill-gap analysis can only be performed on resumes/CVs.

Output Format:

Resume Detected: Yes/No

If Yes:
Suggested Roles:
1. Role Name
   Reason

2. Role Name
   Reason

3. Role Name
   Reason

Skill Gap Analysis (if JD available):
- Matching Skills:
- Missing Skills:
- Recommendations:

If No:
Document Type:
Explanation:
Apology:
"""
    response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            'role':'user',
            'content':prompt
        }]
    )
    ans=response.choices[0].message.content
    return score,ans

#Document_chat agent
def chat(data,pdf):
    prompt=f"""
You are an expert of AI
Document:
{pdf}
Query:
{data}

Talk with user in the context of document and query
Rule:
1.Explain user query
2.Solve the query
3.Make comfortable bond
"""
    response=client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role':'user','content':prompt}]
    )
    answer=response.choices[0].message.content
    return answer
#To clear cache
import streamlit as st
@st.cache_data
def cache_data(data):
    return search(data)
#To collect data from internet
def search(data):
    results=[]
    try:
        with DDGS() as ddgs:
            search=list(ddgs.text(f"{data} statistics trends data report",
                                  max_results=10))
            for result in  search:
                results.append(
                    f"""
                    TITLE: {result.get('title', '')}

                    CONTENT:{result.get('body', '')}

                    SOURCE:{result.get('href', '')}
                    """)
    except Exception as e:
        print("SEARCH ERROR:", e)
    return "\n\n".join(results)
# Research Agent
def research_generate(data):
    web_data=cache_data(data)
    web_data = web_data[:3000]
    prompt = f"""
You are a world-class AI Research Analyst, Data Analyst, and Technical Writer.

Your task is to generate a highly detailed, professional, analytical, and insightful research report STRICTLY using the provided research data.

TOPIC:
{data}

RESEARCH DATA:
{web_data}

==================================================
IMPORTANT INSTRUCTIONS
==================================================

1. Use ONLY the provided research data.
2. Never invent facts, statistics, percentages, references, rankings, measurements, or values.
3. If information is missing, clearly mention the limitation.
4. Write in a professional research-report style.
5. Generate detailed explanations and deep analysis.
6. Expand sections properly with meaningful insights.
7. Use headings, subheadings, bullet points, and structured formatting.
8. Include:
   - trends
   - comparisons
   - advantages
   - disadvantages
   - opportunities
   - challenges
   - future scope
   - practical applications
   whenever relevant.

==================================================
DATA ANALYSIS & VISUALIZATION RULES
==================================================

If the research data contains:
- numbers
- statistics
- percentages
- rankings
- financial values
- measurements
- survey results
- timelines
- growth metrics
- comparisons
- scientific data

Then you MUST:

1. Extract the numerical data.
2. Analyze trends and relationships.
3. Compare values when possible.
4. Explain insights from the data.
5. Generate visualization-ready JSON.
6. Select the best chart automatically:
   - bar
   - line
   - pie
   - scatter
   - area

IMPORTANT:
You MUST ALWAYS return a VISUALIZATION_JSON section.

==================================================
VISUALIZATION JSON FORMAT
==================================================

If numerical/statistical data exists, return EXACTLY this format:

VISUALIZATION_JSON:
[
    {{
        "title": "Chart Title",
        "chart_type": "bar",
        "x": ["A", "B", "C"],
        "y": [10, 20, 30]
    }}
]

Rules:
- Return ONLY valid JSON.
- Use ONLY real extracted values.
- Never invent data.
- Arrays must be properly aligned.
- Multiple charts are allowed.
- Do not write explanations inside JSON.

If no valid numerical/statistical data exists, return EXACTLY:

VISUALIZATION_JSON:
[]

IMPORTANT:
Do NOT write only the heading "Visualization Data".
You MUST return actual VISUALIZATION_JSON.

==================================================
THEORETICAL TOPIC RULES
==================================================

If the topic is:
- theoretical
- conceptual
- philosophical
- descriptive
- historical
- educational

Then:
- Focus on explanation and analysis.
- Do NOT invent numerical data.
- Return:

VISUALIZATION_JSON:
[]

==================================================
OUTPUT FORMAT
==================================================

# Title

# Abstract

# Introduction

# Background / Context

# Key Concepts

# Detailed Analysis

# Key Findings

# Trend Analysis
(If applicable)

# Comparative Analysis
(If applicable)

# Statistical / Numerical Insights
(If applicable)

# Visualization Data

# Numerical Examples & Calculations
(Only if sufficient numerical data exists)

# Advantages

# Challenges / Limitations

# Future Scope

# Practical Applications

# Expert Insights

# Conclusion

# References

==================================================
FINAL RULES
==================================================

- Never hallucinate information.
- Never create fake statistics.
- Never create fake references.
- Never create fictional charts.
- Maintain professional research quality.
- Produce deep, meaningful, and useful analysis.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1200,
            messages=[
                {
                    'role':'user',
                    'content':prompt
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

         
