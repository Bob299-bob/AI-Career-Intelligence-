import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
from xml.sax.saxutils import escape
from Document_brain import pdf_extract,RAG,Retrieve,Output,extract_skills,ATS,chat,research_generate

st.set_page_config(
    page_title="AI Career Intelligence",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<div style='margin-top:40px;'></div>
<h1 style='text-align:center; color:#38bdf8; font-size:40px;'>
🧠 AI Career Intelligence OS
</h1>
<p style='text-align:center; color:#94a3b8; padding:10px;'>
Resume • Chat • ATS • Research • Insights
</p>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0b1220;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
}

/* Link button (Streamlit uses anchor tag inside) */
.stLinkButton a {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    color: white !important;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    text-decoration: none;
    display: inline-block;
}

/* Hover effect for both */
.stButton>button:hover,
.stLinkButton a:hover {
    transform: scale(1.05);
    transition: 0.2s;
}
            
.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.2s;
}

/* Input fields */
input, textarea {
    border-radius: 10px !important;
}

/* Cards effect */
.block-container {
    padding: 2rem;
}

/* Chat bubbles */
.stChatMessage {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)


#Session_state
if 'page' not in st.session_state:
    st.session_state.page="Home"
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.sidebar.button('Home'):
    st.session_state.page="Home"
st.sidebar.link_button("Interview Coach", "https://ai-interview-coach-kfczgvadjvk8opbvekxmqc.streamlit.app/")
data_path=st.file_uploader('Upload your file',type=['pdf'])
query=st.text_input('Enter Your Query')
if st.button('Press'):
    st.session_state.page="press"    
    if data_path and query is not None:
        pdf_text=pdf_extract(data_path)
        index,pdf=RAG(pdf_text)
        chunks=Retrieve(query,index,pdf)
        answer=Output(query,chunks)
        st.success(answer)
    else:
        st.error('Please fill above fields')
if st.sidebar.button('Skill Gap Analyzer'):
    st.session_state.page="skill"
    if data_path and query is not None:
        pdf_text=pdf_extract(data_path)
        index,pdf=RAG(pdf_text) 
        chunks = Retrieve(query, index, pdf)
        context = "\n".join(chunks) 
        answer=extract_skills(context,query)
        st.success(answer)
    else:
        st.error('Please fill above fields')
if st.sidebar.button('Resume Analyzer'):
    st.session_state.page="analyze"
    if data_path and query is not None:
        pdf_text=pdf_extract(data_path)
        index,pdf=RAG(pdf_text)
        chunks = Retrieve(query, index, pdf)
        context = "\n".join(chunks) 
        ans=ATS(context,query)
        st.success(ans)
    else:
         st.error('Please fill above fields') 
if st.sidebar.button('Document chat'):
     st.session_state.page="Document_chat"
if st.session_state.page=="Document_chat":
    if data_path and query is not None:    
        st.session_state.page="Document_chat"
        st.header("📄 Document Brain")
        st.info("Type your query in the box below to continue.👇")
        # Old messages show karo
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if st.button("New Chat"):
                st.session_state.messages = []
                st.rerun()  
        # Chat input
        data = st.chat_input("Ask about your document")
        if data:
            # User message save
            st.session_state.messages.append({"role": "user","content": data})
            pdf_text=pdf_extract(data_path)
            index,pdf=RAG(pdf_text)
            chunks = Retrieve(data, index, pdf)
            context = "\n".join(chunks[:3])   # important limit
            answer = chat(data, context)
            st.session_state.messages.append({"role": "assistant","content": answer})    
            st.rerun()     
    else:
        st.error('Please fill above fields')

def extract_graphs(report):
    try:
        if "VISUALIZATION_JSON:" not in report:
            return []
        json_text = report.split("VISUALIZATION_JSON:")[1].strip()
        # remove markdown blocks
        json_text = json_text.replace("```json", "")
        json_text = json_text.replace("```", "")
        # find first complete json array
        start = json_text.find("[")
        end = json_text.rfind("]") + 1
        json_text = json_text[start:end]
        return json.loads(json_text)
    except Exception as e:
        print("Graph Extraction Error:", e)
        return []
def generate_pdf(report_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []
    # Title
    content.append(Paragraph("AI Research Report", styles["Title"]))
    content.append(Spacer(1, 12))
    # Body
    for line in report_text.split("\n"):
        safe_line = escape(line)
        content.append(Paragraph(safe_line, styles["BodyText"]))
        content.append(Spacer(1, 6))
    doc.build(content)
    buffer.seek(0)
    return buffer
import json
import pandas as pd

if st.sidebar.button('Research Agent'):
    st.session_state.page='Research'
    st.rerun()
if st.session_state.page=='Research':
    with st.form('AI Research Assistant'):
        data=st.text_input('Enter Research Topic',placeholder="Press Enter key")
        submit=st.form_submit_button('Generate Report')

    if data.strip()!="":
        answer=research_generate(data)
        st.subheader("Reasearch Report")  
        pdf_down = generate_pdf(research_generate(data))   
        st.download_button(label="Download Report PDF",data=pdf_down,
                           file_name="ai_report.pdf",mime="application/pdf")
        st.warning('Please be pateint, it takes a few seconds to generate the report')         
        if "VISUALIZATION_JSON:" in research_generate(data):
            clean_report = research_generate(data).split("VISUALIZATION_JSON:")[0]
        else:
            clean_report = research_generate(data)
        st.markdown(clean_report)
        graphs = extract_graphs(research_generate(data))
        if graphs:
            st.subheader("📊 Visualizations")
            for graph in graphs:
                if not graph["x"] or not graph["y"]:
                    continue
                df = pd.DataFrame({
                    "Category": graph["x"],
                    "Value": graph["y"]
                    })
                chart_type = graph["chart_type"].lower()
                st.write(graph["title"])
                print(chart_type)
                if chart_type == "bar":
                    st.bar_chart(df.set_index("Category"))
                elif chart_type == "line":
                    st.line_chart(df.set_index("Category"))
                elif chart_type == "area":
                    st.area_chart(df.set_index("Category"))
                elif chart_type == "pie":
                    st.pyplot(df.set_index("Category").plot.pie(y="Value",autopct="%1.1f%%").figure)
                else:
                    st.dataframe(df)
        st.divider()   
    

