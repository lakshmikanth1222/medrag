
import os, requests, streamlit as st
from datetime import datetime

API_BASE_URL=os.getenv("API_BASE_URL","http://localhost:8000")

st.set_page_config(page_title="Medical AI Assistant",layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages=[]
if "patient_id" not in st.session_state:
    st.session_state.patient_id=None

@st.cache_data(ttl=60)
def load_patients():
    return requests.get(f"{API_BASE_URL}/patients",timeout=20).json()["patients"]

with st.sidebar:
    st.title("Patients")
    try:
        pts=load_patients()
        names={p["name"]:p["patient_id"] for p in pts}
        selected=st.selectbox("Select Patient",list(names.keys()))
        st.session_state.patient_id=names[selected]
    except Exception as e:
        st.error(str(e))

st.title("Medical Record AI Assistant")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt:=st.chat_input("Ask about patient records"):
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            r=requests.post(
                f"{API_BASE_URL}/chat",
                json={
                    "message":prompt,
                    "patient_id":st.session_state.patient_id
                },
                timeout=120
            )
            ans=r.json()["answer"]
            st.markdown(ans)

    st.session_state.messages.append(
        {"role":"assistant","content":ans}
    )
