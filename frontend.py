import streamlit as st
import requests
import json
from typing import Optional
import time
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_BASE_URL = "http://localhost:8000"

# ==========================================
# 2. STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0F172A; }
    .main-header {
        font-size: 2.8rem; font-weight: 800; text-align: center; padding: 1rem;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    .debug-box {
        background-color: #1E293B;
        color: #94A3B8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
    [data-testid="stChatMessage"] {
        background-color: #1E293B; border-radius: 12px; padding: 15px; margin: 10px 0;
        border: 1px solid #334155; border-left: 4px solid #3B82F6;
    }
    .patient-card {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-bottom: 0.5rem;
        cursor: pointer;
    }
    .patient-card.selected {
        border-left: 4px solid #3B82F6;
        background-color: #2D3A4F;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🩺 Medical Record AI Assistant</h1>', unsafe_allow_html=True)

# ==========================================
# 3. SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your AI medical assistant. Select a patient from the sidebar and ask me anything about their medical records."}
    ]
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "api_healthy" not in st.session_state:
    st.session_state.api_healthy = False
if "api_status" not in st.session_state:
    st.session_state.api_status = {"gemini_api": "Unknown", "database": "Unknown", "patients_count": 0}
if "last_check_time" not in st.session_state:
    st.session_state.last_check_time = 0
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "patients" not in st.session_state:
    st.session_state.patients = []
if "debug_info" not in st.session_state:
    st.session_state.debug_info = ""

# ==========================================
# 4. API FUNCTIONS
# ==========================================

def check_api_health(force=False):
    """Check API health"""
    current_time = time.time()
    if not force and current_time - st.session_state.last_check_time < 30:
        return st.session_state.api_healthy
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.session_state.debug_info += f"\nHealth check: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.api_healthy = data["status"] in ["healthy", "degraded"]
            st.session_state.api_status = {
                "gemini_api": data["gemini_api"],
                "database": data["database"],
                "patients_count": data.get("patients_count", 0)
            }
            st.session_state.last_check_time = current_time
            return True
        else:
            st.session_state.api_healthy = False
            st.session_state.last_check_time = current_time
            return False
    except Exception as e:
        st.session_state.debug_info += f"\nHealth check error: {str(e)}"
        st.session_state.api_healthy = False
        st.session_state.api_status = {
            "gemini_api": "Connection Error",
            "database": "Connection Error",
            "patients_count": 0
        }
        st.session_state.last_check_time = current_time
        return False

def load_patients():
    """Load patients from API"""
    if not st.session_state.api_healthy:
        st.session_state.debug_info += "\nCannot load patients: API not healthy"
        return
    
    try:
        st.session_state.debug_info += "\nFetching patients..."
        response = requests.get(f"{API_BASE_URL}/patients", timeout=10)
        st.session_state.debug_info += f" Response: {response.status_code}"
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.patients = data.get("patients", [])
            st.session_state.debug_info += f" Found {len(st.session_state.patients)} patients"
            
            # Print first patient for debugging
            if st.session_state.patients:
                st.session_state.debug_info += f"\nSample: {st.session_state.patients[0].get('name')}"
        else:
            st.session_state.debug_info += f" Error: {response.text}"
    except Exception as e:
        st.session_state.debug_info += f" Error: {str(e)}"

def send_message(message: str, patient_id: Optional[str] = None):
    """Send message to API"""
    try:
        payload = {
            "message": message,
            "patient_id": patient_id,
            "session_id": st.session_state.session_id
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("session_id"):
                st.session_state.session_id = data["session_id"]
            return data
        else:
            st.error(f"❌ API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

# Initial checks
check_api_health(force=True)

if st.session_state.api_healthy and not st.session_state.patients:
    load_patients()

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 📊 System Status")
    
    # Status indicators
    if st.session_state.api_healthy:
        st.success("✅ Backend: Connected")
    else:
        st.error("❌ Backend: Disconnected")
        st.code("uvicorn backend:app --reload --port 8000")
    
    nemotron_status = st.session_state.api_status.get("gemini_api", "Unknown")
    if nemotron_status == "Connected":
        st.success("✅ Nemotron API: Connected")
    else:
        st.warning(f"⚠️ Nemotron API: {nemotron_status}")
    
    db_status = st.session_state.api_status.get("database", "Unknown")
    patients_count = st.session_state.api_status.get("patients_count", 0)
    if db_status == "Connected":
        st.success(f"✅ Database: Connected ({patients_count} patients)")
    else:
        st.error(f"❌ Database: {db_status}")
    
    if st.button("🔄 Refresh", use_container_width=True):
        check_api_health(force=True)
        load_patients()
        st.rerun()
    
    st.markdown("---")
    
    # Patient Selection
    st.markdown("## 👤 Select Patient")
    
    if st.session_state.patients:
        for patient in st.session_state.patients:
            # Calculate age
            try:
                dob = datetime.strptime(patient['date_of_birth'], '%Y-%m-%d')
                age = datetime.now().year - dob.year
            except:
                age = "?"
            
            # Patient card
            is_selected = st.session_state.selected_patient == patient['patient_id']
            bg_color = "#2D3A4F" if is_selected else "#1E293B"
            
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {'#3B82F6' if is_selected else '#334155'};">
                <b>{patient['name']}</b><br>
                <small>{age} yrs • {patient['gender']} • {patient['abha_id']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Select", key=f"select_{patient['patient_id']}", use_container_width=True):
                if st.session_state.selected_patient != patient['patient_id']:
                    st.session_state.messages = [{"role": "assistant", "content": f"Switched to patient: {patient['name']}. How can I help you with their records?"}]
                st.session_state.selected_patient = patient['patient_id']
                st.rerun()
    else:
        st.warning("No patients found")
        
        # Debug expander
        with st.expander("🔧 Debug Info"):
            st.code(st.session_state.debug_info)
    
    st.markdown("---")
    
    # Sample Questions
    st.markdown("## 💡 Try Asking")
    sample_questions = [
        "What is the diagnosis?",
        "Show me the medications",
        "Summarize lab findings",
        "What is the patient's medical history, summarize it?",
        "What are the doctor's recommendations?",
        "overall summary of the medical record?"
    ]
    
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.session_id = None
        st.rerun()

# ==========================================
# 6. MAIN CHAT
# ==========================================

# Show selected patient
if st.session_state.selected_patient:
    patient = next((p for p in st.session_state.patients if p['patient_id'] == st.session_state.selected_patient), None)
    if patient:
        try:
            dob = datetime.strptime(patient['date_of_birth'], '%Y-%m-%d')
            age = datetime.now().year - dob.year
            st.info(f"👤 **Current Patient:** {patient['name']} | {age} years | {patient['gender']} | ABHA: {patient['abha_id']}")
        except:
            st.info(f"👤 **Current Patient:** {patient['name']}")
else:
    st.info("👈 Please select a patient from the sidebar to start")

# Chat messages
# Chat messages
# Chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about patient's medical records..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Generate response if the last message is from the user
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        if not st.session_state.api_healthy:
            st.error("❌ Backend not connected")
        elif not st.session_state.selected_patient:
            st.warning("⚠️ Please select a patient first")
        else:
            with st.spinner("🔍 Analyzing medical records..."):
                user_query = st.session_state.messages[-1]["content"]
                response_data = send_message(user_query, st.session_state.selected_patient)
            
            if response_data:
                st.markdown(response_data['answer'])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data['answer']
                })
