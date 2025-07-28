import streamlit as st
from mistralai import Mistral
import whisper
import tempfile
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import ast
import torch
import subprocess
import tempfile
import os
import streamlit.components.v1 as components
import speech_recognition as sr
from pydub import AudioSegment
from streamlit_javascript import st_javascript
import pathlib
from streamlit_ace import st_ace
import io, contextlib


st.set_page_config(page_title="AI Code Agent", layout="wide")
st.title("AI Code Agent — Enhanced with Smart Modules")

st.markdown("""
<style>
/* Import Modern Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Root Variables */
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #8b5cf6;
    --accent: #06b6d4;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --dark: #0f172a;
    --dark-light: #1e293b;
    --gray: #64748b;
    --gray-light: #94a3b8;
    --white: #ffffff;
    --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --glass-bg: rgba(255, 255, 255, 0.95);
    --glass-border: rgba(255, 255, 255, 0.2);
    --shadow-soft: 0 10px 40px rgba(0, 0, 0, 0.1);
    --shadow-hard: 0 20px 60px rgba(0, 0, 0, 0.15);
}

/* Global Resets */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Inter', sans-serif;
}

/* Hide Default Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Container Enhancement */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    margin: 2rem auto;
    box-shadow: var(--shadow-hard);
    border: 1px solid var(--glass-border);
}

/* Custom Header */
.custom-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    padding: 2rem 3rem;
    margin: -2rem -3rem 3rem -3rem;
    border-radius: 24px 24px 0 0;
    color: white;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.custom-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
    opacity: 0.3;
}

.header-content {
    position: relative;
    z-index: 2;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: linear-gradient(45deg, #ffffff, #e2e8f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.main-subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
    font-weight: 400;
    margin-bottom: 1rem;
}

.feature-badges {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}

.badge {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Enhanced Sidebar */
.css-1d391kg, .css-1lcbmhc {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid var(--glass-border) !important;
}

.sidebar .sidebar-content {
    padding: 2rem 1rem;
}

/* Navigation Enhancement */
.stSelectbox > div > div {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
}

.stRadio > div {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid var(--glass-border);
}

.stRadio > div > label {
    background: transparent !important;
    color: var(--dark) !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    margin: 0.25rem 0.5rem !important;
    transition: all 0.3s ease !important;
    border: 1px solid transparent !important;
}

.stRadio > div > label:hover {
    background: var(--primary) !important;
    color: white !important;
    transform: translateY(-2px);
    box-shadow: var(--shadow-soft);
}

/* Tab Enhancement */
.stTabs [data-baseweb="tab-list"] {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 0.5rem;
    border: 1px solid var(--glass-border);
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--gray) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.5rem !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
    box-shadow: var(--shadow-soft) !important;
}

/* Chat Messages Enhancement */
.stChatMessage {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    border-radius: 16px !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: var(--shadow-soft) !important;
    transition: all 0.3s ease !important;
}

.stChatMessage:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hard);
}

/* Code Block Enhancement */
.stCodeBlock {
    background: var(--dark) !important;
    border-radius: 12px !important;
    border: 1px solid var(--dark-light) !important;
    box-shadow: var(--shadow-soft) !important;
    overflow: hidden !important;
}

.stCodeBlock pre {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
}

/* Button Enhancement */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: var(--shadow-soft) !important;
    text-transform: none !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--shadow-hard) !important;
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--secondary) 100%) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Input Field Enhancement */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* File Uploader Enhancement */
.stFileUploader {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    border: 2px dashed var(--primary) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    text-align: center !important;
    transition: all 0.3s ease !important;
}

.stFileUploader:hover {
    background: rgba(99, 102, 241, 0.05) !important;
    transform: translateY(-2px);
    box-shadow: var(--shadow-soft);
}

/* Expander Enhancement */
.stExpander {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-soft) !important;
}

.stExpander > div > div > div > div {
    padding: 1.5rem !important;
}

/* Metrics Enhancement */
[data-testid="metric-container"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px) !important;
    padding: 1.5rem !important;
    border-radius: 12px !important;
    border: 1px solid var(--glass-border) !important;
    box-shadow: var(--shadow-soft) !important;
    transition: all 0.3s ease !important;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-hard);
}

/* Spinner Enhancement */
.stSpinner {
    color: var(--primary) !important;
}

/* Column Enhancement */
.stColumn {
    padding: 0.5rem !important;
}

/* Success/Warning/Error Messages */
.stSuccess, .stWarning, .stError, .stInfo {
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* Ace Editor Enhancement */
.ace_editor {
    border-radius: 12px !important;
    box-shadow: var(--shadow-soft) !important;
    border: 1px solid var(--dark-light) !important;
}

/* Sidebar Navigation Items */
.sidebar-nav-item {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    padding: 0.75rem 1rem;
    margin: 0.25rem 0;
    border-radius: 8px;
    border: 1px solid var(--glass-border);
    transition: all 0.3s ease;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--dark);
    font-weight: 500;
}

.sidebar-nav-item:hover {
    background: var(--primary);
    color: white;
    transform: translateX(4px);
}

/* Professional Headers */
.section-header {
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    text-align: center;
}

.subsection-header {
    color: var(--dark);
    font-size: 1.3rem;
    font-weight: 600;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary);
    display: inline-block;
}

/* Tool Cards */
.tool-card {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-soft);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.tool-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.tool-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-hard);
}

.tool-card:hover::before {
    transform: scaleX(1);
}

/* Status Indicators */
.status-success {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.status-warning {
    background: rgba(245, 158, 11, 0.1);
    color: var(--warning);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

.status-error {
    background: rgba(239, 68, 68, 0.1);
    color: var(--danger);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
        margin: 1rem;
    }
    
    .custom-header {
        padding: 1.5rem;
        margin: -1rem -1rem 2rem -1rem;
    }
    
    .main-title {
        font-size: 2rem;
    }
    
    .feature-badges {
        flex-direction: column;
        align-items: center;
    }
}

/* Loading Animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s ease-in-out infinite;
}

/* Floating Action Button */
.fab {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 1.5rem;
    box-shadow: var(--shadow-hard);
    cursor: pointer;
    transition: all 0.3s ease;
    z-index: 1000;
}

.fab:hover {
    transform: translateY(-4px) scale(1.1);
    box-shadow: 0 20px 40px rgba(99, 102, 241, 0.4);
}
</style>
""", unsafe_allow_html=True)

# === CUSTOM HEADER ===
st.markdown("""
<div class="custom-header">
    <div class="header-content">
        <h1 class="main-title">🚀 CodeCraft AI</h1>
        <p class="main-subtitle">Professional Development Assistant - Powered by Advanced AI</p>
    </div>
</div>
""", unsafe_allow_html=True)

# === Mistral Client ===
API_KEY = "yeP0BrAleW2X2Mu4zqx1mZBt1GvXN1EX"
client = Mistral(api_key=API_KEY)

@st.cache_resource
def create_agent():
    agent = client.beta.agents.create(
        model="mistral-medium-2505",
        name="CodeChat_Agent",
        description="Conversational code assistant"
    )
    return agent.id

agent_id = create_agent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def detect_intent(prompt):
    prompt_lower = prompt.lower()
    if "bug" in prompt_lower:
        return "fix_bugs"
    elif "summarize" in prompt_lower:
        return "summarize"
    elif "docstring" in prompt_lower:
        return "docstring"
    elif "test" in prompt_lower or "unit test" in prompt_lower:
        return "testgen"
    elif "optimize" in prompt_lower:
        return "optimize"
    elif "refactor" in prompt_lower:
        return "refactor"
    elif "quality" in prompt_lower:
        return "quality"
    elif "explain line" in prompt_lower:
        return "line_explain"
    elif "preview html" in prompt_lower or "show html" in prompt_lower:
        return "live_preview"
    else:
        return "chat"

# === Mistral Ask ===
def ask_agent_streaming(prompt):
    messages = [{"role": "user", "content": prompt}]
    for entry in st.session_state.chat_history:
        messages.insert(0, {"role": "assistant", "content": entry["agent"]})
        messages.insert(0, {"role": "user", "content": entry["user"]})
    response = client.chat.complete(model="mistral-medium-2505", messages=messages)
    return response.choices[0].message.content

def ask_agent_streaming1(prompt):
    messages = [{"role": "user", "content": prompt}]
    for entry in st.session_state.chat_history:
        messages.insert(0, {"role": "assistant", "content": entry["agent"]})
        messages.insert(0, {"role": "user", "content": entry["user"]})
    response = client.chat.complete(model="mistral-small-latest", messages=messages)
    return response.choices[0].message.content


# === Suggestions Generator ===
def get_suggestions(agent_output):
    prompt = f"""
Suggest 3 follow-up questions based on this output:
\"\"\"{agent_output}\"\"\"
Only return a bulleted list.
"""
    return client.chat.complete(model="mistral-medium-2505", messages=[{"role": "user", "content": prompt}]).choices[0].message.content

def transcribe_audio(file) -> str:
    model = whisper.load_model("base")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    result = model.transcribe(tmp_path)
    return result["text"]

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# 🔄 Function to convert image ➝ caption ➝ code
def image_to_code(img):
    processor, model = load_blip()
    inputs = processor(img, return_tensors="pt").to("cpu")
    model.to("cpu")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # 🧠 Stronger prompt
    prompt = (
        f"You are an expert frontend developer.\n"
        f"The following is a description of a user interface extracted from an image:\n\n"
        f"\"{caption}\"\n\n"
        f"Generate clean, semantic HTML and CSS (or React) code that replicates this UI. Use appropriate structure and styling."
    )

    return prompt, ask_agent_streaming(prompt)  # returns both for debugging


def analyze_code_quality(code):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as f:
        f.write(code)
        tmp_path = f.name
    result = subprocess.run(["pylint", tmp_path, "--disable=all", "--enable=C,R,W,E"], capture_output=True, text=True)
    return result.stdout

def explain_line(line):
    if not line.strip():
        return "Skipped (empty line)"
    prompt = f"Explain this Python code line:\n{line}"
    response = client.chat.complete(
        model = "mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def summarize_code(code):
    prompt = f"Summarize this entire Python code:\n{code}"
    response = client.chat.complete(
        model = "mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_lecture_notes(code):
    prompt = f"Generate developer lecture notes from this Python code:\n{code}"
    response = client.chat.complete(
        model = "mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_quiz(code):
    prompt = f"Generate a short Python quiz (MCQs) from this code:\n{code}"
    response = client.chat.complete(
        model = "mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def chat_with_code_mistral(question, code_context):
    prompt = f"""You are a coding assistant. Given the following code context:

{code_context}

Answer the following user question:
{question}
"""

    response = client.chat.complete(
        model = "mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()



TASK_CONFIG = {
    "fix_bugs": {
        "label": "🔧 Bug Fixing Module...",
        "prompt": "Fix bugs in this code with explanation:\n\n{input}"
    },
    "summarize": {
        "label": "🧠 Code Summary...",
        "prompt": "Summarize the following code:\n\n{input}"
    },
    "docstring": {
        "label": "📝 Adding Docstrings...",
        "prompt": "Add detailed docstrings:\n\n{input}"
    },
    "testgen": {
        "label": "🧪 Generating Unit Tests...",
        "prompt": "Write unit tests using pytest for:\n\n{input}"
    },
    "optimize": {
        "label": "🚀 Optimizing Code...",
        "prompt": "Optimize this code for performance and readability:\n\n{input}"
    },
    "refactor": {
        "label": "♻️ Refactoring Code...",
        "prompt": "Refactor the following code with improvements:\n\n{input}"
    },
    "quality": {
        "label": "📊 Analyzing Code Quality...",
        "prompt": None
    },
    "line_explain": {
        "label": "📚 Explaining Line by Line...",
        "prompt": "Explain the following code line by line:\n\n{input}"
    },
    "live_preview": {
        "label": "🖥️ Live HTML Preview...",
        "prompt": None
    },
    "chat": {
        "label": "🤖 Chat Assistant...",
        "prompt": "{input}"
    }
}

# === Sidebar Navigation ===
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("🔍 Choose Section:", [
    "💬 Chat & Voice", 
    "📁 File & Image", 
    "🔧 Code Tools", 
    "📊 Analyze Tools",
    "💻 Code Editor" 
])


if st.sidebar.button("🔁 Reset All"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

if section == "💬 Chat & Voice":
    tab1, tab2 = st.tabs(["💬 Chat", "🎙️ Voice Input"])

    with tab1:
        st.subheader("💬 Chat Interface")
        for entry in st.session_state.chat_history:
            with st.chat_message("user", avatar="👤"):
                st.markdown(entry["user"])
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(entry["agent"])

    user_prompt = st.chat_input("Type your code or question here...")
    if user_prompt:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_prompt)

        task = detect_intent(user_prompt)
        config = TASK_CONFIG.get(task, TASK_CONFIG["chat"])

        if task == "quality":
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing quality..."):
                    result = f"Mocked quality check result for: {user_prompt}"
                    st.markdown(result)
            st.session_state.chat_history.append({"user": user_prompt, "agent": result})

        elif task == "live_preview":
            with st.chat_message("assistant", avatar="🤖"):
                st.code(user_prompt, language="html")
                if st.button("🔍 Show HTML Preview"):
                    st.components.v1.html(user_prompt, height=500)
            st.session_state.chat_history.append({"user": user_prompt, "agent": "✅ Preview shown."})

        else:
            final_prompt = config["prompt"].format(input=user_prompt)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response = ask_agent_streaming(final_prompt)
                    st.markdown(response)
            st.session_state.chat_history.append({"user": user_prompt, "agent": response})

            with st.expander("💡 Suggestions"):
                st.markdown(get_suggestions(response))

    with tab2:
        st.subheader("🎙️ Voice Input (Auto Send to Chat)")
        st.markdown("Click the button below to speak. Your voice will be transcribed and sent to the AI agent automatically.")

        # Use JavaScript to capture speech
        transcript = st_javascript("""
        const sleep = (ms) => new Promise(r => setTimeout(r, ms));
        const start = async () => {
            const recognition = new webkitSpeechRecognition();
            recognition.lang = 'en-US';
            recognition.continuous = false;
            recognition.interimResults = false;

            const result = await new Promise((resolve, reject) => {
                recognition.onresult = (event) => {
                    resolve(event.results[0][0].transcript);
                };
                recognition.onerror = reject;
                recognition.start();
            });

            await sleep(100);
            return result;
        };
        start();
        """)

        if transcript:
            st.success(f"🗣️ Transcribed: {transcript}")
            with st.spinner("🧠 AI is thinking..."):
                response = ask_agent_streaming(transcript)
                st.session_state.chat_history.append({"user": transcript, "agent": response})
                st.markdown(response)
                with st.expander("💡 Suggestions"):
                    st.markdown(get_suggestions(response))
        else:
            st.info("Click the button below to start voice recognition.")
            st.button("🎤 Start Speaking")

elif section == "📁 File & Image":
    tab3, tab4 = st.tabs(["📂 Code File", "🖼️ Image Upload"])

    with tab3:
        st.subheader("📂 Upload or Load Code Files")

        if "file_contents" not in st.session_state:
            st.session_state.file_contents = {}

        supported_extensions = (".py", ".txt", ".md", ".html", ".css", ".js", ".json", ".xml", ".java", ".cpp", ".c", ".h", ".sql", ".csv", ".yaml", ".yml")  # Add more as needed

        mode = st.radio("Select Input Mode:", ["Upload Files", "GitHub Repo URL", "Local Folder"], horizontal=True)

        # 1. Upload Files
        if mode == "Upload Files":
            uploaded_files = st.file_uploader("Upload multiple code/text files", accept_multiple_files=True)
            if uploaded_files:
                new_contents = {}
                for file in uploaded_files:
                    try:
                        content = file.read().decode("utf-8")
                        new_contents[file.name] = content
                    except Exception:
                        st.warning(f"Skipping unreadable file: {file.name}")
                st.session_state.file_contents = new_contents


        # 2. GitHub Repo Clone
        elif mode == "GitHub Repo URL":
            github_url = st.text_input("Enter GitHub repository URL:")
            if st.button("🔗 Clone & Load"):
                with st.spinner("Cloning repository..."):
                    import subprocess, os, tempfile, glob
                    temp_dir = tempfile.mkdtemp()
                    try:
                        subprocess.run(["git", "clone", github_url, temp_dir], check=True)
                        new_contents = {}
                        all_files = glob.glob(os.path.join(temp_dir, "**", "*.*"), recursive=True)
                        for path in all_files:
                            if path.lower().endswith(supported_extensions):
                                try:
                                    with open(path, "r", encoding="utf-8") as f:
                                        relative_path = os.path.relpath(path, temp_dir)
                                        new_contents[relative_path] = f.read()
                                except Exception:
                                    continue
                        st.session_state.file_contents = new_contents
                        st.success(f"Loaded {len(new_contents)} files.")
                    except Exception as e:
                        st.error(f"Clone failed: {e}")


        # 3. Local Folder Load
        elif mode == "Local Folder":
            local_path = st.text_input("Enter local folder path:")
            if st.button("📁 Load Folder"):
                import glob
                try:
                    new_contents = {}
                    all_files = glob.glob(os.path.join(local_path, "**", "*.*"), recursive=True)
                    for path in all_files:
                        if path.lower().endswith(supported_extensions):
                            try:
                                with open(path, "r", encoding="utf-8") as f:
                                    new_contents[os.path.relpath(path, local_path)] = f.read()
                            except Exception:
                                continue
                    st.session_state.file_contents = new_contents
                    st.success(f"Loaded {len(new_contents)} files.")
                except Exception as e:
                    st.error(f"Error: {e}")
        file_contents = st.session_state.file_contents

        if file_contents:
            file_list = list(file_contents.keys())

            if "selected_file" not in st.session_state or st.session_state.selected_file not in file_list:
                st.session_state.selected_file = file_list[0]

            selected_file = st.selectbox(
                "Select a file to view",
                file_list,
                index=file_list.index(st.session_state.selected_file),
                key="selected_file"
            )

            code = file_contents[st.session_state.selected_file]

            # Detect language
            ext = pathlib.Path(selected_file).suffix.lower()
            language_map = {
                ".py": "python", ".md": "markdown", ".html": "html",
                ".css": "css", ".js": "javascript", ".json": "json",
                ".xml": "xml", ".java": "java", ".cpp": "cpp", ".c": "c",
                ".csv": "text", ".txt": "text"
            }
            language = language_map.get(ext, "text")

            # Show preview
            if ext == ".md":
                st.markdown(code)
            else:
                st.code(code, language=language)

            # Action buttons
            col1, col2, col3, col4, col5 = st.columns(5)

            if col1.button("🎓 Lecture Notes", key="lecnotes"):
                with st.spinner("Generating notes..."):
                    notes = generate_lecture_notes(code)
                    st.markdown(notes)

            if col2.button("🌊 Explain Line by Line", key="lineexplain"):
                with st.spinner("Explaining code..."):
                    explanation = ""
                    for i, line in enumerate(code.split("\n"), 1):
                        explanation_text = explain_line(line)
                        explanation += f"**Line {i}:** `{line.strip()}`\n\n➡️ {explanation_text}\n\n"
                    st.markdown(explanation)

            if col3.button("🔢 Quiz Me", key="quizme"):
                with st.spinner("Generating quiz..."):
                    quiz = generate_quiz(code)
                    st.markdown(quiz)

            if col4.button("📄 Summarize File", key="summarizefile"):
                with st.spinner("Summarizing..."):
                    summary = summarize_code(code)
                    st.markdown(f"### 📃 Summary of `{selected_file}`\n\n{summary}")
            if col5.button("💬 Chat with Code", key="chatcode"):
                st.session_state.chat_active = True
            
            if st.session_state.get("chat_active", False):
                st.markdown("### 💬 Chat with Your Code")

                chat_mode = st.radio("Chat Scope:", ["Current File", "All Files"], horizontal=True)

                user_input = st.text_input("Ask a question about your code:")

                if user_input:
                    with st.spinner("Thinking with Mistral..."):
                        if chat_mode == "Current File":
                            context = code
                        else:
                            # Combine all file contents
                            context = "\n\n".join(f"# {name}\n{content}" for name, content in file_contents.items())

                        response = chat_with_code_mistral(user_input, context)
                        st.markdown(f"**🧠 Mistral Response:**\n\n{response}")


    with tab4:
        st.subheader("🖼️ Image to Code")
        image_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        if st.button("Generate Code from Image") and image_file:
            try:
                img = Image.open(image_file).convert("RGB")

                with st.spinner("Generating code from image..."):
                    prompt_used, code_output = image_to_code(img)

                st.code(code_output, language="html")
                st.components.v1.html(code_output, height=500)

                # Save in chat history
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({"user": "Generated code for uploaded image", "agent": code_output})

                # Caption & suggestions
                with st.expander("📜 Prompt Used"):
                    st.markdown(prompt_used)

                with st.expander("💡 Suggestions"):
                    st.markdown(get_suggestions(code_output))

            except Exception as e:
                st.error(f"❌ Failed to generate code: {e}")

elif section == "📊 Analyze Tools":
    st.title("📊 Code Analysis Tools")

    # === Initialize Session State ===
    st.session_state.setdefault("analyze_file_contents", {})

    supported_extensions = (
        ".py", ".txt", ".md", ".html", ".css", ".js", ".json", ".xml",
        ".java", ".cpp", ".c", ".h", ".sql", ".csv", ".yaml", ".yml"
    )
    # === Input Mode Selection ===
    st.markdown("### 🧾 Select Code Input Mode")
    input_mode = st.radio(
        "Choose input mode:",
        ["Paste Code", "Upload Files", "GitHub Repo", "Local Folder"],
        horizontal=True
    )
    # === Input Mode Handling ===
    if input_mode == "Paste Code":
        pasted_code = st.text_area("Paste your code here:", height=200)
        if pasted_code.strip():
            st.session_state.analyze_file_contents = {"pasted_code.py": pasted_code}

    elif input_mode == "Upload Files":
        uploaded_files = st.file_uploader("Upload multiple files", accept_multiple_files=True)
        new_contents = {}
        if uploaded_files:
            for file in uploaded_files:
                try:
                    content = file.read().decode("utf-8")
                    new_contents[file.name] = content
                except Exception:
                    st.warning(f"Skipping unreadable file: {file.name}")
            st.session_state.analyze_file_contents = new_contents

    elif input_mode == "GitHub Repo":
        github_url = st.text_input("Enter GitHub repo URL:")
        if st.button("🔗 Clone Repo"):
            import tempfile, subprocess, glob
            temp_dir = tempfile.mkdtemp()
            try:
                subprocess.run(["git", "clone", github_url, temp_dir], check=True)
                new_contents = {}
                all_files = glob.glob(os.path.join(temp_dir, "**", "*.*"), recursive=True)
                for path in all_files:
                    if path.lower().endswith(supported_extensions):
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                new_contents[os.path.relpath(path, temp_dir)] = f.read()
                        except Exception:
                            continue
                st.session_state.analyze_file_contents = new_contents
                st.success(f"✅ Loaded {len(new_contents)} files.")
            except Exception as e:
                st.error(f"❌ Clone failed: {e}")

    elif input_mode == "Local Folder":
        local_path = st.text_input("Enter full local folder path:")
        if st.button("📁 Load Folder"):
            import glob
            try:
                new_contents = {}
                all_files = glob.glob(os.path.join(local_path, "**", "*.*"), recursive=True)
                for path in all_files:
                    if path.lower().endswith(supported_extensions):
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                new_contents[os.path.relpath(path, local_path)] = f.read()
                        except Exception:
                            continue
                st.session_state.analyze_file_contents = new_contents
                st.success(f"✅ Loaded {len(new_contents)} files.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    # === File Selector & Preview ===
    file_list = list(st.session_state.analyze_file_contents.keys())
    if not file_list:
        st.warning("⚠️ No code available. Please provide input.")
    else:
        selected_file = st.selectbox("📂 Select a file to analyze:", file_list)
        code_input = st.session_state.analyze_file_contents[selected_file]
        st.code(code_input, language="python")

        # === Analysis Tools Buttons ===
        col1, col2, col3 = st.columns(3)

        if col1.button("📈 Analyze Complexity"):
            with st.spinner("Analyzing time/space complexity..."):
                prompt = f"Analyze the time and space complexity of this code:\n\n{code_input}"
                response = ask_agent_streaming(prompt)
                st.markdown(f"### 📈 Complexity Analysis\n\n{response}")

        if col2.button("🔐 Security Audit"):
            with st.spinner("Performing security check..."):
                prompt = (
                    f"Perform a security audit of this code. "
                    f"Look for vulnerabilities like SQL injection, unsafe eval, etc.:\n\n{code_input}"
                )
                response = ask_agent_streaming(prompt)
                st.markdown(f"### 🔐 Security Report\n\n{response}")

        if col3.button("📄 Check Readability"):
            with st.spinner("Evaluating readability..."):
                prompt = f"Rate the readability and maintainability of this code and suggest improvements:\n\n{code_input}"
                response = ask_agent_streaming(prompt)
                st.markdown(f"### 📄 Readability Feedback\n\n{response}")


elif section == "🔧 Code Tools":
    st.title("🔧 Code Tools")

    st.markdown("### 📥 Paste Your Code Below")

    code_input = st_ace(
        placeholder="Paste your Python code here...",
        language="python",
        theme="monokai",  # You can use "twilight", "github", "dracula", etc.
        keybinding="vscode",  # familiar shortcuts
        font_size=14,
        tab_size=4,
        show_gutter=True,
        show_print_margin=False,
        wrap=True,
        height=200,
        key="code_tools_editor"
    )


    # Initialize states
    st.session_state.setdefault("translated_output", "")
    st.session_state.setdefault("add_comment_output", "")
    st.session_state.setdefault("api_output", "")
    st.session_state.setdefault("translate_triggered", False)
    st.session_state.setdefault("selected_lang", "Java")
    st.session_state.setdefault("active_tool", "")  # 🔁 Track last used tool

    # === Control Buttons ===
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💬 Add Comments"):
            with st.spinner("Adding comments..."):
                prompt = f"Add helpful inline comments to the following Python code:\n\n{code_input}"
                st.session_state.add_comment_output = ask_agent_streaming(prompt)
                st.session_state.active_tool = "comment"

    with col2:
        if st.button("🔁 Translate Code"):
            st.session_state.translate_triggered = True
            st.session_state.active_tool = "translate"


    with col3:
        if st.button("⚙ Generate FastAPI"):
            with st.spinner("Generating FastAPI endpoint..."):
                prompt = f"You're a senior Python developer. Convert the following Python function or logic into a production-ready FastAPI endpoint with request/response models, including route decorators.:\n\n{code_input}"
                st.session_state.api_output = ask_agent_streaming1(prompt)
                st.session_state.active_tool = "api"
                

    # === Language Selection (for translation only)
    if st.session_state.translate_triggered:
        st.markdown("### 🎯 Choose Target Language")
        st.session_state.selected_lang = st.selectbox(
            "Language:", ["Java", "C++", "JavaScript", "Go", "Rust"], key="translate_lang"
        )

        if st.button("✅ Submit Translation"):
            with st.spinner(f"Translating to {st.session_state.selected_lang}..."):
                prompt = f"Translate this Python code to {st.session_state.selected_lang}:\n\n{code_input}"
                st.session_state.translated_output = ask_agent_streaming(prompt)
                st.session_state.active_tool = "translate"

    # === 🔄 Reset Option
    if st.button("🔄 Reset Output"):
        st.session_state.translated_output = ""
        st.session_state.add_comment_output = ""
        st.session_state.api_output = ""
        st.session_state.translate_triggered = False
        st.session_state.active_tool = ""

    if st.session_state.active_tool == "comment":
        st.markdown("### 💬 Code with Comments")
        st.code(st.session_state.add_comment_output, language="python")

    elif st.session_state.active_tool == "translate":
        st.markdown(f"### 🌐 Translated to {st.session_state.selected_lang}")
        st.code(st.session_state.translated_output, language=st.session_state.selected_lang.lower())

    elif st.session_state.active_tool == "api":
        st.markdown("### ⚙ FastAPI Endpoint")
        st.code(st.session_state.api_output, language="python")

    elif st.session_state.active_tool == "":
        st.info("No output yet. Run a tool to see results.")



elif section == "💻 Code Editor":
    st.title("💻 Interactive Code Editor with Output & Preview")

    editor_lang = st.selectbox("🗂️ Choose Language", ["python", "html", "css", "javascript", "java", "c", "cpp"])

    code_input = st_ace(
        placeholder=f"Write your {editor_lang} code here...",
        language=editor_lang,
        theme="dracula",
        keybinding="vscode",
        font_size=14,
        tab_size=4,
        show_gutter=True,
        show_print_margin=False,
        wrap=True,
        height=400,
        auto_update=True,
        key="code_editor_block"
    )

    st.markdown("### 🖥️ Output or Preview")

    if st.button("🚀 Run Code / Preview"):
        if editor_lang == "python":
            try:
                output_buffer = io.StringIO()
                with contextlib.redirect_stdout(output_buffer):
                    exec(code_input, {})
                st.success("✅ Code executed successfully!")
                st.text_area("📤 Output", output_buffer.getvalue(), height=200)
            except Exception as e:
                st.error(f"❌ Error: {e}")

        elif editor_lang in ["c", "cpp", "java"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_ext = {"c": ".c", "cpp": ".cpp", "java": ".java"}[editor_lang]
                filename = os.path.join(tmpdir, "program" + file_ext)

                with open(filename, "w") as f:
                    f.write(code_input)

                try:
                    if editor_lang == "c":
                        exe_path = os.path.join(tmpdir, "program.out")
                        subprocess.run(["gcc", filename, "-o", exe_path], check=True)
                        result = subprocess.run([exe_path], capture_output=True, text=True)

                    elif editor_lang == "cpp":
                        exe_path = os.path.join(tmpdir, "program.out")
                        subprocess.run(["g++", filename, "-o", exe_path], check=True)
                        result = subprocess.run([exe_path], capture_output=True, text=True)

                    elif editor_lang == "java":
                        subprocess.run(["javac", filename], check=True)
                        result = subprocess.run(["java", "-cp", tmpdir, "program"], capture_output=True, text=True)

                    st.success("✅ Execution successful!")
                    st.text_area("📤 Output", result.stdout, height=200)
                    if result.stderr:
                        st.text_area("⚠️ Errors", result.stderr, height=200)

                except subprocess.CalledProcessError as e:
                    st.error("❌ Compilation or Execution Error")
                    st.code(e.stderr if e.stderr else str(e))

        elif editor_lang == "html":
            st.success("✅ HTML Preview Below")
            components.html(code_input, height=500)

        elif editor_lang == "javascript":
            html_template = f"""
            <html>
            <body>
            <h4>JavaScript Output:</h4>
            <div id="output"></div>
            <script>
            try {{
                {code_input}
            }} catch (e) {{
                document.getElementById('output').innerText = e;
            }}
            </script>
            </body>
            </html>
            """
            components.html(html_template, height=500)

        elif editor_lang == "css":
            html_preview = f"""
            <html>
            <head><style>{code_input}</style></head>
            <body><div class="test">CSS applied successfully!</div></body>
            </html>
            """
            st.success("✅ CSS Preview Below")
            components.html(html_preview, height=500)

    st.download_button(
        label="📥 Download Code",
        data=code_input,
        file_name=f"your_code.{editor_lang}",
        mime="text/plain"
    )
