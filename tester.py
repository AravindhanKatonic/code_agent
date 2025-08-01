import streamlit as st
import os
import ast
import subprocess
import tempfile
import shutil
from io import BytesIO
import zipfile
from git import Repo
import time
from datetime import datetime
import json
from mistralai import Mistral

# ---------------- Configuration ----------------
st.set_page_config(
    page_title="AutoTestCopilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ---------------- Constants and Setup ----------------
UPLOAD_DIR = "uploads"
TEST_DIR = "generated_tests"
REPORT_DIR = "reports"
CONFIG_FILE = "config.json"

# Create directories
for directory in [UPLOAD_DIR, TEST_DIR, REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ---------------- Configuration Management ----------------
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"mistral_api_key": "", "model": "mistral-medium-2505", "temperature": 0.3}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'functions' not in st.session_state:
    st.session_state.functions = []
if 'generated_tests' not in st.session_state:
    st.session_state.generated_tests = {}

client = None
if st.session_state.config.get("mistral_api_key"):
    client = Mistral(api_key=st.session_state.config.get("mistral_api_key"))

# ---------------- Header ----------------
st.markdown("""
<div class="main-header">
    <h1>🤖 AutoTestCopilot</h1>
    <p>Professional Automated Test Generator for Python Code</p>
    <p><em>Generate comprehensive test suites with AI-powered analysis</em></p>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar Configuration ----------------
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("⚙️ Configuration")

    with st.expander("🔑 API Settings", expanded=False):
        api_key = st.text_input(
            "Mistral API Key",
            value=st.session_state.config.get("mistral_api_key", ""),
            type="password",
            help="Enter your Mistral API key"
        )

        model = st.selectbox(
            "Model",
            ["mistral-small-2402", "mistral-medium-2312", "mistral-medium-2505"],
            index=2 if st.session_state.config.get("model") == "mistral-medium-2505" else 0
        )

        temperature = st.slider(
            "Temperature",
            0.0, 1.0,
            st.session_state.config.get("temperature", 0.3),
            help="Controls randomness in AI responses"
        )

        if st.button("💾 Save Configuration"):
            st.session_state.config = {
                "mistral_api_key": api_key,
                "model": model,
                "temperature": temperature
            }
            save_config(st.session_state.config)
            st.success("Configuration saved!")
            client = Mistral(api_key=api_key)

    st.markdown('</div>', unsafe_allow_html=True)

    st.header("📥 Input Options")
    input_mode = st.radio(
        "Choose Input Mode",
        ["📁 Upload Python Files", "🔗 GitHub Repository"],
        help="Select how you want to provide your Python code"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Function Definitions ----------------
def extract_functions_from_code(code, filename):
    try:
        tree = ast.parse(code)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(code, node)
                docstring = ast.get_docstring(node) or "No docstring available"
                args = [arg.arg for arg in node.args.args]
                functions.append({
                    "name": node.name,
                    "code": func_code,
                    "filename": filename,
                    "docstring": docstring,
                    "args": args,
                    "line_number": node.lineno
                })
        return functions
    except SyntaxError as e:
        st.error(f"Syntax error in {filename}: {e}")
        return []
    except Exception as e:
        st.error(f"Error extracting functions from {filename}: {e}")
        return []

def generate_test_code(func_info, config):
    if not client:
        st.error("Please configure your Mistral API key in the sidebar")
        return None

    prompt = f"""
You are a senior Python testing expert. Create comprehensive pytest test cases for the following function.

Function Details:
- Name: {func_info['name']}
- Arguments: {func_info['args']}
- Docstring: {func_info['docstring']}

Code:
```python
{func_info['code']}
```

Requirements:
1. Write multiple test cases covering edge cases, normal cases, and error cases
2. Use pytest fixtures where appropriate
3. Mock external dependencies using unittest.mock
4. Include docstrings for test functions
5. Use descriptive test function names
6. Add parametrized tests where beneficial
7. Include setup and teardown if needed
8. Follow pytest best practices

Generate only the test code with proper imports.
"""

    try:
        response = client.chat.complete(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=config["temperature"]
        )
        raw_code = response.choices[0].message.content

        # Remove triple backticks or markdown artifacts
        cleaned_code = raw_code.strip().removeprefix("```python").removesuffix("```").strip()

        return cleaned_code
    except Exception as e:
        st.error(f"Error generating test code: {e}")
        return None


def run_pytest_and_report():
    """Run pytest and generate HTML report"""
    report_path = os.path.join(REPORT_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    try:
        result = subprocess.run(
            ["pytest", TEST_DIR, f"--html={report_path}", "--self-contained-html", "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.stdout, result.stderr, report_path, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Test execution timed out", report_path, 1
    except Exception as e:
        return "", f"Error running tests: {e}", report_path, 1

def clone_github_repo(github_url, clone_dir):
    """Clone GitHub repository"""
    try:
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        
        Repo.clone_from(github_url, clone_dir)
        return True, "Repository cloned successfully"
    except Exception as e:
        return False, f"Failed to clone repository: {e}"

# ---------------- Main Application Logic ----------------

# File Input Section
code_files = []

if "📁 Upload Python Files" in input_mode:
    with st.container():
        st.subheader("📁 File Upload")
        
        uploaded_files = st.file_uploader(
            "Select Python files to analyze",
            type=["py"],
            accept_multiple_files=True,
            help="Upload one or more Python files containing functions to test"
        )
        
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                filepath = os.path.join(UPLOAD_DIR, file.name)
                
                with open(filepath, "wb") as f:
                    f.write(file.read())
                code_files.append(filepath)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ All files uploaded successfully!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

elif "🔗 GitHub Repository" in input_mode:
    with st.container():
        st.subheader("🔗 GitHub Repository")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            github_url = st.text_input(
                "Repository URL",
                placeholder="https://github.com/username/repository",
                help="Enter the full GitHub repository URL"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            clone_button = st.button("🔄 Clone Repository", type="primary")
        
        if clone_button and github_url:
            with st.spinner("Cloning repository..."):
                temp_repo_dir = os.path.join(tempfile.gettempdir(), "cloned_repo")
                success, message = clone_github_repo(github_url, temp_repo_dir)
                
                if success:
                    # Find Python files
                    for root, _, files in os.walk(temp_repo_dir):
                        for file in files:
                            if file.endswith(".py") and not file.startswith("test_"):
                                code_files.append(os.path.join(root, file))
                    
                    if code_files:
                        st.markdown(f'<div class="success-box">✅ {message}<br>Found {len(code_files)} Python files</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ No Python files found in the repository</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)

# Function Analysis Section
if code_files:
    st.header("📊 Code Analysis")
    
    # Extract functions from all files
    all_functions = []
    
    with st.spinner("Analyzing code files..."):
        progress_bar = st.progress(0)
        
        for i, file_path in enumerate(code_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                    filename = os.path.basename(file_path)
                    functions = extract_functions_from_code(code, filename)
                    all_functions.extend(functions)
            except Exception as e:
                st.warning(f"Could not read {file_path}: {e}")
            
            progress_bar.progress((i + 1) / len(code_files))
    
    st.session_state.functions = all_functions
    progress_bar.empty()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📁 Files Processed", len(code_files))
    with col2:
        st.metric("🔧 Functions Found", len(all_functions))
    with col3:
        st.metric("🧪 Tests Generated", len(st.session_state.generated_tests))
    with col4:
        unique_files = len(set(f['filename'] for f in all_functions))
        st.metric("📄 Files with Functions", unique_files)
    
    if all_functions:
        st.subheader("🔍 Function Explorer")
        
        # Group functions by file
        files_dict = {}
        for func in all_functions:
            filename = func['filename']
            if filename not in files_dict:
                files_dict[filename] = []
            files_dict[filename].append(func)
        
        # File selector
        selected_file = st.selectbox(
            "Select File",
            list(files_dict.keys()),
            help="Choose a file to view its functions"
        )
        
        if selected_file:
            file_functions = files_dict[selected_file]
            
            # Function selector
            function_options = [f"{func['name']} (line {func['line_number']})" for func in file_functions]
            selected_func_display = st.selectbox(
                "Select Function",
                function_options,
                help="Choose a function to generate tests for"
            )
            
            if selected_func_display:
                # Find the selected function
                func_name = selected_func_display.split(" (line")[0]
                selected_function = next(f for f in file_functions if f['name'] == func_name)
                
                # Display function details
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Function Code:**")
                    st.code(selected_function['code'], language="python")
                
                with col2:
                    st.markdown("**Function Info:**")
                    st.write(f"**Name:** {selected_function['name']}")
                    st.write(f"**File:** {selected_function['filename']}")
                    st.write(f"**Line:** {selected_function['line_number']}")
                    st.write(f"**Arguments:** {', '.join(selected_function['args']) if selected_function['args'] else 'None'}")
                    
                    if selected_function['docstring'] != "No docstring available":
                        st.markdown("**Docstring:**")
                        st.write(selected_function['docstring'])
                
                # Test Generation Section
                st.subheader("🧪 Test Generation")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    generate_button = st.button(
                        f"🚀 Generate Test for {selected_function['name']}",
                        type="primary",
                        use_container_width=True
                    )
                
                with col2:
                    if selected_function['name'] in st.session_state.generated_tests:
                        st.success("✅ Test already generated")
                
                if generate_button:
                    if not st.session_state.config.get("mistral_api_key"):
                        st.error("Please configure your OpenAI API key in the sidebar first!")
                    else:
                        with st.spinner(f"Generating test for {selected_function['name']}..."):
                            test_code = generate_test_code(selected_function, st.session_state.config)
                            
                            if test_code:
                                # Save the test
                                test_filename = f"test_{selected_function['name']}.py"
                                test_path = os.path.join(TEST_DIR, test_filename)
                                
                                with open(test_path, "w", encoding="utf-8") as f:
                                    f.write(test_code)
                                
                                st.session_state.generated_tests[selected_function['name']] = {
                                    'code': test_code,
                                    'filename': test_filename,
                                    'generated_at': datetime.now().isoformat()
                                }
                                
                                st.markdown('<div class="success-box">✅ Test generated successfully!</div>', unsafe_allow_html=True)
                                
                                # Display generated test
                                with st.expander("📝 View Generated Test", expanded=True):
                                    st.code(test_code, language="python")
                
                # Show existing test if available
                if selected_function['name'] in st.session_state.generated_tests:
                    test_info = st.session_state.generated_tests[selected_function['name']]
                    
                    with st.expander("📝 Previously Generated Test", expanded=False):
                        st.code(test_info['code'], language="python")
                        st.caption(f"Generated: {test_info['generated_at']}")

# Test Execution Section
if st.session_state.generated_tests:
    st.header("🚀 Test Execution")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        run_tests_button = st.button("▶️ Run All Tests", type="primary", use_container_width=True)
    
    with col2:
        view_tests_button = st.button("👀 View Generated Tests", use_container_width=True)
    
    with col3:
        clear_tests_button = st.button("🗑️ Clear All Tests", use_container_width=True)
    
    if run_tests_button:
        with st.spinner("Running tests..."):
            stdout, stderr, report_path, return_code = run_pytest_and_report()
            
            if return_code == 0:
                st.markdown('<div class="success-box">✅ All tests passed!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">⚠️ Some tests failed or had issues</div>', unsafe_allow_html=True)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Test Output")
                st.text_area("Standard Output", stdout, height=300)
            
            with col2:
                st.subheader("❌ Errors/Warnings")
                st.text_area("Standard Error", stderr, height=300)
            
            # Download report
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button(
                        "📥 Download HTML Report",
                        f.read(),
                        file_name=f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
    
    if view_tests_button:
        st.subheader("📝 Generated Test Files")
        
        for func_name, test_info in st.session_state.generated_tests.items():
            with st.expander(f"🧪 {test_info['filename']}", expanded=False):
                st.code(test_info['code'], language="python")
                st.caption(f"Generated: {test_info['generated_at']}")
    
    if clear_tests_button:
        st.session_state.generated_tests = {}
        # Clear test directory
        for file in os.listdir(TEST_DIR):
            if file.startswith("test_") and file.endswith(".py"):
                os.remove(os.path.join(TEST_DIR, file))
        st.success("All tests cleared!")
        st.rerun()

# Download Section
if st.session_state.generated_tests:
    st.header("📦 Download Tests")
    
    # Create zip file with all tests
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for func_name, test_info in st.session_state.generated_tests.items():
            zip_file.writestr(test_info['filename'], test_info['code'])
        
        # Add requirements.txt
        requirements = "pytest>=7.0.0\npytest-html>=3.0.0\nunittest-mock\n"
        zip_file.writestr("requirements.txt", requirements)
        
    
    zip_buffer.seek(0)
    
    st.download_button(
        "⬇️ Download Complete Test Suite",
        zip_buffer.getvalue(),
        file_name=f"test_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True
    )

# Footer
st.markdown("---")
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("🧹 Cleanup")
    
    if st.button("🗑️ Reset Everything", use_container_width=True):
        # Clear directories
        for directory in [UPLOAD_DIR, TEST_DIR, REPORT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
        
        # Clear session state
        st.session_state.functions = []
        st.session_state.generated_tests = {}
        
        st.success("Everything cleared!")
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**AutoTestCopilot v2.0**")
    st.markdown("*Professional Test Generation Tool*")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d')}*")

# Help section
with st.expander("❓ How to Use AutoTestCopilot", expanded=False):
    st.markdown("""
    ## 🚀 Getting Started
    
    1. **Configure API**: Set your OpenAI API key in the sidebar
    2. **Upload Code**: Choose to upload Python files or clone a GitHub repository
    3. **Select Functions**: Browse and select functions you want to test
    4. **Generate Tests**: Click the generate button for each function
    5. **Run Tests**: Execute all tests and view results
    6. **Download**: Get your complete test suite as a ZIP file
    
    ## 💡 Tips
    
    - Provide functions with clear docstrings for better test generation
    - The AI works best with well-structured, documented code
    - Review generated tests before using in production
    - Use the HTML report for detailed test analysis
    
    ## 🛠️ Requirements
    
    - OpenAI API key
    - Internet connection for AI generation
    - Python environment with pytest installed for local execution
    """)
