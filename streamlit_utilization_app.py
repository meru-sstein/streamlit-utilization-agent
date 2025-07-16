import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from snowflake.snowpark import Session
import snowflake.connector
import numpy as np
from typing import Any, Dict, List, Optional

# Enhanced Configuration
USER_SNOWFLAKE = 'sam.stein@wearemeru.com'
ACCOUNT_SNOWFLAKE = 'FNJLIDA-GLA82740'
ROLE_SNOWFLAKE = 'MERU_DEVELOPER'
WAREHOUSE_SNOWFLAKE = 'CORTEX_WH'
AUTHENTICATOR_SNOWFLAKE = 'externalbrowser'
DATABASE_SNOWFLAKE = 'MERU_PREPARED'
SCHEMA_SNOWFLAKE = 'UTILIZATION_DASHBOARD'

# Cortex Analyst Configuration
CORTEX_DATABASE = 'MERU_PREPARED'
CORTEX_SCHEMA = 'UTILIZATION_DASHBOARD'
CORTEX_STAGE = 'RAG_YAML'  # You may need to create this stage
CORTEX_FILE = 'utilization_agent.yaml'
HOST_SNOWFLAKE = f'{ACCOUNT_SNOWFLAKE}.snowflakecomputing.com'

USE_CORTEX_ANALYST = True

# Enhanced Page Configuration
st.set_page_config(
    page_title="Advanced Utilization Analytics with Cortex",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    .info-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
    .response-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .cortex-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .sidebar-metric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
    }
    .context-pill {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.25rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def init_snowflake_connections():
    """Initialize both Snowpark and Connector sessions for Cortex Analyst."""
    # Initialize Snowpark session (for backward compatibility)
    if 'snowflake_session' not in st.session_state:
        try:
            with st.spinner("🔄 Connecting to Snowflake (Snowpark)..."):
                st.session_state.snowflake_session = Session.builder.configs({
                    'account': ACCOUNT_SNOWFLAKE,
                    'user': USER_SNOWFLAKE,
                    'role': ROLE_SNOWFLAKE,
                    'warehouse': WAREHOUSE_SNOWFLAKE,
                    'authenticator': AUTHENTICATOR_SNOWFLAKE,
                    'database': DATABASE_SNOWFLAKE,
                    'schema': SCHEMA_SNOWFLAKE
                }).create()
                
                # Test connection
                test_result = st.session_state.snowflake_session.sql("SELECT CURRENT_USER(), CURRENT_ROLE()").collect()
                st.success(f"✅ Snowpark connected as {test_result[0][0]} with role {test_result[0][1]}")
        except Exception as e:
            st.error(f"❌ Failed to connect via Snowpark: {str(e)}")
            return False
    
    # Initialize Connector session (for Cortex Analyst API)
    if 'snowflake_conn' not in st.session_state or st.session_state.snowflake_conn is None:
        try:
            with st.spinner("🔄 Connecting to Snowflake (Connector for Cortex)..."):
                st.session_state.snowflake_conn = snowflake.connector.connect(
                    user=USER_SNOWFLAKE,
                    account=ACCOUNT_SNOWFLAKE,
                    host=HOST_SNOWFLAKE,
                    port=443,
                    warehouse=WAREHOUSE_SNOWFLAKE,
                    role=ROLE_SNOWFLAKE,
                    authenticator=AUTHENTICATOR_SNOWFLAKE,
                    database=CORTEX_DATABASE,
                    schema=CORTEX_SCHEMA
                )
                st.success("✅ Connector session established for Cortex Analyst")
        except Exception as e:
            st.error(f"❌ Failed to connect via Connector: {str(e)}")
            return False
    
    return True

def send_cortex_message(prompt: str) -> Dict[str, Any]:
    """Calls the Cortex Analyst REST API and returns the response."""
    try:
        request_body = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "semantic_model_file": f"@{CORTEX_DATABASE}.{CORTEX_SCHEMA}.{CORTEX_STAGE}/{CORTEX_FILE}",
        }
        
        resp = requests.post(
            url=f"https://{HOST_SNOWFLAKE}/api/v2/cortex/analyst/message",
            json=request_body,
            headers={
                "Authorization": f'Snowflake Token="{st.session_state.snowflake_conn.rest.token}"',
                "Content-Type": "application/json",
            },
        )
        
        request_id = resp.headers.get("X-Snowflake-Request-Id")
        if resp.status_code < 400:
            return {**resp.json(), "request_id": request_id}
        else:
            raise Exception(
                f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.text}"
            )
    except Exception as e:
        st.error(f"❌ Cortex Analyst API Error: {str(e)}")
        return None

def display_cortex_content(
    content: List[Dict[str, str]],
    request_id: Optional[str] = None,
    message_index: Optional[int] = None,
) -> None:
    """Displays content from Cortex Analyst response."""
    message_index = message_index or len(st.session_state.get('cortex_messages', []))
    
    if request_id:
        with st.expander("🔍 Request ID", expanded=False):
            st.code(request_id)
    
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            st.markdown("### 💡 Suggested Follow-up Questions:")
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                if st.button(suggestion, key=f"suggestion_{message_index}_{suggestion_index}"):
                    st.session_state.active_suggestion = suggestion
                    st.rerun()
        elif item["type"] == "sql":
            with st.expander("📝 Generated SQL Query", expanded=False):
                st.code(item["statement"], language="sql")
            
            with st.expander("📊 Query Results", expanded=True):
                try:
                    with st.spinner("⚡ Executing SQL query..."):
                        df = pd.read_sql(item["statement"], st.session_state.snowflake_conn)
                        
                        if len(df.index) > 1:
                            data_tab, line_tab, bar_tab = st.tabs(["📋 Data", "📈 Line Chart", "📊 Bar Chart"])
                            
                            with data_tab:
                                st.dataframe(df, use_container_width=True)
                                
                                # Add download button
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv,
                                    file_name=f"cortex_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            if len(df.columns) > 1:
                                df_chart = df.set_index(df.columns[0])
                                with line_tab:
                                    st.line_chart(df_chart, use_container_width=True)
                                with bar_tab:
                                    st.bar_chart(df_chart, use_container_width=True)
                        else:
                            st.dataframe(df, use_container_width=True)
                            
                except Exception as e:
                    st.error(f"❌ Error executing SQL: {str(e)}")

def process_cortex_message(prompt: str) -> None:
    """Processes a message through Cortex Analyst and displays the response."""
    if 'cortex_messages' not in st.session_state:
        st.session_state.cortex_messages = []
    
    # Add user message
    st.session_state.cortex_messages.append(
        {"role": "user", "content": [{"type": "text", "text": prompt}], "timestamp": datetime.now()}
    )
    
    # Display user message
    with st.container():
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {prompt}</div>', unsafe_allow_html=True)
    
    # Get and display assistant response
    with st.container():
        st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
        st.markdown("**🤖 Cortex Analyst:**")
        
        with st.spinner("🧠 Cortex Analyst is thinking..."):
            response = send_cortex_message(prompt=prompt)
            
            if response:
                request_id = response["request_id"]
                content = response["message"]["content"]
                display_cortex_content(content=content, request_id=request_id)
                
                # Add assistant message to history
                st.session_state.cortex_messages.append(
                    {"role": "assistant", "content": content, "request_id": request_id, "timestamp": datetime.now()}
                )
            else:
                st.error("❌ Failed to get response from Cortex Analyst")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_conversation_history():
    """Display Cortex conversation history in sidebar."""
    if 'cortex_messages' in st.session_state and st.session_state.cortex_messages:
        with st.expander(f"🤖 Cortex Conversations ({len([msg for msg in st.session_state.cortex_messages if msg['role'] == 'user'])})"):
            for msg in st.session_state.cortex_messages[-10:]:  # Show last 5 conversations
                if msg['role'] == 'user':
                    st.caption(f"🔍 {msg['content'][0]['text'][:50]}...")
                    st.caption(f"_{msg['timestamp'].strftime('%H:%M:%S')}_")
                    st.markdown("---")

def init_session_state():
    """Initialize session state variables."""
    if 'cortex_messages' not in st.session_state:
        st.session_state.cortex_messages = []
    if 'active_suggestion' not in st.session_state:
        st.session_state.active_suggestion = None

def main():
    # Initialize session state
    init_session_state()
    
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🤖 AI-Powered Utilization Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Snowflake Cortex Analyst")
    
    # Initialize connections
    if not init_snowflake_connections():
        st.error("❌ Unable to connect to Snowflake. Please check your connection settings.")
        st.stop()
    
    # Sidebar with enhanced information
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Connection status
        if 'snowflake_conn' in st.session_state and st.session_state.snowflake_conn:
            st.markdown('<div class="success-card">✅ <strong>Cortex Analyst Ready</strong></div>', unsafe_allow_html=True)
            
            # Connection details
            with st.expander("📊 Configuration"):
                st.caption(f"**Account**: {ACCOUNT_SNOWFLAKE}")
                st.caption(f"**Database**: {CORTEX_DATABASE}")
                st.caption(f"**Schema**: {CORTEX_SCHEMA}")
                st.caption(f"**Stage**: {CORTEX_STAGE}")
                st.caption(f"**Semantic Model**: {CORTEX_FILE}")
                
            # Mode indicator
            st.markdown('<div class="info-card"><strong>Mode</strong>: 🤖 Cortex Analyst</div>', unsafe_allow_html=True)
            
            # Display conversation history
            display_conversation_history()
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.cortex_messages = []
                st.session_state.active_suggestion = None
                st.rerun()
                
        else:
            st.markdown('<div class="alert-card">❌ <strong>Not Connected</strong></div>', unsafe_allow_html=True)
        
        # Quick start guide
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        **Ask questions like:**
        - "Show me utilization trends for the last month"
        - "Which consultants are overallocated?"
        - "What's our billable rate this week?"
        - "Show project performance metrics"
        - "Compare forecast vs actual hours"
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🤖 Cortex Chat", "📊 Sample Questions", "⚙️ Setup Guide"])
    
    with tab1:
        st.markdown("### 💬 Chat with Cortex Analyst")
        st.markdown("Ask questions about your utilization data in natural language!")
        
        # Example questions as clickable buttons
        st.markdown("**🎯 Try these examples:**")
        
        col1, col2 = st.columns(2)
        
        example_questions = [
            "Show me the current utilization dashboard",
            "What are our utilization trends over the last 3 months?",
            "Which consultants have the highest billable rates?",
            "Show me project performance for this quarter",
            "Who is overallocated in the next 2 weeks?",
            "Compare our forecast accuracy by consultant"
        ]
        
        for i, question in enumerate(example_questions):
            col = col1 if i % 2 == 0 else col2
            if col.button(f"💡 Example {i+1}", help=question, key=f"example_{i}"):
                st.session_state.example_question = question
                st.rerun()
        
        # Chat interface
        if user_input := st.chat_input("Ask me anything about your utilization data..."):
            process_cortex_message(prompt=user_input)
        
        # Handle example question selection
        if hasattr(st.session_state, 'example_question'):
            process_cortex_message(prompt=st.session_state.example_question)
            del st.session_state.example_question
        
        # Handle suggestions
        if st.session_state.active_suggestion:
            process_cortex_message(prompt=st.session_state.active_suggestion)
            st.session_state.active_suggestion = None
        
        # Display conversation history
        if 'cortex_messages' in st.session_state:
            for message_index, message in enumerate(st.session_state.cortex_messages):
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"][0]["text"]}</div>', unsafe_allow_html=True)
                elif message["role"] == "assistant":
                    st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
                    st.markdown("**🤖 Cortex Analyst:**")
                    display_cortex_content(
                        content=message["content"],
                        request_id=message.get("request_id"),
                        message_index=message_index,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📝 Sample Questions for Your Utilization Data")
        
        categories = {
            "📊 Dashboard & Overview": [
                "Show me the current utilization dashboard",
                "What's our overall capacity utilization this week?",
                "Give me a summary of key utilization metrics",
                "How many active consultants do we have?"
            ],
            "📈 Trends & Analysis": [
                "Show utilization trends for the last 3 months",
                "How has our billable rate changed over time?",
                "What's the trend in our forecast accuracy?",
                "Compare this quarter vs last quarter utilization"
            ],
            "👥 People & Capacity": [
                "Which consultants are overallocated next week?",
                "Who has the highest billable rates?",
                "Show me individual consultant workloads",
                "Which team members have available capacity?"
            ],
            "💼 Projects & Performance": [
                "Show project performance metrics",
                "Which projects have the highest revenue?",
                "What's our project efficiency by client?",
                "Show me projects with low billable rates"
            ],
            "🎯 Forecasting": [
                "How accurate are our forecasts?",
                "Compare forecast vs actual hours",
                "Which consultants have the most variance in forecasting?",
                "Show forecast accuracy trends"
            ]
        }
        
        for category, questions in categories.items():
            with st.expander(category, expanded=False):
                for question in questions:
                    if st.button(question, key=f"sample_{question}"):
                        st.session_state.selected_sample = question
                        st.switch_page("🤖 Cortex Chat")  # This would need to be adjusted based on your tab structure
        
        if hasattr(st.session_state, 'selected_sample'):
            st.info(f"Selected: {st.session_state.selected_sample}")
            if st.button("🚀 Ask This Question"):
                # Switch to chat tab and ask the question
                process_cortex_message(prompt=st.session_state.selected_sample)
                del st.session_state.selected_sample
    
    with tab3:
        st.markdown("### ⚙️ Setup Guide for Cortex Analyst")
        
        st.markdown("""
        #### 📋 Prerequisites Checklist
        
        ✅ **Snowflake Account Setup**
        - Account with Cortex Analyst enabled
        - Appropriate role and warehouse permissions
        - Database and schema access
        
        ✅ **Required Components**
        - Semantic model file (`utilization_agent.yaml`)
        - Stage for storing the semantic model
        - Proper table permissions
        """)
        
        st.markdown("""
        #### 🎯 Semantic Model Configuration
        
        Your semantic model file should be uploaded to:
        ```
        @MERU_PREPARED.UTILIZATION_DASHBOARD.CORTEX_STAGE/utilization_agent.yaml
        ```
        """)
        
        st.code(f"""
-- Create stage if it doesn't exist
CREATE STAGE IF NOT EXISTS {CORTEX_DATABASE}.{CORTEX_SCHEMA}.{CORTEX_STAGE};

-- Upload your semantic model file
PUT file://path/to/utilization_agent.yaml @{CORTEX_DATABASE}.{CORTEX_SCHEMA}.{CORTEX_STAGE};

-- Verify the file is uploaded
LIST @{CORTEX_DATABASE}.{CORTEX_SCHEMA}.{CORTEX_STAGE};
        """, language="sql")
        
        st.markdown("""
        #### 🔧 Current Configuration
        """)
        
        config_data = {
            "Setting": ["Account", "Database", "Schema", "Stage", "Semantic Model", "Warehouse"],
            "Value": [ACCOUNT_SNOWFLAKE, CORTEX_DATABASE, CORTEX_SCHEMA, CORTEX_STAGE, CORTEX_FILE, WAREHOUSE_SNOWFLAKE]
        }
        
        st.dataframe(pd.DataFrame(config_data), use_container_width=True)
        
        if st.button("🔍 Test Cortex Connection"):
            if 'snowflake_conn' in st.session_state:
                test_response = send_cortex_message("Hello, can you help me understand my utilization data?")
                if test_response:
                    st.success("✅ Cortex Analyst connection successful!")
                    st.json(test_response)
                else:
                    st.error("❌ Connection test failed")
            else:
                st.error("❌ No active connection found")
    
    # Footer
    st.markdown("---")
    st.markdown("### 📈 Powered by Snowflake Cortex Analyst")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🤖 AI Features**
        - Natural language queries
        - Automatic SQL generation
        - Smart suggestions
        - Context-aware responses
        """)
    
    with col2:
        st.markdown("""
        **📊 Data Sources**
        - Utilization actuals
        - Forecast data
        - Capacity planning
        - Project performance
        """)
    
    with col3:
        st.markdown("""
        **🎯 Benefits**
        - No SQL knowledge required
        - Real-time insights
        - Interactive visualizations
        - Conversation-based analytics
        """)

if __name__ == "__main__":
    main()