import os
import uuid
import re
from flask import Flask, render_template, request, jsonify, session, url_for
from werkzeug.utils import secure_filename
from langchain_core.messages import ToolMessage
from .agent import create_agent_executor
from .tools import execute_python_in_sandbox

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

GLOBAL_SESSION_ID = "persistent_app_session"
session_agents = {}

def get_agent(session_id: str):
    try:
        if session_id not in session_agents:
            session_agents[session_id] = create_agent_executor()
        return session_agents[session_id]
    except Exception as e:
        print(f"Error creating agent for session {session_id}: {str(e)}")
        return None

@app.route('/')
def index():
    session['session_id'] = GLOBAL_SESSION_ID
    return render_template('index.html', session_id=GLOBAL_SESSION_ID)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    session_id = GLOBAL_SESSION_ID
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        import builtins
        if not hasattr(builtins, '_session_store'):
            builtins._session_store = {}
        
        # Load the dataframe first
        import pandas as pd
        df = pd.read_csv(filepath)
        
        # Generate comprehensive data profile using hybrid profiler
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
            from intelligence.hybrid_data_profiler import generate_dataset_profile_for_agent
            
            print(f"Generating comprehensive profile for {filename}...")
            data_profile = generate_dataset_profile_for_agent(
                df, 
                context={'filename': filename, 'upload_session': session_id}
            )
            
            # Store both dataframe and profile in session
            builtins._session_store[session_id] = {
                'dataframe': df,
                'data_profile': data_profile,
                'filename': filename
            }
            
            # Create rich summary for agent
            profile_summary = f"""Dataset '{filename}' successfully loaded and analyzed:

DATASET OVERVIEW:
• Shape: {data_profile.dataset_insights.total_records:,} rows × {data_profile.dataset_insights.total_features} columns
• Data Quality Score: {data_profile.dataset_insights.data_quality_score}/100 ({data_profile.dataset_insights.overall_health})
• Missing Data: {data_profile.dataset_insights.missing_data_percentage}% of cells
• Anomalies Detected: {data_profile.dataset_insights.anomaly_count}
• Data Freshness: {data_profile.dataset_insights.data_freshness}

COLUMN ANALYSIS:
• Numeric Columns: {len([col for col, info in data_profile.ai_agent_context['column_details'].items() if 'int' in info['dtype'] or 'float' in info['dtype']])}
• Categorical Columns: {len([col for col, info in data_profile.ai_agent_context['column_details'].items() if 'object' in info['dtype']])}
• Key Columns: {', '.join(data_profile.dataset_insights.key_columns) if data_profile.dataset_insights.key_columns else 'None detected'}
• Temporal Columns: {', '.join(data_profile.dataset_insights.temporal_columns) if data_profile.dataset_insights.temporal_columns else 'None detected'}
• High Cardinality: {', '.join(data_profile.dataset_insights.high_cardinality_columns[:3]) if data_profile.dataset_insights.high_cardinality_columns else 'None'}

DETECTED DOMAINS: {', '.join(data_profile.dataset_insights.detected_domains) if data_profile.dataset_insights.detected_domains else 'General purpose data'}

TOP RECOMMENDATIONS:
{chr(10).join(f'• {rec}' for rec in data_profile.recommendations[:5])}

COLUMNS AVAILABLE:
{', '.join(data_profile.ai_agent_context['dataset_overview']['column_names'])}

The dataset is now ready for analysis. You can create visualizations, perform statistical analysis, or explore relationships in the data."""
            
            print(f"Data profile generated successfully. Quality score: {data_profile.dataset_insights.data_quality_score}")
            
        except Exception as e:
            print(f"Profile generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to basic storage
            builtins._session_store[session_id] = {
                'dataframe': df,
                'filename': filename
            }
            profile_summary = f"Dataset '{filename}' loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns. Columns: {', '.join(df.columns.tolist())}"
        
        # Load data into sandbox
        initial_code = f"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('{filepath}')
print("✅ Data loaded successfully!")
print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{', '.join(df.columns.tolist())}}")
print("\\nDataset is ready for analysis and visualization.")
"""
        
        result = execute_python_in_sandbox(code=initial_code, session_id=session_id)
        
        return jsonify({
            'status': 'success',
            'message': profile_summary,
            'filename': filename,
            'dataset_info': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict()
            }
        })
    
    return jsonify({'error': 'Upload failed'}), 500

@app.route('/api/agent/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    session_id = GLOBAL_SESSION_ID
    
    agent = get_agent(session_id)
    
    if not agent:
        return jsonify({"status": "error", "detail": "Failed to initialize agent. Check server logs for details."}), 500

    try:
        input_state = {
            "messages": [("user", user_message)], 
            "session_id": session_id,
            "python_executions": 0
        }
        final_response_content = ""
        all_plot_urls = []

        for chunk in agent.stream(input_state, config={"configurable": {"thread_id": session_id}}):
            # Chunk is a dictionary where keys are node names
            print(f"DEBUG: Received chunk with keys: {list(chunk.keys())}")
            
            if "action" in chunk:
                # Tool execution step
                print("DEBUG: Processing tool execution")
                tool_messages = chunk["action"]["messages"]
                for tool_msg in tool_messages:
                    if isinstance(tool_msg, ToolMessage) and tool_msg.name == "python_code_interpreter":
                        print(f"DEBUG: Tool result: {tool_msg.content[:200]}...")
                        try:
                            if "Generated" in tool_msg.content:
                                plot_files = re.findall(r"plot[\w-]+\.png", tool_msg.content)
                                for pf in plot_files:
                                    url = url_for("static", filename=f"plots/{pf}")
                                    if url not in all_plot_urls:
                                        all_plot_urls.append(url)
                        except Exception as e:
                            pass
            if "agent" in chunk:
                print("DEBUG: Processing agent response")
                final_message = chunk["agent"]["messages"][-1]
                if final_message:
                    final_response_content = final_message.content
                    print(f"DEBUG: Final response: {final_response_content[:100]}...")
        if not final_response_content:
            final_state = agent.invoke(input_state, config={"configurable": {"thread_id": session_id}})
            final_response_content = final_state["messages"][-1].content
        return jsonify({
            "status": "success",
            'response': final_response_content,
            'plots': all_plot_urls
        })
    except Exception as e:
        return jsonify({"status":"error", "detail":f"Chat error: {str(e)}"}), 500

if __name__ == '__main__':
    os.makedirs(os.path.join('static', app.config['GENERATED_PLOTS_FOLDER']), exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)