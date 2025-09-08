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

session_agents = {}

def get_agent(session_id: str):
    if session_id not in session_agents:
        session_agents[session_id] = create_agent_executor()
    return session_agents[session_id]

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    return render_template('index.html', session_id=session['session_id'])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    session_id = session.get('session_id')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and session_id:
        filename = secure_filename(file.filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        initial_code = f"""
                        import pandas as pd
                        df = pd.read_csv('{filepath}')
                        print("Data loaded. Here's an overview:")
                        df.info()
                        """
        result = execute_python_in_sandbox(code=initial_code, session_id=session_id)
        result_str = f"STDOUT: {result.get('stdout', '')}\nSTDERR: {result.get('stderr', '')}"
        
        agent = get_agent(session_id)
        initial_messages = [
            ("user", f"The data from '{filename}' has been loaded. Please provide a summary."),
            ToolMessage(content=result_str, tool_call_id="initial_load")
        ]
        
        response = agent.invoke(
            {"messages": initial_messages, "session_id": session_id},
            config={"configurable": {"thread_id": session_id}}
        )
        return jsonify({
            'status': 'success',
            'message': response['messages'][-1].content,
            'filename': filename
        })
    
    return jsonify({'error': 'Upload failed'}), 500

@app.route('/api/agent/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    session_id = data.get('session_id') or session.get('session_id')
    
    if not session_id:
       session_id = str(uuid.uuid4())
       session["session_id"] = session_id    
    agent = get_agent(session_id)
    
    try:
        input_messages = [{"messages":[("user", user_message)]}]
        final_response_content = ""
        all_plot_urls = []

        for chunk in agent.stream(input_messages, config={"configurable": {"thread_id": session_id}}):
            # Chunk is a dictionary where keys are node names
            if "action" in chunk:
                # Tool execution step
                tool_messages = chunk["action"]["messages"]
                for tool_msg in tool_messages:
                    if isinstance(tool_msg, ToolMessage) and tool_msg.name == "python_code_interpreter":
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
                final_message = chunk["agent"]["messages"][-1]
                if final_message:
                    final_response_content = final_message.content
        if not final_response_content:
            final_state = agent.invoke(input_messages, config={"configurable": {"thread_id": session_id}})
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