from flask import Flask, render_template, request, jsonify
import tempfile
import os
from plagiarism.cli import compare_hierarchies, compare_two_files, load_config

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    code_a = data.get('code_a', '')
    code_b = data.get('code_b', '')
    mode = data.get('mode', 'basic') # 'basic' or 'hierarchy'
    
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f2:
        f1.write(code_a.encode('utf-8'))
        f2.write(code_b.encode('utf-8'))
        path_a, path_b = f1.name, f2.name

    try:
        config = load_config(None)
        if mode == 'hierarchy':
            report = compare_hierarchies(path_a, path_b, config)
        else:
            report = compare_two_files(path_a, path_b, config)
        
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in [path_a, path_b]:
            if os.path.exists(p): os.remove(p)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')