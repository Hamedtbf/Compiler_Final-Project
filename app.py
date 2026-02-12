from flask import Flask, render_template, request, jsonify
import tempfile
import os
from plagiarism.cli import compare_hierarchies, compare_two_files, load_config

app = Flask(__name__)

if __name__ == '__main__':
    app.run(debug=True)