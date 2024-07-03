import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from huggingface_hub import login
# from logic import LlmLogicBrain
from hf_logic import HfLlmLogicBrain
from dotenv import load_dotenv
import os

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    # Class attribute for brain
    brain = None
    load_dotenv()

    @classmethod
    def initialize_brain(cls):
        # Replace 'your_token_here' with your actual Hugging Face token
        hf_token = os.getenv('HF_TOKEN')  # Ensure to provide the actual token here if not using environment variable
        login(hf_token)
        cls.brain = HfLlmLogicBrain(hf_token)

    def do_GET(self):
        # Check if brain is initialized
        if SimpleHTTPRequestHandler.brain is None:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,PUT,DELETE')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            self.send_header('Access-Control-Allow-Credentials', 'true')
            self.end_headers()
            response = {'error': 'Brain not initialized'}
            self.wfile.write(json.dumps(response).encode())
            return

        # Check the path
        if self.path.startswith('/makethejump/bot'):
            # Parse query parameters
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)
            
            # Check for 'prompt' in the query parameters
            if 'prompt' not in query_params:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,PUT,DELETE')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type,Authorization')
                self.send_header('Access-Control-Allow-Credentials', 'true')
                self.end_headers()
                response = {'error': "'prompt' query parameter is required"}
                self.wfile.write(json.dumps(response).encode())
                return

            prompt = query_params['prompt'][0]
            
            print(f'Prompt: {prompt}')
            response = self.brain.respond(prompt)
            print(f'Response: {response}')
            
            response_data = {'response': response}
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,PUT,DELETE')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            self.send_header('Access-Control-Allow-Credentials', 'true')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS,PUT,DELETE')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            self.send_header('Access-Control-Allow-Credentials', 'true')
            self.end_headers()
            response = {'error': 'Not found'}
            self.wfile.write(json.dumps(response).encode())

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    handler_class.initialize_brain()
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
