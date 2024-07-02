import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import BaseServer
from urllib.parse import urlparse, parse_qs
from huggingface_hub import login

from logic import LlmLogicBrain

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):   
    def __init__(self, request, client_address, server) -> None:
        super().__init__(request, client_address, server)
       
        # Replace 'your_token_here' with your actual Hugging Face token
        hf_token = "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"
        login(token)
        
        brain = LlmLogicBrain(hf_token)
    
    def do_GET(self):
        # Check the path
        if self.path.startswith('/makethejump/bot'):
            
                        # Parse query parameters
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)
            
            # Check for 'prompt' in the query parameters
            if 'prompt' not in query_params:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'error': "'prompt' query parameter is required"}
                self.wfile.write(json.dumps(response).encode())
                return

            prompt = query_params['prompt'][0]
            
            response = self.brain.respond(prompt)
            
            response_data = {'response': response}
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'error': 'Not found'}
            self.wfile.write(json.dumps(response).encode())

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
