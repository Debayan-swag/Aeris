import http.server
import socketserver
import subprocess
import sys
import os
import threading
import webbrowser
from urllib.parse import parse_qs, urlparse

PORT = 8000
STREAMLIT_PORT = 8501

class AerisHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Handle launch request
        if parsed_path.path == '/launch':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Launch Streamlit in a separate thread with FAST LOADING optimizations
            def launch_streamlit():
                script_dir = os.path.dirname(os.path.abspath(__file__))
                gui_path = os.path.join(script_dir, "GUI.py")
                
                # Launch with ALL performance optimizations
                subprocess.Popen([
                    sys.executable, 
                    "-m", 
                    "streamlit", 
                    "run", 
                    gui_path,
                    "--server.port=" + str(STREAMLIT_PORT),
                    "--server.headless=true",
                    "--browser.gatherUsageStats=false",
                    "--server.fileWatcherType=none",
                    "--server.runOnSave=false",
                    "--client.showErrorDetails=false",
                    "--client.toolbarMode=minimal",
                    "--runner.fastReruns=true",
                    "--runner.enforceSerializableSessionState=false",
                    "--logger.level=error",
                    "--logger.messageFormat=%(message)s",
                    "--global.developmentMode=false",
                    "--global.showWarningOnDirectExecution=false",
                    "--server.maxUploadSize=200",
                    "--server.maxMessageSize=200"
                ])
            
            threading.Thread(target=launch_streamlit, daemon=True).start()
            
            response = b'{"status": "launched", "url": "http://localhost:' + str(STREAMLIT_PORT).encode() + b'"}'
            self.wfile.write(response)
            return
        
        # Serve static files
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), AerisHandler) as httpd:
        print(f"Aeris server running at http://localhost:{PORT}/")
        print(f"Open http://localhost:{PORT}/index.html in your browser")
        print("Press Ctrl+C to stop the server")
        
        # Auto-open browser
        webbrowser.open(f"http://localhost:{PORT}/index.html")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    main()
