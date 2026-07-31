# examples/plano/ovms_id_bridge.py
"""A lightweight bridge that adds a top-level `"id"` field to OVMS responses.

Plano (planoai) strictly requires a top-level `"id"` field in chat.completion
responses. OVMS omits this field. This bridge listens on port 8001, forwards
requests to OVMS on port 8000, and injects `"id": "chatcmpl-ovms"` if missing.
"""
import json
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler


class OVMSBridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        # Handle both Content-Length and Transfer-Encoding: chunked from Plano
        transfer_encoding = self.headers.get("Transfer-Encoding", "").lower()
        if "chunked" in transfer_encoding:
            chunks = []
            while True:
                line = self.rfile.readline().strip()
                if not line:
                    break
                try:
                    chunk_len = int(line.split(b";")[0], 16)
                except ValueError:
                    break
                if chunk_len == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(chunk_len))
                self.rfile.readline()
            body = b"".join(chunks)
        else:
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
        
        # Forward request to OVMS on port 8000
        target_url = f"http://127.0.0.1:8000{self.path}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "OpenAI/Python 2.51.0",
            "Accept": "application/json",
        }
        req = urllib.request.Request(target_url, data=body, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_bytes = resp.read()
                try:
                    data = json.loads(resp_bytes.decode("utf-8"))
                    if isinstance(data, dict) and "id" not in data:
                        data["id"] = "chatcmpl-ovms-bridge"
                    out_bytes = json.dumps(data).encode("utf-8")
                except Exception:
                    out_bytes = resp_bytes

                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out_bytes)))
                self.end_headers()
                self.wfile.write(out_bytes)
        except urllib.error.HTTPError as exc:
            err_body = exc.read()
            print(f"[Bridge Error] OVMS HTTP {exc.code}: {err_body.decode('utf-8', errors='ignore')}")
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as exc:
            print(f"[Bridge Error] {exc}")
            err_body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)

    def log_message(self, format, *args):
        pass  # Quiet logging


def main():
    server = HTTPServer(("0.0.0.0", 8001), OVMSBridgeHandler)
    print("OVMS ID Bridge listening on http://0.0.0.0:8001 -> forwarding to :8000...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping bridge.")


if __name__ == "__main__":
    main()
