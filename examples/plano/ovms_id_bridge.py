# examples/plano/ovms_id_bridge.py
"""A lightweight bridge that adds a top-level `"id"` field to OVMS responses.

Plano (planoai) strictly requires a top-level `"id"` field in chat.completion
responses. OVMS omits this field. This bridge listens on port 8001, forwards
requests to OVMS on port 8002, and injects `"id": "chatcmpl-ovms"` if missing.
"""
import json
import urllib.request
import argparse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler


#: Where OVMS listens. ONE definition, read by both the class default and the
#: --target-port flag, because they disagreed: the class said 8000 while the
#: flag defaulted to 8002, so importing this handler without running main()
#: silently forwarded to the wrong port. 8002 rather than 8000 because plano
#: takes 8000 as the client-facing listener and OVMS moves out of its way
#: (model.ovms_port in the workflow).
OVMS_PORT = 8002


class OVMSBridgeHandler(BaseHTTPRequestHandler):
    target_port = OVMS_PORT

    # HTTP/1.1 because Envoy (what plano runs on) treats an HTTP/1.0 upstream
    # differently and stopped reading long bodies part-way: a 5 KB answer
    # arrived as 2.8 KB, cut mid-string, and the JSON would not parse. Short
    # answers fitted and looked fine, which is what made it look like a plano
    # bug rather than a protocol mismatch.
    #
    # This line only works paired with ThreadingHTTPServer below. HTTP/1.1
    # keeps connections ALIVE, and a single-threaded server serves one
    # connection to exhaustion, so Envoy's pooled connection would hold the
    # bridge and every later request would queue behind it forever.
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
        
        # Forward request to OVMS on target port
        target_url = f"http://127.0.0.1:{self.target_port}{self.path}"
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
    # Threading, not plain HTTPServer: see protocol_version above. Envoy holds
    # a pooled keep-alive connection open, and a single-threaded server would
    # sit inside that one connection's handler loop and never accept another.
    # Symptom is not an error but a HANG, with each request slower than the
    # last as they queue: 5s, 33s, 64s, 100s, then never.
    # Loopback by DEFAULT. This used to bind 0.0.0.0 unconditionally, which
    # put an unauthenticated door to the GPU on every interface: anyone on the
    # same LAN or coffee-shop wifi could POST to /v3/chat/completions and spend
    # your hardware. Nothing here checks credentials, because nothing here was
    # ever meant to be reachable from off-box.
    #
    # It is a FLAG rather than a hardcoded 127.0.0.1 because the wider bind is
    # genuinely needed in the setup this example documents: with plano in WSL2
    # or Docker and OVMS on the Windows host, the bridge has to be reachable
    # from another network namespace, and loopback is not. So that case opts in
    # explicitly, and everyone else is closed by default.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind. Default 127.0.0.1 (this "
                             "machine only). Use 0.0.0.0 ONLY when plano runs "
                             "in WSL2/Docker and must reach this host across a "
                             "network namespace - it exposes an "
                             "unauthenticated endpoint to your whole network.")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--target-port", type=int, default=OVMS_PORT,
                        help=f"Port where OVMS is listening. "
                             f"Default {OVMS_PORT}.")
    args = parser.parse_args()

    if args.host == "0.0.0.0":
        print("WARNING: binding 0.0.0.0 exposes this unauthenticated bridge to "
              "your entire network. Firewall the port, or prefer 127.0.0.1.")
    OVMSBridgeHandler.target_port = args.target_port
    server = ThreadingHTTPServer((args.host, args.port), OVMSBridgeHandler)
    print(f"OVMS ID Bridge listening on http://{args.host}:{args.port} "
          f"-> forwarding to :{args.target_port}...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping bridge.")


if __name__ == "__main__":
    main()
