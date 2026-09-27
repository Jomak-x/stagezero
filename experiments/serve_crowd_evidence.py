"""Loopback-only evidence sink for real browser crowd benchmarks and captures."""
import argparse
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import json
from pathlib import Path
import re
from urllib.parse import urlparse,parse_qs

class EvidenceHandler(SimpleHTTPRequestHandler):
    output:Path
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.output), **kwargs)
    def end_headers(self):
        origin=self.headers.get('Origin','')
        if origin=='http://127.0.0.1:24970':self.send_header('Access-Control-Allow-Origin',origin)
        super().end_headers()
    def do_OPTIONS(self):
        self.send_response(204);self.send_header('Access-Control-Allow-Methods','POST, OPTIONS');self.send_header('Access-Control-Allow-Headers','Content-Type');self.end_headers()
    def do_POST(self):
        if self.headers.get('Origin')!='http://127.0.0.1:24970':self.send_error(403);return
        u=urlparse(self.path)
        if u.path not in ('/report','/video'):self.send_error(404);return
        try:n=int(self.headers.get('Content-Length','0'))
        except ValueError:self.send_error(400);return
        if not 0<n<=256*1024*1024:self.send_error(413);return
        raw=self.rfile.read(n)
        if len(raw)!=n:self.send_error(400);return
        name=parse_qs(u.query).get('name',['report' if u.path=='/report' else 'playback'])[0]
        name=re.sub(r'[^a-zA-Z0-9._-]','_',name)[:120]
        if u.path=='/report':
            try:data=json.loads(raw)
            except (ValueError,UnicodeDecodeError):self.send_error(400);return
            raw=(json.dumps(data,indent=2)+'\n').encode();suffix='.json'
        else:suffix='.webm'
        if not name.endswith(suffix):name+=suffix
        dest=self.output/name
        if dest.exists():
            import time
            dest=self.output/(dest.stem+'-'+str(time.time_ns())+suffix)
        dest.write_bytes(raw)
        self.send_response(201);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps({'saved':dest.name,'bytes':len(raw)}).encode())
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,default=24973);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);EvidenceHandler.output=a.output.resolve();print(f'Crowd evidence listening on 127.0.0.1:{a.port}',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),EvidenceHandler).serve_forever()
