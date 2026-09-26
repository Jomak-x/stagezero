"""Neon reference image → our private GPU worker → textured character GLB."""
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import requests

from character_generation import validate_glb, MAX_GLB_BYTES
from character_reference import validate_reference


class SelfHostedCharacterGenerator:
    source = 'neon-trellis'

    def __init__(self, reference_generator, token, url='http://127.0.0.1:8770', *,
                 transport=None, clock=None, sleep=None, deadline=1800, poll_interval=2):
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/')):
            raise ValueError('Character worker must use a local SSH tunnel')
        if not isinstance(token, str) or not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise ValueError('Character worker authentication is missing')
        if (type(deadline) not in (int, float) or not 0 < deadline <= 7200 or
                not math.isfinite(deadline) or type(poll_interval) not in (int, float) or
                not 0 < poll_interval <= 60 or not math.isfinite(poll_interval)):
            raise ValueError('Invalid character worker polling limits')
        self.reference_generator = reference_generator
        self._token = token
        self.url = url.rstrip('/')
        self.transport = transport or requests
        self.clock, self.sleep = clock or time.monotonic, sleep or time.sleep
        self.deadline, self.poll_interval = deadline, poll_interval
        self.reference_image = None
        self.on_reference = None
        self._seeded_reference = None

    def use_reference(self, prompt, png):
        """Seed one matching generation with a previously completed design image."""
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 800:
            raise ValueError('Describe the character in 1–800 characters')
        validate_reference(png)
        self._seeded_reference = (prompt.strip(), png)

    @classmethod
    def from_env(cls):
        from neon_character_reference import NeonCharacterReference
        token_path = Path(__file__).resolve().parent / '.runtime/api-token'
        try:
            token = token_path.read_text().strip()
        except OSError:
            raise ValueError('Character worker is not configured; run the character backend launcher') from None
        return cls(NeonCharacterReference.from_env(), token,
                   os.environ.get('STAGEZERO_CHARACTER_BACKEND_URL', 'http://127.0.0.1:8770'))

    def _request(self, method, path, *, data=None, limit=128000, check=None):
        try:
            remaining = check() if check is not None else None
            timeout = (min(5, remaining), min(45, remaining)) if remaining is not None else (5, 45)
            with getattr(self.transport, method)(self.url + path,
                    headers={'Authorization': 'Bearer ' + self._token, 'Content-Type': 'image/png'},
                    data=data, timeout=timeout, stream=True, allow_redirects=False) as response:
                if response.status_code == 409:
                    raise ValueError('Character GPU is busy; wait for its current generation to finish')
                if response.status_code in (401, 403):
                    raise ValueError('Character worker authentication failed; check the private tunnel configuration')
                if not 200 <= response.status_code < 300:
                    raise ValueError(f'Character worker returned HTTP {response.status_code}')
                body = bytearray()
                for chunk in response.iter_content(65536):
                    # Finish reading job creation so its ID can be cancelled on failure.
                    if check is not None and method != 'post':
                        check()
                    body.extend(chunk)
                    if len(body) > limit:
                        raise ValueError('Character worker response is too large')
                return bytes(body)
        except requests.RequestException:
            raise ValueError('Cannot reach the character GPU; start its backend and private tunnel') from None

    def _json(self, method, path, data=None, *, check=None):
        try:
            doc = json.loads(self._request(method, path, data=data, check=check))
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError('Character worker returned an invalid response') from None
        if not isinstance(doc, dict):
            raise ValueError('Character worker returned an invalid response')
        return doc

    def generate(self, prompt, progress=lambda message: None, cancelled=lambda: False):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 800:
            raise ValueError('Describe the character in 1–800 characters')
        expires = self.clock() + self.deadline
        job = None

        def check():
            if cancelled():
                raise ValueError('Character generation stopped')
            remaining = expires - self.clock()
            if remaining <= 0:
                raise ValueError('Character generation timed out')
            return remaining

        try:
            check()
            health = self._json('get', '/health', check=check)
            if health.get('ready') is not True:
                raise ValueError('Character GPU is still being set up; try again when it is ready')
            if health.get('busy'):
                raise ValueError('Character GPU is busy; wait for its current generation to finish')
            seeded = self._seeded_reference
            self._seeded_reference = None
            if seeded is not None and seeded[0] == prompt.strip():
                progress('1 / 3 · Reusing your completed design…')
                self.reference_image = seeded[1]
            else:
                progress('1 / 3 · Designing the character with Neon…')
                self.reference_image = self.reference_generator.generate(prompt.strip(), progress=progress, cancelled=cancelled)
            check()
            if not isinstance(self.reference_image, bytes) or len(self.reference_image) > 10 * 1024 * 1024:
                raise ValueError('Character reference exceeds the GPU worker image limit')
            if self.on_reference is not None:
                self.on_reference(self.reference_image)
            progress('2 / 3 · Building geometry and textures on our GPU…')
            doc = self._json('post', '/jobs', data=self.reference_image, check=check)
            job = doc.get('id')
            if not isinstance(job, str) or not re.fullmatch('[a-f0-9]{32}', job):
                job = None
                raise ValueError('Character worker returned an invalid job identifier')
            while True:
                check()
                doc = self._json('get', '/jobs/' + job, check=check)
                status = doc.get('status')
                if status == 'succeeded':
                    break
                if status in ('failed', 'cancelled'):
                    if status == 'failed' and doc.get('error') == 'GPU memory busy':
                        raise ValueError('The shared GPU is still busy; try again when other GPU work finishes')
                    raise ValueError('Character GPU could not finish this model; inspect the worker log or try another description')
                if status not in ('queued', 'generating'):
                    raise ValueError('Character worker returned an unknown job status')
                stage = {
                    'checking_gpu': 'Checking GPU memory',
                    'waiting_gpu_memory': 'Waiting for the shared GPU to have enough free memory',
                    'gpu_available': 'Preparing the reconstruction model',
                    'loading_pipeline': 'Loading the reconstruction model',
                    'generating': 'Reconstructing geometry and materials',
                    'generating_phase': 'Reconstructing geometry and materials',
                    'removing_reference_background': 'Preparing reference detail',
                    'baking_texture': 'Baking detailed textures',
                    'enhancing_front_texture': 'Restoring face and clothing detail',
                    'exporting': 'Exporting the textured mesh',
                    'complete': 'Finishing the character',
                }.get(doc.get('stage'))
                if stage:
                    progress('2 / 3 · ' + stage + '…')
                self.sleep(min(self.poll_interval, max(0, expires - self.clock())))
            check()
            progress('3 / 3 · Loading the textured 3D character…')
            data = self._request('get', '/jobs/' + job + '/result', limit=MAX_GLB_BYTES, check=check)
            check()
            return validate_glb(data)
        except Exception:
            if job is not None:
                try:
                    self._request('delete', '/jobs/' + job)
                except (ValueError, OSError):
                    pass
            raise
