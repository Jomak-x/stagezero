"""Offline containment tests for the current generated-character GLB validator."""
import json
import struct
import unittest

from character_generation import MAX_GLB_BYTES, validate_glb


def glb(document=None, binary=b'\0\0\0\0'):
    document = document or {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': len(binary)}]}
    encoded = json.dumps(document).encode()
    encoded += b' ' * (-len(encoded) % 4)
    chunks = struct.pack('<I4s', len(encoded), b'JSON') + encoded
    if binary is not None:
        chunks += struct.pack('<I4s', len(binary), b'BIN\0') + binary
    return struct.pack('<4sII', b'glTF', 2, 12 + len(chunks)) + chunks


class GLBValidationTests(unittest.TestCase):
    def test_valid_embedded_buffer_and_images(self):
        doc = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 4}],
               'bufferViews': [{'buffer': 0, 'byteLength': 4}],
               'images': [{'bufferView': 0, 'mimeType': 'image/png'}]}
        data = glb(doc)
        self.assertIs(validate_glb(data), data)
        data = glb({'asset': {'version': '2.0'},
                    'buffers': [{'byteLength': 1, 'uri': 'data:application/octet-stream;base64,AA=='}],
                    'images': [{'uri': 'data:image/png;base64,AA=='}]}, binary=None)
        self.assertIs(validate_glb(data), data)

    def test_rejects_malformed_or_external_glb(self):
        valid = glb()
        bad = [b'bad', valid[:-1], valid[:8] + struct.pack('<I', len(valid) + 4) + valid[12:],
               valid[:4] + struct.pack('<I', 1) + valid[8:],
               glb({'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 4, 'uri': 'https://evil/b.bin'}]}, binary=None),
               glb({'asset': {'version': '2.0'}, 'images': [{'uri': '../texture.png'}]}, binary=None),
               glb({'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 100}]}, binary=b'\0' * 4),
               glb({'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 4}],
                    'bufferViews': [{'buffer': 0, 'byteLength': 8}]}),
               b'\0' * (MAX_GLB_BYTES + 1)]
        for blob in bad:
            with self.subTest(length=len(blob)), self.assertRaises(ValueError):
                validate_glb(blob)

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        for encoded in (b'{"asset":{"version":"2.0"},"asset":{"version":"2.0"}}',
                        b'{"asset":{"version":"2.0"},"value":NaN}'):
            encoded += b' ' * (-len(encoded) % 4)
            data = struct.pack('<4sII', b'glTF', 2, 20 + len(encoded))
            data += struct.pack('<I4s', len(encoded), b'JSON') + encoded
            with self.assertRaises(ValueError):
                validate_glb(data)


if __name__ == '__main__':
    unittest.main()
