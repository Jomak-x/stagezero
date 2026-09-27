"""Bounded reference validation for the character worker."""
import io
from PIL import Image

def validate_reference(data):
    if not isinstance(data, bytes) or len(data) > 10 * 1024 * 1024:
        raise ValueError('Character reference image is too large')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'PNG' or image.width * image.height > 16_000_000:
                raise ValueError('Character reference must be a bounded PNG')
            image.verify()
    except (OSError, SyntaxError):
        raise ValueError('Character reference PNG is invalid') from None
