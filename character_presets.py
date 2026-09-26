"""Bundled, previously generated characters that are ready without a GPU service."""
from pathlib import Path

ASSET_FOLDER = Path(__file__).resolve().parent / 'assets' / 'character-presets'
PRESETS = (
    {'key': 'explorer', 'name': 'Explorer', 'detail': 'Leather jacket · adventure ready'},
    {'key': 'mechanic', 'name': 'Space mechanic', 'detail': 'Silver hair · orange flight suit'},
    {'key': 'ranger', 'name': 'Park ranger', 'detail': 'Forest jacket · hiking boots'},
)


def preset_data(key):
    if key not in {preset['key'] for preset in PRESETS}:
        raise ValueError('Unknown character preset')
    return (ASSET_FOLDER / (key + '.glb')).read_bytes(), preset_reference(key)


def preset_reference(key):
    if key not in {preset['key'] for preset in PRESETS}:
        raise ValueError('Unknown character preset')
    return (ASSET_FOLDER / (key + '.png')).read_bytes()
