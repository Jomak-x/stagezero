"""Scene-wide light accents and deterministic effect rendering."""
import numpy as np
from scene_effects import EffectSceneLayer
from scene_composition import LIGHTING

PALETTES = {
    'neutral': ((205, 223, 255), (255, 225, 195), .0),
    'warm': ((255, 180, 105), (255, 222, 170), 13.),
    'moonlight': ((107, 152, 255), (135, 230, 212), 16.),
    'neon': ((63, 211, 255), (224, 91, 255), 22.),
    'sunset': ((255, 141, 76), (138, 156, 235), 18.),
}


class SceneAtmosphereLayer:
    def __init__(self, server):
        self.server = server
        self.effects = EffectSceneLayer(server)
        self.signature = None
        self.lights = [server.scene.add_light_point('/scene-light/key', position=(-3, 3.5, 1), intensity=0., distance=14.),
                       server.scene.add_light_point('/scene-light/rim', position=(3, 3, -2), intensity=0., distance=14.)]

    def update(self, effects, seconds, lighting='neutral'):
        if lighting not in LIGHTING:
            raise ValueError('Unknown lighting preset')
        if lighting != self.signature:
            a, b, strength = PALETTES[lighting]
            for handle, color in zip(self.lights, (a, b)):
                handle.color = color
                handle.intensity = strength
            if hasattr(self.server.scene, 'set_background_image'):
                skies = {'neutral': ((19,26,36),(43,55,66)),
                         'warm': ((28,30,36),(66,60,54)),
                         'sunset': ((32,39,60),(99,70,57)),
                         'moonlight': ((10,17,39),(42,65,84)),
                         'neon': ((13,13,32),(40,30,67))}
                top, bottom = skies[lighting]
                blend = np.linspace(0,1,256)[:,None,None]
                image = np.repeat((np.array(top)[None,None,:]*(1-blend)+np.array(bottom)[None,None,:]*blend).astype(np.uint8), 8, axis=1)
                self.server.scene.set_background_image(image)
            self.signature = lighting
        self.effects.update(effects, seconds)
