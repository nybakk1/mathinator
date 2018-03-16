# -*- coding: utf-8 -*-


class DivingBoard:
    """Young’s modulus of the wood."""
    E = 1.3 * (10 ** 10)

    def __init__(self, length, width, thickness):
        self.length = length
        self.width = width
        self.thickness = thickness
        self.I = (self.width * (self.thickness ** 3)) / 12
