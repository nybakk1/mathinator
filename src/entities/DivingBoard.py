# -*- coding: utf-8 -*-


class DivingBoard:
    """Young’s modulus of the wood."""
    E = 1.3 * (10 ** 10)

    def __init__(self, length, width, thickness):
        """
        Construct a diving board object.
        :param length: the length of the board in meters.
        :param width: the width of the board in meters.
        :param thickness: the thickness of the board in meters
        """
        self.length = length
        self.width = width
        self.thickness = thickness

        """Area moment of inertia around the center of mass of a beam"""
        self.I = (self.width * (self.thickness ** 3)) / 12
