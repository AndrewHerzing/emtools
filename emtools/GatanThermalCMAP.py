# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 17:50:09 2016

@author: aherzing
"""
import matplotlib
thermal = matplotlib.colors.LinearSegmentedColormap.from_list('gatan_colormap',['black','blue','green','red','yellow','white'],256,1.0)

JustRed = matplotlib.colors.LinearSegmentedColormap.from_list('red_colormap',['black','red'],256,1.0)
JustGreen = matplotlib.colors.LinearSegmentedColormap.from_list('green_colormap',['black','green'],256,1.0)
JustBlue = matplotlib.colors.LinearSegmentedColormap.from_list('blue_colormap',['black','blue'],256,1.0)