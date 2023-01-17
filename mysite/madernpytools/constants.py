"""Martijn Zeestraten, Maart 2018"""
from enum import Enum

spectral_keys = ['Sxx', 'Sxf', 'Sff']


class Errors(Enum):
    no_error = 0
    file_exists = 1

