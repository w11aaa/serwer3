import hashlib as Hashlib
import imageio as ImageIO
import shutil as Shutil
import tempfile as Temp
try:
    import streamlit as App
except ImportError:
    class MockApp:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
        def markdown(self, *args, **kwargs): pass
        def error(self, msg): print(f"ERROR: {msg}")
        def success(self, msg): print(f"SUCCESS: {msg}")
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    App = MockApp()

import cv2 as CV2
import os as OS
import numpy as NP
try:
    import plotly.express as Chart
except ImportError:
    class MockChart:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    Chart = MockChart()

import matplotlib.pyplot as Plt
try:
    import torch as Torch
except ImportError:
    class MockTorch:
        def __getattr__(self, name): return lambda *args, **kwargs: None
        class cuda:
            @staticmethod
            def is_available(): return False
    Torch = MockTorch()

import zipfile as Zip
from PIL import Image
try:
    from ultralytics import YOLO
except ImportError:
    class YOLO:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return []
        def export(self, *args, **kwargs): return "mock_path"
        def to(self, *args, **kwargs): pass
from io import BytesIO as BIO
try:
    from streamlit_pdf_viewer import pdf_viewer as PDF
except ImportError:
    PDF = lambda *args, **kwargs: None

from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.drawing.image import Image as XLImage
from pandas import Series, DataFrame, cut
from openpyxl.worksheet.table import Table
from openpyxl.worksheet.table import TableStyleInfo

import sys as Sys
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table as PDFTable, TableStyle, Image as PDFImage, Paragraph, Spacer
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    A4 = (595.27, 841.89)
    class MockColors:
        def __getattr__(self, name): return "#000000"
        def HexColor(self, val): return val
    colors = MockColors()
    def getSampleStyleSheet(): 
        class MockStyle:
            def __init__(self): self.name = "Normal"
        return {"Title": MockStyle(), "Normal": MockStyle()}
    class SimpleDocTemplate:
        def __init__(self, *args, **kwargs): pass
        def build(self, *args, **kwargs): pass
    class ParagraphStyle:
        def __init__(self, *args, **kwargs): pass
    class pdfmetrics:
        @staticmethod
        def registerFont(*args, **kwargs): pass
    class TTFont:
        def __init__(self, *args, **kwargs): pass
    PDFTable = lambda *args, **kwargs: None
    TableStyle = lambda *args, **kwargs: None
    PDFImage = lambda *args, **kwargs: None
    Paragraph = lambda *args, **kwargs: None
    Spacer = lambda *args, **kwargs: None
    inch = 72.0
    
    # 将 mock 注入 sys.modules
    from types import ModuleType
    m = ModuleType("reportlab")
    Sys.modules["reportlab"] = m
    m.lib = ModuleType("reportlab.lib")
    Sys.modules["reportlab.lib"] = m.lib
    m.lib.pagesizes = ModuleType("reportlab.lib.pagesizes")
    Sys.modules["reportlab.lib.pagesizes"] = m.lib.pagesizes
    m.lib.styles = ModuleType("reportlab.lib.styles")
    Sys.modules["reportlab.lib.styles"] = m.lib.styles
    m.lib.units = ModuleType("reportlab.lib.units")
    Sys.modules["reportlab.lib.units"] = m.lib.units
    m.platypus = ModuleType("reportlab.platypus")
    Sys.modules["reportlab.platypus"] = m.platypus
    m.pdfbase = ModuleType("reportlab.pdfbase")
    Sys.modules["reportlab.pdfbase"] = m.pdfbase
    m.pdfbase.ttfonts = ModuleType("reportlab.pdfbase.ttfonts")
    Sys.modules["reportlab.pdfbase.ttfonts"] = m.pdfbase.ttfonts
    
    m.pdfbase.pdfmetrics = pdfmetrics
    m.pdfbase.ttfonts.TTFont = TTFont
    m.lib.styles.ParagraphStyle = ParagraphStyle