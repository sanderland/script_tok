"""Compile the working notes to PDF.

    uv run python paper_utils/morphology/notes/build.py
"""

import os

import typst

HERE = os.path.dirname(os.path.abspath(__file__))
typst.compile(os.path.join(HERE, "notes.typ"), output=os.path.join(HERE, "notes.pdf"), root=HERE)
print(os.path.join(HERE, "notes.pdf"))
