#!/bin/bash
# LaTeX build script for research paper

cd "$(dirname "$0")/src"

# Ensure TinyTeX is in PATH
export PATH="$HOME/bin:$PATH"

echo "Building LaTeX document..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo ""
echo "Build complete! PDF: $(pwd)/main.pdf"
echo "Size: $(du -h main.pdf | cut -f1)"
