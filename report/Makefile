.PHONY: clean $(file_list) main

tex_files = $(filter-out main.tex, $(shell ls *tex))
file_list = $(basename $(tex_files))
pdf_viewer = zathura

$(file_list): %: %.pdf
	$(pdf_viewer) $*.pdf

main: main.pdf
	$(pdf_viewer) main.pdf

%.pdf: %.tex
	pdflatex $* && bibtex $* && pdflatex $* && pdflatex $*

main.pdf: $(tex_files) main.tex
	pdflatex main && bibtex main && pdflatex main && pdflatex main

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
