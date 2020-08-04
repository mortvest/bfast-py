.PHONY: clean $(file_list) main

tex_files = $(filter-out main.tex, $(shell ls *tex))
file_list = $(basename $(tex_files))

$(file_list): %: %.pdf
	zathura $*.pdf

%.pdf: %.tex
	pdflatex $* && bibtex $* && pdflatex $* && pdflatex $*

main.pdf: $(tex_files) main.tex
	pdflatex main.tex

main: main.pdf
	zathura main.pdf

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
