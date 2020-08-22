.PHONY: clean $(file_list) main

images = $(shell ls imgs/*.png)
latex_command = pdflatex -halt-on-error
tex_files = $(filter-out main.tex, $(shell ls *tex))
file_list = $(basename $(tex_files))
pdf_viewer = zathura
bib_file = mybib.bib


main: main.pdf
	$(pdf_viewer) main.pdf

main.pdf: $(tex_files) main.tex $(bib_file) $(images)
	$(latex_command) main && bibtex main && pdflatex main && pdflatex main

$(file_list): %: %.pdf
	$(pdf_viewer) $*.pdf

%.pdf: %.tex $(bib_file) $(images)
	$(latex_command) $* && bibtex $* && pdflatex $* && pdflatex $*

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
