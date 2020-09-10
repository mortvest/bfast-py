.PHONY: clean $(file_list) pdf presentation

images = $(shell ls imgs/*.png)
latex_command = pdflatex -halt-on-error
tex_files = $(filter-out presentation.tex, $(shell ls *tex))
file_list = $(basename $(tex_files))
pdf_viewer = zathura
pdf_viewer_p = zathura --mode=presentation --page=1


pdf: presentation.pdf
	$(pdf_viewer) presentation.pdf

presentation: presentation.pdf
	$(pdf_viewer_p) presentation.pdf

presentation.pdf: $(tex_files) presentation.tex $(bib_file) $(images)
	$(latex_command) presentation

$(file_list): %: %.pdf
	$(pdf_viewer) $*.pdf

%.pdf: %.tex $(images)
	$(latex_command) $*

clean:
	rm -f *.aux *.log *.nav *.out *.toc *.snm *.pdf *.vrb
