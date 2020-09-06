.PHONY: clean presentation

images = $(shell ls imgs/*.png)
latex_command = pdflatex -halt-on-error
pdf_viewer_p = zathura --mode=presentation --page=1
pdf_viewer = zathura


pdf: presentation.pdf
	$(pdf_viewer) presentation.pdf

presentation: presentation.pdf
	$(pdf_viewer_p) presentation.pdf

presentation.pdf: presentation.tex $(images)
	$(latex_command) presentation.tex %% $(latex_command) presentation.tex

clean:
	rm -f *.aux *.log *.nav *.out *.toc *.snm *.pdf
