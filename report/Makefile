.PHONY: clean $(file_list)

file_list = main instro stl mosum breakpoints bfast0n bfast implementation validation conclusion source_code

$(file_list): %: %.pdf
	zathura $*.pdf

%.pdf: %.tex
	pdflatex $* && bibtex $* && pdflatex $* && pdflatex $*

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
