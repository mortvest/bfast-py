# mosum: mosum.pdf
# 	zathura --fork mosum.pdf

# f_%: report.pdf
# 	zathura --fork $*.pdf

file_list = main instro stl mosum breakpoints bfast0n bfast implementation validation conclusion source_code

$(file_list): %: %.pdf
	zathura $*.pdf

%.pdf:
	# pdflatex $* && bibtex $* && pdflatex $* && pdflatex $*
	pdflatex $* && bibtex $* && pdflatex $* && pdflatex $*

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf


