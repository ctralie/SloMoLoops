FILE=paper

default : $(FILE).pdf

$(FILE).pdf : $(FILE).tex
	pdflatex $(FILE); bibtex $(FILE); pdflatex $(FILE); pdflatex $(FILE)

clean:
	rm $(FILE).pdf $(FILE).log $(FILE).aux $(FILE).bbl $(FILE).blg $(FILE).lbl $(FILE).brf
