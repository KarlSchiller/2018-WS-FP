all:  build/main.pdf

build:
	mkdir -p build

build/main.pdf: build/tables/Gauss-Fit-Parameter.tex \
	build/tables/det_eff.tex \
	build/tables/anleitung_eu.tex | build

build/tables/Gauss-Fit-Parameter.tex build/tables/det_eff.tex build/tables/anleitung_eu.tex: calculations/auswertung2.py | build
	python calculations/auswertung2.py


build/main.pdf: FORCE | build
	  TEXINPUTS="$(call translate,build:)" \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
		--interaction=nonstopmode \
		--halt-on-error
	main.tex

light: FORCE | build
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex

clean:
	rm -rf build

FORCE:

.PHONY: all clean
