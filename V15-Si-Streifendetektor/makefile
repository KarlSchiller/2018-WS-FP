# Windows-freundliche Übersetzung in MSys2/Cydwin
ifeq (,$(shell sh -c 'cygpath --version 2> /dev/null'))
  # Unix
  pwd := $$(pwd)
  translate = $1
else
  # Windows mit MSys2/Cygwin
  pwd := $$(cygpath -m "$$(pwd)")
  translate = $(shell echo '$1' | sed 's/:/;/g')
endif

all:  build/main.pdf | build

# further dependencies of auswertung.py
calc: rohdaten/ui-characteristic.txt

# python script
calc: auswertung.py matplotlibrc header-matplotlib.tex | build
	TEXINPUTS=$$(pwd): python auswertung.py

build/ui-characteristic.pdf: calc | build

build/main.pdf: FORCE | build
	  TEXINPUTS="$(call translate,build:)" \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
		--interaction=nonstopmode \
		--halt-on-error \
	main.tex

fast: FORCE | build
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	biber --output-directory=build main.bcf
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex
	lualatex --output-directory=build --interaction=nonstopmode --halt-on-error main.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE: build/ui-characteristic.pdf | build

.PHONY: all clean
