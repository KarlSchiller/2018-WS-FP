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

all:  build/main.pdf

build:
	mkdir -p build

build/main.pdf: build/blau_pi.tex | build

build/main.pdf: FORCE | build
	  TEXINPUTS="$(call translate,build:)" \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
		--interaction=nonstopmode
		--halt-on-error
	main.tex

build/blau_pi.tex: auswertung.py rohdaten/eichung.txt | build
	python auswertung.py

clean:
	rm -rf build

FORCE:

.PHONY: all clean
