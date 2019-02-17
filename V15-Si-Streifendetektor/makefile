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

build/calibration.pdf: calc | build
build/cce-vergleich.pdf: calc | build
build/ccel.pdf: calc | build
build/common-mode.pdf: calc | build
build/delay-scan.pdf: calc | build
build/hitmap.pdf: calc | build
build/laser-delay.pdf: calc | build
build/number-of-channels.pdf: calc | build
build/number-of-clusters.pdf: calc | build
build/pedestal.pdf: calc | build
build/spectrum-adc.pdf: calc | build
build/spectrum-eV.pdf: calc | build
build/streifen-uebersicht.pdf: calc | build
build/streifen.pdf: calc | build
build/ui-characteristic.pdf: calc | build
build/ui-characteristic.tex: calc | build
build/umrechnung.pdf: calc | build
build/vergleich.pdf: calc | build

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

build:
	mkdir -p build

clean:
	rm -rf build

FORCE: build/calibration.pdf \
	build/cce-vergleich.pdf \
	build/ccel.pdf \
	build/common-mode.pdf \
	build/delay-scan.pdf \
	build/hitmap.pdf \
	build/laser-delay.pdf \
	build/number-of-channels.pdf \
	build/number-of-clusters.pdf \
	build/pedestal.pdf \
	build/spectrum-adc.pdf \
	build/spectrum-eV.pdf \
	build/streifen-uebersicht.pdf \
	build/streifen.pdf \
	build/ui-characteristic.pdf \
	build/ui-characteristic.tex \
	build/umrechnung.pdf \
	build/vergleich.pdf | build

.PHONY: all clean
