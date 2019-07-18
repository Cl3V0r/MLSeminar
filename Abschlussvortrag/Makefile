all: build/vortrag.pdf


TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build

build/vortrag.pdf: FORCE | build
	latexmk $(TeXOptions) vortrag.tex

FORCE:

build:
	mkdir -p build/

clean:
	rm -rf build