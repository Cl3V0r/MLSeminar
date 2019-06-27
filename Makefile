all: preprocessing

preprocessing:build 
	cd python/ && python preprocessing.py

build:
	mkdir build
	mkdir build/plots
	mkdir build/preprocessed
	mkdir build/graph
	mkdir model

clean:
	rm -rf build
