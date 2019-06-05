all: preprocessing

preprocessing:build 
	cd python/ && python preprocessing.py


build:
	mkdir build
	mkdir build/plots
	mkdir build/preprocessed

clean:
	rm -rf build