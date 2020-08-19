clean_spoken_digits:
	python create.py

clean_spoken_digits.zip: clean_spoken_digits
	zip -r clean_spoken_digits.zip clean_spoken_digits

all: clean_spoken_digits.zip

clean:
	rm -rf clean_spoken_digits
