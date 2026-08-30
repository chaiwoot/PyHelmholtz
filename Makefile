main:
	echo 'usage: make build'

build:
	python -m build

check:
	twine check dist/*

upload:
	twine upload dist/*

clean:  # for Linux only
	rm -rf dist/ pyhelmholtz.egg-info/
