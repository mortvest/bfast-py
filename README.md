# bfast-py
Python implementation of the BFAST and BFAST0n structural change detection algorithms 
by Jan Verbeselt et al. The implementation is based on the original 
[R implementation](https://github.com/bfast2/bfast). 

## Dependencies
The implementation was tested using Python 3.8.3.
In order to install all the necessary libraries, start the virtual environment and import
the requirements:

`pip install -f requirements.txt`

## How to run the tests
Tests for each file in the `src` directory are contained withing that
source file. In order to run the test, run:

`python file.py`

In order to get the more verbose output, run:

`python file.py --log=INFO`

In order to see the debug information, run:

`python file.py --log=DEBUG`
