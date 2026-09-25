.PHONY: setup sample_data test batch run clean help

PYTHON ?= python3
PIP ?= pip3

help:
	@echo "GreenSense: Tree & Asset Census Pipeline Commands:"
	@echo "  make setup        Install all required Python dependencies"
	@echo "  make sample_data  Generate synthetic fixture photos & populate historical database"
	@echo "  make test         Run pytest test suite"
	@echo "  make batch        Run CLI batch detection on sample_data/"
	@echo "  make run          Launch the Streamlit dashboard"
	@echo "  make clean        Remove generated database, sample data, and cache files"

setup:
	$(PIP) install -r requirements.txt

sample_data:
	$(PYTHON) generate_sample_register.py

test:
	pytest -v tests/

batch:
	$(PYTHON) detect.py --batch sample_data/ --zone "Manyata Block D Garden"

evaluate:
	$(PYTHON) evaluate.py

run:
	streamlit run app.py

clean:
	rm -rf sample_data asset_register.db asset_register.csv annotated_* .pytest_cache tests/__pycache__ utils/__pycache__ __pycache__
