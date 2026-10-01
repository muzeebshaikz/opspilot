.PHONY: install data eval abstention adversarial test serve demo all clean
PY := .venv/Scripts/python.exe

install:
	$(PY) -m pip install -r requirements.txt

data eval:          ## generate corpus + eval set, build index, run experiments
	$(PY) -m scripts.build_index

abstention:
	$(PY) scripts/calibrate_abstention.py

adversarial:
	$(PY) scripts/run_adversarial.py

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m uvicorn opspilot.api.main:app --reload --port 8000

demo:
	$(PY) scripts/demo_scenarios.py --base http://127.0.0.1:8000

all: data abstention adversarial test

clean:
	rm -rf data/qdrant data/opspilot.db artifacts/reports/*.json
