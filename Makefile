.PHONY: all orchestrator evalLib worker dashboard clean

all: orchestrator evalLib

orchestrator:
	cd mlInfraStudio && go mod tidy && go build -o bin/orchestrator orchestrator.go

evalLib:
	g++ -O2 -shared -fPIC -o mlInfraStudio/libeval.so mlInfraStudio/eval.cpp

worker:
	pip install -q scikit-learn numpy requests
	python mlInfraStudio/worker.py

dashboard:
	pip install -q streamlit plotly pandas requests
	streamlit run mlInfraStudio/dashboard.py

clean:
	rm -f mlInfraStudio/bin/orchestrator mlInfraStudio/libeval.so