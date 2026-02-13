train:
	python3 -m src.main train --train_path data/train_data.csv

predict:
	python3 -m src.main predict --test_path data/test_data.csv --artifacts_dir ./artifacts --output_path data/submission.csv

deploy-swagger:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

.PHONY: train predict deploy-swagger