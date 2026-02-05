train:
	python3 -m src.main train --train_path data/train_data.csv

predict:
	python3 -m src.main predict --test_path data/test_data.csv --artifacts_dir ./artifacts --output_path data/submission.csv

.PHONY: train predict