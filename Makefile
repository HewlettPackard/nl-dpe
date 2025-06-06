.PHONY: create_python_env download_datasets remove_datasets evaluate_sst_2 evaluate_cola evaluate_mrpc

create_python_env:
	virtualenv ./.env --python=3
	./.env/bin/pip install poetry==2.1.2
	poetry install --without=dev


download_datasets:
	mkdir -p ./datasets/mrpc
	wget https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt -P ./datasets/mrpc
	wget https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt -P ./datasets/mrpc
	python ./nl_dpe/download_glue_data.py --data_dir=./datasets/glue --tasks=all --path_to_mrpc ./datasets/mrpc


remove_datasets:
	rm -rf ./datasets/glue
	rm -rf ./datasets/mrpc


evaluate_sst_2: datasets/glue/SST-2/dev.tsv
	python ./nl_dpe/evaluate.py sst-2 ./models/wise-tern-584/ ./datasets/glue/SST-2


evaluate_cola: datasets/glue/CoLA/dev.tsv
	python ./nl_dpe/evaluate.py cola ./models/blushing-dove-984/ ./datasets/glue/CoLA


evaluate_mrpc: datasets/glue/MRPC/dev.tsv
	python ./nl_dpe/evaluate.py mrpc ./models/spiffy-snake-501/ ./datasets/glue/MRPC

