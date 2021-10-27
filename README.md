# temporal-shape-dataset
A simpler action template dataset

### Generate dataset
`python generate_classification_dataset.py --num-sequences 10 --object-mode dot --symbol-size 2 --textured-background 0`

### Test run to train on existing data
`cd src; python main.py --config configs/convlstm.json --job_identifier test --fast_dev_run=True --log_every_n_steps=5 --gpus=1`
