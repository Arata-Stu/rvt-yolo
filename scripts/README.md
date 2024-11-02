# Preprocess Dataset

## run
```bash
DATA_DIR=/path/to/input_dir
DEST_DIR=/path/to/output_dir
NUM_PROCESSES=20
python3 preprocess_dataset.py ${DATA_DIR} ${DEST_DIR} conf_preprocess/representation/event_frame.yaml \
conf_preprocess/extraction/const_duration.yaml conf_preprocess/filter_gen1.yaml -ds gen1 -np ${NUM_PROCESSES}

```