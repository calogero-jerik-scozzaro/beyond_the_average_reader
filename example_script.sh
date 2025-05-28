#!/bin/bash
python3 main.py -p all_readers -m LSTM -n 10
python3 main.py -p top_k_similar_readers -m LSTM -n 10