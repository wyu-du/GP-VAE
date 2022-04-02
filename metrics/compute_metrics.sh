OUT_DIR=twitter_url_ae_2021-09-09-13-13-04/outs.json 

python quality.py --reference_path ../data/twitter_url --reference_file tst.tsv \
--output_path ../models/transformer/ \
--output_file $OUT_DIR

python diversity.py \
--output_path ../models/transformer/ \
--output_file $OUT_DIR
