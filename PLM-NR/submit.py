import os
import csv
import datetime
import logging
import utils
import tensorflow as tf
from parameters import parse_args


args = parse_args()


def save_txt(input_files_glob, outfile):
    rows = []
    def update_tsv(infile):
        with open(infile, 'r') as in_file:
            tsv_reader = csv.reader(in_file, delimiter='\t')
            for row in tsv_reader:
                row[1] = row[1].replace(', ', ',')
                rows.append(row)

    for i, input_file in enumerate(tf.compat.v1.gfile.Glob(input_files_glob)):
        logging.info(f"processing scoring output file: {i} - {input_file}") 
        update_tsv(input_file)
        logging.info(f"finished processing scoring output file: {i} - {input_file}")

    logging.info(f"sorting scores: {scoring_output_files} and saved to {outfile}")
    rows.sort(key = lambda x: int(x[0]))  # sort in place

    with open(outfile, 'a') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(rows)


if __name__ == "__main__":
    utils.setuplogger()
    scoring_output_files = '202111181*' # !!!change this to the files generated from the scoring run!!!

    input_files_glob = os.path.join(os.path.expanduser(args.model_dir), "prediction_{}.tsv".format(scoring_output_files))    
    outfile = os.path.join(os.path.expanduser(args.model_dir), "prediction_{}.txt".format(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")))

    logging.info(f"start processing scoring output files: {input_files_glob}")
    save_txt(input_files_glob, outfile)
    logging.info(f"finished processing scoring output files: {scoring_output_files} and saved to {outfile}")
