import argparse
import utils
import logging
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="train_test",
                        choices=['train', 'test', 'train_test'])
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default=
        "~/mind/",
    )
    parser.add_argument("--dataset",
                        type=str,
                        default='MIND')
    parser.add_argument(
        "--train_dir",
        type=str,
        default='train',
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default='test',
    )
    parser.add_argument("--filename_pat", type=str, default="behaviors*.tsv")
    parser.add_argument("--scoring_output", type=str, default="epoch-1-0--0.55456-0.76515*.tsv")
    parser.add_argument("--model_dir", type=str, default='~/mind/PLM-NR/model')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--enable_slate_data", type=utils.str2bool, default=False)
    parser.add_argument("--slate_length", type=int, default=2)  # slate_length - neg_ratio = pos_ratio, so slate_length must > neg_ratio
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    parser.add_argument("--enable_hvd", type=utils.str2bool, default=True)
    parser.add_argument("--enable_incremental", type=utils.str2bool, default=True)
    parser.add_argument("--enable_detect_anomaly", type=utils.str2bool, default=False)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--filter_num_user", type=int, default=0)
    parser.add_argument("--filter_num_word", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=2000)    

    # model training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ltr_loss", type=str, default='softmax', choices=['pointwise', 'pairwise', 'softmax', 'cb_ndcg', 'neural_ndcg'])
    parser.add_argument("--optimizer", type=str, default='AdamW', choices=['Adam', 'AdamW'])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--correct_bias", type=utils.str2bool, default=True)
    parser.add_argument("--clip_grad_norm", type=float, default=None)  # None for no clipping, 2.0 for clipping norm 2.0
    parser.add_argument("--enable_lr_scheduler", type=utils.str2bool, default=True)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument("--fineune_options", type=int, default=-2, choices=[0, -2, -12])
    parser.add_argument("--enable_evaluation", type=utils.str2bool, default=True)

    # model architecture options
    parser.add_argument("--enable_fastformer_user", type=utils.str2bool, default=True)
    parser.add_argument("--enable_fastformer_text", type=utils.str2bool, default=False)
    parser.add_argument("--enable_multihead_user", type=utils.str2bool, default=False)
    parser.add_argument("--enable_multihead_text", type=utils.str2bool, default=True)
    parser.add_argument("--enable_multihead_fastformer_text", type=utils.str2bool, default=False)
    parser.add_argument("--enable_additive_user_attributes", type=utils.str2bool, default=True)
    parser.add_argument("--enable_additive_news_attributes", type=utils.str2bool, default=True)
    parser.add_argument("--interaction", type=str, default='concatenation', choices=['cosine', 'hadamard', 'concatenation'])
    parser.add_argument("--enable_weight_tfboard", type=utils.str2bool, default=True)

    # model feature options
    parser.add_argument(
        "--news_attributes",
        type=str,
        nargs='+',
        default=['title', 'abstract', 'domain', 'category', 'subcategory'],
        choices=['title', 'abstract', 'body', 'category', 'domain', 'subcategory'])
    parser.add_argument(
        "--user_attributes",
        type=str,
        nargs='+',
        default=['click_docs', 'user_id'],
        choices=['click_docs', 'user_id'])
    parser.add_argument("--process_uet", type=utils.str2bool, default=False)
    parser.add_argument("--process_bing", type=utils.str2bool, default=False)
    parser.add_argument("--num_words_title", type=int, default=24)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=50)
    parser.add_argument("--num_words_uet", type=int, default=16)
    parser.add_argument("--num_words_bing", type=int, default=26)
    parser.add_argument("--do_lower_case", type=utils.str2bool, default=True)    
    parser.add_argument(
        "--user_log_length",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--word_embedding_dim",
        type=int,
        default=None,
    )  # None to let the program assign it automatically by pre-trained model
    parser.add_argument("--embedding_source",
                        type=str,
                        default='random',
                        choices=['glove', 'muse', 'random'])
    parser.add_argument("--freeze_embedding",
                        type=utils.str2bool,
                        default=False)
    parser.add_argument("--use_padded_news_embedding",
                        type=utils.str2bool,
                        default=False)
    parser.add_argument("--padded_news_different_word_index",
                        type=utils.str2bool,
                        default=False)
    parser.add_argument(
        "--news_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--news_query_vector_dim",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--user_query_vector_dim",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=20,
    )
    parser.add_argument("--user_log_mask", type=utils.str2bool, default=True)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--max_steps_per_epoch", type=int, default=10000)

    parser.add_argument(
        "--load_ckpt_train",
        type=str,
        default=None,  # means to take latest trained model
        help="choose which ckpt to load and train"
    )

    parser.add_argument(
        "--load_ckpt_test",
        type=str,
        default=None,  # means to take latest trained model
        help="choose which ckpt to load and test"
    )
    # share
    parser.add_argument("--title_share_encoder", type=utils.str2bool, default=False)

    # pretrain
    parser.add_argument("--num_layers", type=int, default=None) # None to let the program assign it automatically by pre-trained model
    parser.add_argument("--pretrain_lm_path", type=str,
                        default="~/mind/MiniLM-L12-H384-uncased",
                        choices=['~/mind/bert-base-uncased', '~/mind/MiniLM-L12-H384-uncased', '~/mind/unilm-base-cased', '~/mind/unilm-large-cased', '~/mind/roberta-large'])
    parser.add_argument("--use_pretrain_news_encoder", type=utils.str2bool, default=False)
    parser.add_argument("--pretrain_news_encoder_path", type=str, default="~/mind/bert-base-uncased") 
    parser.add_argument("--pretrain_user_id_encoder_path", type=str, default=None)
    parser.add_argument("--pretrain_news_id_encoder_path", type=str, default=None)
    # parser.add_argument("--pretrain_user_id_encoder_path", type=str, default="~/mind/MIND/train/embeddings_user_id.pt")
    # parser.add_argument("--pretrain_news_id_encoder_path", type=str, default="~/mind/MIND/train/embeddings_news_id.pt")

    # inference
    parser.add_argument("--ignore_unseen_user", type=utils.str2bool, default=False)
    parser.add_argument("--save_embedding", type=utils.str2bool, default=False)

    args = parser.parse_args()

    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
