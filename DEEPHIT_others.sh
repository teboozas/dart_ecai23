
# python search_others_modified_deephit.py --dataset=flchain --use_BN --start_fold=0 --end_fold=1
# python search_others_modified_deephit.py --dataset=flchain --use_BN --start_fold=1 --end_fold=2 --start_iter=185
# python search_others_modified_deephit.py --dataset=flchain --use_BN --start_fold=2 --end_fold=3 --start_iter=185
# python search_others_modified_deephit.py --dataset=flchain --use_BN --start_fold=3 --end_fold=4 --start_iter=185
# python search_others_modified_deephit.py --dataset=flchain --use_BN --start_fold=4 --end_fold=5 --start_iter=185


# # python search_others_modified_deephit.py --dataset=metabric --use_BN --start_fold=0 --end_fold=1
# python search_others_modified_deephit.py --dataset=metabric --use_BN --start_fold=1 --end_fold=2 --start_iter=185
# python search_others_modified_deephit.py --dataset=metabric --use_BN --start_fold=2 --end_fold=3 --start_iter=185
# python search_others_modified_deephit.py --dataset=metabric --use_BN --start_fold=3 --end_fold=4 --start_iter=185
# python search_others_modified_deephit.py --dataset=metabric --use_BN --start_fold=4 --end_fold=5 --start_iter=185


export CUDA_VISIBLE_DEVICES=0
# python search_others_modified_deephit.py --dataset=support --use_BN --start_fold=0 --end_fold=1
python search_others_modified_deephit.py --dataset=support --use_BN --start_fold=1 --end_fold=2 --start_iter=185
python search_others_modified_deephit.py --dataset=support --use_BN --start_fold=2 --end_fold=3 --start_iter=185
python search_others_modified_deephit.py --dataset=support --use_BN --start_fold=3 --end_fold=4 --start_iter=185
python search_others_modified_deephit.py --dataset=support --use_BN --start_fold=4 --end_fold=5 --start_iter=185


# python search_others_modified_deephit.py --dataset=gbsg --use_BN --start_fold=0 --end_fold=1
python search_others_modified_deephit.py --dataset=gbsg --use_BN --start_fold=1 --end_fold=2 --start_iter=185
python search_others_modified_deephit.py --dataset=gbsg --use_BN --start_fold=2 --end_fold=3 --start_iter=185
python search_others_modified_deephit.py --dataset=gbsg --use_BN --start_fold=3 --end_fold=4 --start_iter=185
python search_others_modified_deephit.py --dataset=gbsg --use_BN --start_fold=4 --end_fold=5 --start_iter=185