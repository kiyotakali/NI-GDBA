from Graph_level_Models.helpers.config import args_parser

from Graph_level_Models.helpers.metrics_utils import log_test_results
import numpy as np
import json
import wandb

args = args_parser()
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000, size=5)


project_name = [args.proj_name, args.proj_name+ "debug"]
proj_name = project_name[0]

def main(args):
    with open(args.config) as f:
        config = json.load(f)
    model_name = config['model']
    if args.defense == "scaffold":
        from backdoor_graph_scaffold import main as backdoor_main
    else:
        from model_based_attack import main as backdoor_main


    average_all_clean_acc_list, average_local_attack_success_rate_acc_list, average_local_clean_acc_list, final_attack_success_rate_list_list, logger = [], [], [], [], []
    for i in range(len(seeds)):
        args.seed = seeds[i]
        average_all_clean_acc, average_local_attack_success_rate_acc, average_local_clean_acc, average_local_unchanged_acc,average_transfer_attack_success_rate, final_attack_success_rate = backdoor_main(args, logger)
        final_attack_success_rate_list_list.append(final_attack_success_rate)

    mean_final_attack_success_rate = np.mean(np.array(final_attack_success_rate_list_list))
    std_final_attack_success_rate = np.std(np.array(final_attack_success_rate_list_list))
    print(f"mean_final_attack_success_rate: {mean_final_attack_success_rate}, std_final_attack_success_rate: {std_final_attack_success_rate}")

if __name__ == "__main__":
    main(args)