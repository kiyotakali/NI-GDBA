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

# def main(args):
#     with open(args.config) as f:
#         config = json.load(f)
#     model_name = config['model']
#     if args.defense == "scaffold":
#         from backdoor_graph_scaffold import main as backdoor_main
#     else:
#         from model_based_attack import main as backdoor_main

#     results = []
#     logger = []
#     frac_of_avg_list = [0.05, 0.10, 0.15,0.20]  # 请根据需要调整这些值
#     density_list = [0.05, 0.10, 0.15,0.20,0.25,0.30]  # 请根据需要调整这些值

#     for frac_of_avg in frac_of_avg_list:
#         for density in density_list:
#             args.frac_of_avg = frac_of_avg
#             args.density = density
#             final_attack_success_rate_list = []
#             for i in range(len(seeds)):
#                 args.seed = seeds[i]
#                 _, _, _, _, _, final_attack_success_rate = backdoor_main(args, logger)
#                 final_attack_success_rate_list.append(final_attack_success_rate)
#             avg_success_rate = sum(final_attack_success_rate_list) / len(final_attack_success_rate_list)
#             results.append((frac_of_avg, density, final_attack_success_rate))

#     for frac_of_avg, density, success_rates in results:
#         print(f"frac_of_avg: {frac_of_avg}, density: {density}")
#         print(f"各次实验的攻击成功率: {success_rates}")
#         print("-" * 50)

# if __name__ == "__main__":
#     main(args)

import json
import os
from datetime import datetime

# ... (其他导入和代码保持不变)

def main(args):
    with open(args.config) as f:
        config = json.load(f)
    model_name = config['model']
    if args.defense == "scaffold":
        from backdoor_graph_scaffold import main as backdoor_main
    else:
        from model_based_attack import main as backdoor_main

    results = []
    logger = []
    frac_of_avg_list = [0.05, 0.10, 0.15, 0.20]
    density_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # 创建结果目录和文件
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results__AIDS_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    for frac_of_avg in frac_of_avg_list:
        for density in density_list:
            args.frac_of_avg = frac_of_avg
            args.density = density
            final_attack_success_rate_list = []
            for i in range(len(seeds)):
                args.seed = seeds[i]
                _, _, _, _, _, final_attack_success_rate = backdoor_main(args, logger)
                final_attack_success_rate_list.append(final_attack_success_rate)
            avg_success_rate = sum(final_attack_success_rate_list) / len(final_attack_success_rate_list)
            
            result = {
                "frac_of_avg": frac_of_avg,
                "density": density,
                "avg_success_rate": avg_success_rate,
                "success_rates": final_attack_success_rate_list
            }
            results.append(result)

            # 打印当前结果
            print(f"frac_of_avg: {frac_of_avg}, density: {density}")
            print(f"平均攻击成功率: {avg_success_rate}")
            print(f"各次实验的攻击成功率: {final_attack_success_rate_list}")
            print("-" * 50)

            # 将当前结果追加到文件
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"结果已更新到文件: {filepath}")

    print(f"所有结果已保存到文件: {filepath}")
    for result in results:
        print(f"frac_of_avg: {result['frac_of_avg']}, density: {result['density']}")
        print(f"平均攻击成功率: {result['avg_success_rate']}")
        print(f"各次实验的攻击成功率: {result['success_rates']}")
        print("-" * 50)

if __name__ == "__main__":
    main(args)