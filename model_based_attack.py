import  random
import torch
from torch import nn
import json
import os
import time
import numpy as np
import copy
from torch.utils.data import DataLoader
from Graph_level_Models.helpers.config import args_parser
from Graph_level_Models.datasets.gnn_util import  transform_dataset,  split_dataset, Backdoor, Backdoor1,get_poisoned_dataloader_with_global_trigger, get_poisoned_dataloader_with_local_trigger 
from Graph_level_Models.datasets.TUs import TUsDataset
from Graph_level_Models.nets.TUs_graph_classification.load_net import gnn_model
from Graph_level_Models.helpers.evaluate import gnn_evaluate_accuracy
from Graph_level_Models.defenses.defense import foolsgold
from Graph_level_Models.trainer.workerbase  import WorkerBase
from Graph_level_Models.aggregators.aggregation import fed_avg,fed_opt, fed_median, fed_trimmedmean, fed_multi_krum, fed_bulyan

def server_robust_agg(args, grad):  ## server aggregation
    grad_in = np.array(grad).reshape((args.num_workers, -1)).mean(axis=0)
    return grad_in.tolist()


class ClearDenseClient(WorkerBase):
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device,
                 grad_stub, args, scheduler, avg_nodes, nfeat, net_params1):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               attack_iter=attack_iter, test_iter=test_iter, config=config,
                                               optimizer=optimizer, device=device)
        self.client_id = client_id
        self.grad_stub = None
        self.args = args
        self.scheduler = scheduler
        self.backdoor = Backdoor1(args, device, self.model, nfeat, avg_nodes, net_params1)

    def update(self):
        pass


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def main(args, logger):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    with open(args.config) as f:
        config = json.load(f)
    with open(args.config1) as f1:
        config1 = json.load(f1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device: {}'.format(device))
    torch.cuda.set_device(args.device_id)
    dataset = TUsDataset(args)
    args.device = device
    collate = dataset.collate
    MODEL_NAME = config['model']
    MODEL_NAME1 = config1['model']
    net_params = config['net_params']
    net_params1 = config1['net_params']
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    net_params1['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]

    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    net_params['n_classes'] = num_classes
    net_params['dropout'] = args.dropout
    net_params1['n_classes'] = num_classes
    net_params1['dropout'] = args.dropout
    args.epoch_backdoor = int(args.epoch_backdoor * args.epochs)

    model = gnn_model(MODEL_NAME, net_params)
    global_model = gnn_model(MODEL_NAME, net_params)
    global_model = global_model.to(device)
    client = []


    # logger data
    loss_func = nn.CrossEntropyLoss()
    # Load data
    partition, avg_nodes = split_dataset(args, dataset)
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    triggers = []

    all_workers_clean_test_list = []
    for i in range(args.num_workers):
        local_model = copy.deepcopy(model)
        local_model = local_model.to(device)

        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)




        train_dataset = partition[i]
        test_dataset = partition[args.num_workers + i]

        print("Client %d training data num: %d" % (i, len(train_dataset)))
        print("Client %d testing data num: %d" % (i, len(test_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  drop_last=drop_last,
                                  collate_fn=dataset.collate)
        attack_loader = None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 drop_last=drop_last,
                                 collate_fn=dataset.collate)
        all_workers_clean_test_list.append(test_loader)
        client.append(ClearDenseClient(client_id=i, model=local_model, loss_func=loss_func, train_iter=train_loader,
                                       attack_iter=attack_loader, test_iter=test_loader, config=config,
                                       optimizer=optimizer, device=device, grad_stub=None, args=args,
                                       scheduler=scheduler, avg_nodes=avg_nodes, nfeat=net_params['in_dim'], net_params1 = net_params1))
    # check model memory address
    for i in range(args.num_workers):
        add_m = id(client[i].model)
        add_o = id(client[i].optimizer)
        print('model {} address: {}'.format(i, add_m))
        print('optimizer {} address: {}'.format(i, add_o))
    # prepare backdoor local backdoor dataset
    test_clean_loader_list = []
    test_unchanged_loader_list = []



    weight_history = []
    global_test_acc_list = []
    for epoch in range(args.epochs):
        print('epoch:', epoch)

        # worker results
        worker_results = {}
        for i in range(args.num_workers):
            worker_results[f"client_{i}"] = {"train_loss": None, "train_acc": None, "test_loss": None, "test_acc": None}

        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        different_clients_test_accuracy_local_trigger = []

        for i in range(args.num_workers):
            att_list = []
            train_loss, train_acc, test_loss, test_acc = client[i].gnn_train(global_model,args)
            different_clients_test_accuracy_local_trigger.append(test_acc)
            client[i].scheduler.step()
            print('Client %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f'
                  % (i, train_loss, train_acc, test_loss, test_acc))


            # save worker results
            for ele in worker_results[f"client_{i}"]:
                if ele == "train_loss":
                    worker_results[f"client_{i}"][ele] = train_loss
                elif ele == "train_acc":
                    worker_results[f"client_{i}"][ele] = train_acc
                elif ele == "test_loss":
                    worker_results[f"client_{i}"][ele] = test_loss
                elif ele == "test_acc":
                    worker_results[f"client_{i}"][ele] = test_acc




        # wandb logger
        # logger.log(worker_results)

        selected_clients = random.sample(client, args.num_selected_models)
        # if there is a defense applied
        print("args.defense:", args.defense)
        print("frac_of_avg:", args.frac_of_avg)
        print("density", args.density)
        if args.defense == 'foolsgold':
            weights = []
            for i in range(args.num_workers):
                weights.append(client[i].get_weights())
                weight_history.append(client[i].get_weights())
            result, weight_history, alpha = foolsgold(args, weight_history, weights)
            for i in range(args.num_workers):
                client[i].set_weights(weights=result)
                client[i].upgrade()
        elif args.defense == 'fedavg':
             global_model = fed_avg(global_model,selected_clients, args)
             # send to local model
             for param_tensor in global_model.state_dict():
                 global_para = global_model.state_dict()[param_tensor]
                 for local_client in client:
                     local_client.model.state_dict()[param_tensor].copy_(global_para)
        elif args.defense == 'fedopt':
             global_model = fed_opt(global_model,selected_clients, args)
             # send to local model
             for param_tensor in global_model.state_dict():
                 global_para = global_model.state_dict()[param_tensor]
                 for local_client in client:
                     local_client.model.state_dict()[param_tensor].copy_(global_para)
        elif args.defense == 'fedprox':
             global_model = fed_avg(global_model,selected_clients, args)
             # send to local model
             for param_tensor in global_model.state_dict():
                 global_para = global_model.state_dict()[param_tensor]
                 for local_client in client:
                     local_client.model.state_dict()[param_tensor].copy_(global_para)
        elif args.defense == 'fed_median':
             global_model = fed_median(global_model,selected_clients, args)
             # send to local model
             for param_tensor in global_model.state_dict():
                 global_para = global_model.state_dict()[param_tensor]
                 for local_client in client:
                     local_client.model.state_dict()[param_tensor].copy_(global_para)
        elif args.defense == 'fed_trimmedmean':
             global_model = fed_trimmedmean(global_model,selected_clients, args)
             # send to local model
             for param_tensor in global_model.state_dict():
                 global_para = global_model.state_dict()[param_tensor]
                 for local_client in client:
                     local_client.model.state_dict()[param_tensor].copy_(global_para)
        elif args.defense == 'fed_multi_krum':
             global_model = fed_multi_krum(global_model,selected_clients, args)
             # send to local model
             for param_tensor in global_model.state_dict():
                 global_para = global_model.state_dict()[param_tensor]
                 for local_client in client:
                     local_client.model.state_dict()[param_tensor].copy_(global_para)
        elif args.defense == 'fed_krum':
             global_model = fed_multi_krum(global_model,selected_clients, args)
             # send to local model
             for param_tensor in global_model.state_dict():
                 global_para = global_model.state_dict()[param_tensor]
                 for local_client in client:
                     local_client.model.state_dict()[param_tensor].copy_(global_para)
        elif args.defense == 'fed_bulyan':
             global_model = fed_bulyan(global_model,selected_clients, args)
             # send to local model
             for param_tensor in global_model.state_dict():
                 global_para = global_model.state_dict()[param_tensor]
                 for local_client in client:
                     local_client.model.state_dict()[param_tensor].copy_(global_para)




        else:
            weights = []
            for i in range(args.num_workers):
                weights.append(client[i].get_weights())
                weight_history.append(client[i].get_weights())
            result, weight_history, alpha = foolsgold(args, weight_history, weights)
            result = server_robust_agg(args, weights)
            for i in range(args.num_workers):
                client[i].set_weights(weights=result)
                client[i].upgrade()


        # evaluate the global model: test_acc
        test_acc = gnn_evaluate_accuracy(client[args.num_workers-1].test_iter, client[0].model)
        print('Global Test Acc: %.3f' % test_acc)
        global_test_acc_list.append(test_acc)

    print("Global Test Acc:", global_test_acc_list)



    # clean accuracy , poison accuracy, attack success rate
    # average all the workers

    all_clean_acc_list = []
    for i in range(args.num_workers):
        tmp_acc = gnn_evaluate_accuracy(all_workers_clean_test_list[i], client[i].model)
        print('Client %d with clean accuracy: %.3f' % (i,  tmp_acc))
        all_clean_acc_list.append(tmp_acc)

    average_all_clean_acc = np.mean(np.array(all_clean_acc_list))

    attack_success_rate_list = []
    local_attack_success_rate_list = []
    attack_loader_list = []
    for i in range(args.num_mali):
        client[i].backdoor.fit(client[i].model, partition[i], args, avg_nodes)
        test_local_trigger = get_poisoned_dataloader_with_local_trigger(partition[i+args.num_workers], avg_nodes, args, client, device, i)
        test_local_trigger_load = DataLoader(test_local_trigger, batch_size=args.batch_size, shuffle=True,
                                        drop_last=drop_last,
                                        collate_fn=dataset.collate)
        attack_loader_list.append(test_local_trigger_load)
        tmp_acc = gnn_evaluate_accuracy(attack_loader_list[i], client[i].model)
        print('Malicious client %d with local trigger, attack success rate: %.4f' % (i, tmp_acc))
        local_attack_success_rate_list.append(tmp_acc)
    test_global_trigger = get_poisoned_dataloader_with_global_trigger(partition[-1], avg_nodes, args, client, device)

    test_global_trigger_load = DataLoader(test_global_trigger, batch_size=args.batch_size, shuffle=True,
                                        drop_last=drop_last,
                                        collate_fn=dataset.collate)
    average_local_attack_success_rate_acc = np.mean(np.array(local_attack_success_rate_list))


    local_clean_acc_list = []
    for i in range(args.num_mali):
        tmp_acc = gnn_evaluate_accuracy(client[i].test_iter, client[i].model)
        print('Malicious client %d with clean data, clean accuracy: %.4f' % (i, tmp_acc))

        local_clean_acc_list.append(tmp_acc)
    average_local_clean_acc = np.mean(np.array(local_clean_acc_list))

    average_local_unchanged_acc = 0


    transfer_attack_success_rate_list = []
    final_attack_success_rate_list = []
    if args.num_workers-args.num_mali <= 0:
        average_transfer_attack_success_rate = -10000.0
    else:
        for i in range(args.num_mali):
            for j in range(args.num_workers - args.num_mali):
                tmp_acc = gnn_evaluate_accuracy(attack_loader_list[i], client[args.num_mali+j].model)
                print('Clean client %d with  local trigger %d: %.3f' % (args.num_mali+j, i, tmp_acc))
                transfer_attack_success_rate_list.append(tmp_acc)
        average_transfer_attack_success_rate = np.mean(np.array(transfer_attack_success_rate_list))

    if args.num_workers-args.num_mali >= 0:
        for i in range(args.num_workers):
            tmp_acc = gnn_evaluate_accuracy(test_global_trigger_load, client[i].model)
            print('client %d with  global trigger : %.3f' % (i, tmp_acc))
            if (i-args.num_mali) >= 0:
                final_attack_success_rate_list.append(tmp_acc)
        final_attack_success_rate = np.mean((np.array(final_attack_success_rate_list)))
        print("final_attack_success_rate:", final_attack_success_rate)

    return average_all_clean_acc, average_local_attack_success_rate_acc, average_local_clean_acc,average_local_unchanged_acc, average_transfer_attack_success_rate, final_attack_success_rate




if __name__ == '__main__':
    main()