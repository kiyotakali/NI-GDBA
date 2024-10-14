import networkx as nx
import random
import torch
import pickle
from tqdm import tqdm
import os
import numpy as np
import copy
import dgl
from torch.utils.data import random_split
from torch.nn import Linear
from Graph_level_Models.layers.gcn_layer import GCNLayer
from torch.utils.data import DataLoader, Dataset

class TriggerDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

def move_to_device(graph, device):
    return graph.to(device)

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]  # graphs
        self.graph_labels = lists[1] # labels

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])

def transform_dataset(trainset, testset, avg_nodes, args, nfeat, model):
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu:0')
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)

    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    train_labels = [graph[1] for graph in trainset]
    num_classes = torch.max(torch.tensor(train_labels)).item() + 1

    tmp_graphs = []
    tmp_idx = []
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg) # avg_nodes is the average number of all grpah's nodes
    for idx, graph in enumerate(train_untarget_graphs):
        if graph[0].num_nodes() > num_trigger_nodes:
            tmp_graphs.append(graph)
            tmp_idx.append(train_untarget_idx[idx])
    n_trigger_graphs = int(args.poisoning_intensity * len(trainset))
    final_idx = []
    if n_trigger_graphs <= len(tmp_graphs):
        train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
        final_idx = tmp_idx[:n_trigger_graphs]

    else:
        train_trigger_graphs = tmp_graphs
        final_idx = tmp_idx

    ##############################################################################################
    print("Start generating trigger position by {}".format(args.trigger_position))
    default_min_num_trigger_nodes = 3
    if num_trigger_nodes < default_min_num_trigger_nodes:
        num_trigger_nodes = default_min_num_trigger_nodes

    #Randomly choose the trigger
    trigger_list = []


    if args.trigger_position == "random":
        for data in train_trigger_graphs:
            # print("data[0].nodes().tolist()",len(data[0].nodes().tolist()))
            # print("num trigger nodes", num_trigger_nodes)
            if len(data[0].nodes().tolist()) < num_trigger_nodes:
                trigger_num = data[0].nodes().tolist()
            else:
                trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
            trigger_list.append(trigger_num)
    elif args.trigger_position == "degree":
        for data in train_trigger_graphs:
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            # sort according to degree
            degree_dict = dict(g.degree())
            sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    elif args.trigger_position == "cluster":
        for data in train_trigger_graphs:
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            #  sort according to cluster
            simple_g = nx.Graph(g)
            clustering_dict = nx.clustering(simple_g,weight='weight')
            sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)

            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    else:
        raise NameError

    ######################################################################
    print("Start preparing for the poisoned test datasets")
    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]
    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    print("The number of test changed graphs is: %d"%len(test_changed_graphs_final))
    test_trigger_list = []
    test_graph_idx = []
    # for gid,graph in enumerate(test_changed_graphs):
    #     trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
    #     test_trigger_list.append(trigger_idx)
    #     test_graph_idx.append(int(gid))

    if args.trigger_position == "random":
        for gid,data in enumerate(test_changed_graphs):
            # print("data[0].nodes().tolist()",len(data[0].nodes().tolist()))
            # print("num trigger nodes", num_trigger_nodes)
            if len(data[0].nodes().tolist()) < num_trigger_nodes:
                trigger_num = data[0].nodes().tolist()
            else:
                trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
            test_trigger_list.append(trigger_num)
            test_graph_idx.append(int(gid))
    elif args.trigger_position == "degree":
        for gid,data in enumerate(test_changed_graphs):
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            # sort according to degree
            degree_dict = dict(g.degree())
            sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
            trigger_num = sorted_nodes[:num_trigger_nodes]
            test_trigger_list.append(trigger_num)
            test_graph_idx.append(int(gid))
    elif args.trigger_position == "cluster":
        for gid,data in enumerate(test_changed_graphs):
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            #  sort according to cluster
            simple_g = nx.Graph(g)
            clustering_dict = nx.clustering(simple_g,weight='weight')
            sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)

            trigger_num = sorted_nodes[:num_trigger_nodes]
            test_trigger_list.append(trigger_num)
            test_graph_idx.append(int(gid))
    else:
        raise NameError


    ######################################################################
    print("Start generating trigger by {}".format(args.trigger_type))

    if args.trigger_type == "renyi":

        G_trigger = nx.erdos_renyi_graph(num_trigger_nodes, args.density, directed=False)
        if G_trigger.edges():
            G_trigger = G_trigger
        else:
            G_trigger = nx.erdos_renyi_graph(num_trigger_nodes, 1.0, directed=False)

    elif args.trigger_type == "ws":
        print("args.avg_degree",args.avg_degree)
        print("num_trigger_nodes",num_trigger_nodes)
        # if args.avg_degree >= num_trigger_nodes:
        #     args.avg_degree = num_trigger_nodes - 1
        G_trigger = nx.watts_strogatz_graph(num_trigger_nodes, args.avg_degree, args.density)
    elif args.trigger_type == "ba":
        if args.avg_degree >= num_trigger_nodes:
            args.avg_degree = num_trigger_nodes - 1
        # n: int Number of nodes
        # m: int Number of edges to attach from a new node to existing nodes
        G_trigger = nx.random_graphs.barabasi_albert_graph(n= num_trigger_nodes, m= args.avg_degree)
    elif args.trigger_type == "rr":
        #d int The degree of each node.
        # n integer The number of nodes.The value of must be even.
        if args.avg_degree >= num_trigger_nodes:
            args.avg_degree = num_trigger_nodes - 1
        if num_trigger_nodes % 2 != 0:
            num_trigger_nodes +=1
        G_trigger = nx.random_graphs.random_regular_graph(d = args.avg_degree, n = num_trigger_nodes)     # generate a regular graph which has 20 nodes & each node has 3 neghbour nodes.

    elif args.trigger_type == "gta":
        # adaptive method for generate the triggers, each poisoned graph have a specific trigger.
        # testing
        # device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
        print('gta_device',device)
        G_trigger = GraphTrojanNet(device, nfeat, int(avg_nodes * args.frac_of_avg), layernum=2).to(device)
        # self.trojan = Trojan_GAT(net_params1, int(avg_nodes * args.frac_of_avg)).to(self.device)
        optimizer_trigger = optim.Adam(G_trigger.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_shadow = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("Start preparing for the poisoned test datasets")
        count = 0
        count1 = 0
        for epoch in range(args.trojan_epochs):
            G_trigger.train()
            if count1 > 0:
                break
            for epoch_shadow in range(args.epochs):
                loss = 0
                correct = 0  # 用于跟踪正确预测的数量
                total = 0  # 用于跟踪总样本数量
                for  i, data in enumerate(train_trigger_graphs):
                    # 创建图的副本
                    g_copy = data[0].clone().to(device)
                    
                    labels = torch.tensor([args.target_label]).to(device)

                    #随机选取一部分节点作为触发节点
                    # trigger_nodes = torch.randperm(g_copy.number_of_nodes())[:num_trigger_nodes].to(self.device)
                    trigger_nodes = trigger_list[i]
                    # print('trigger_nodes,',trigger_nodes)


                    # 生成触发器特征和权重
                    # trojan_feat, trojan_weights = self.trojan(g_copy.ndata['feat'][trigger_nodes], self.args.thrd)

                    # trojan_feat, trojan_weights = self.trojan(g_copy, g_copy.ndata['feat'], trigger_nodes)
                    trojan_feat, trojan_weights= G_trigger(g_copy.ndata['feat'][trigger_nodes])                
                    # print("trojan_weights shape: ",trojan_weights.shape)
                    # print(f"trojan_feat shape: {trojan_feat.shape}")
                    # 将触发器特征注入到副本图中
                    trojan_feat = trojan_feat.view([-1, g_copy.ndata['feat'].shape[1]])
                    # print(f"trigger_nodes shape: {trigger_nodes.shape}")
                    # print(f"trojan_feat shape: {trojan_feat.shape}")
                    g_copy.ndata['feat'][trigger_nodes] = trojan_feat

                    trojan_edge = []
                    # print("Start injecting trigger into the poisoned train datasets")

                    for node1 in range(num_trigger_nodes):
                        for node2 in range(node1,num_trigger_nodes):  # 每个触发器节点有5个潜在的边
                            # print('node2,',node2)
                            if trojan_weights[node1][node2] > args.weight_threshold and trigger_nodes[node1] != trigger_nodes[node2]:
                                # 添加边，权重设置为 trojan_weights[node][edge]
                                # g_copy = dgl.add_edges(g_copy, trigger_nodes[node1], trigger_nodes[node2])
                                node1_tensor = torch.tensor([trigger_nodes[node1]], device=device)
                                node2_tensor = torch.tensor([trigger_nodes[node2]], device=device)
                                g_copy.add_edges(node1_tensor, node2_tensor)
                                g_copy.add_edges(node2_tensor, node1_tensor)
                                trojan_edge.append((trigger_nodes[node1], trigger_nodes[node2]))
                                trojan_edge.append((trigger_nodes[node2], trigger_nodes[node1]))
                    trojan_edges = torch.tensor(trojan_edge, device=device).t()
                # print('trojan_edges,',len(trojan_edges[0]),trojan_edges)
                


                # 前向传播和计算损失
                output = model.forward(g_copy, g_copy.ndata['feat'], g_copy.edata['feat']).to(device)
                # print('output,',output)
                # loss_target = self.args.target_loss_weight * F.nll_loss(output, labels)
                loss_target = model.loss(output, labels)
                
                # 计算准确率
                output = F.softmax(output, dim=1)
                # print('output,',output)
                _, predicted = torch.max(output, 1)  # 使用exp()来获得概率分布
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # print('loss_target,',loss_target)
                # if self.args.homo_loss_weight > 0 and len(trojan_edge) > 0:
                #     loss_homo = self.homo_loss(trojan_edges, g_copy.ndata['feat'], self.args.homo_boost_thrd)
                # print('loss_homo,',loss_homo)
                loss = loss_target + loss
                # print(i,loss)
                # print("num_of_test_trigger_graphs is: %d"%len(train_trigger_graphs))

                loss_target.backward()
                optimizer_trigger.step()
                optimizer_shadow.step()
                accuracy = 100 * correct / total
                print(f'Epoch {epoch*args.epochs+epoch_shadow}, Loss: {loss.item()}, ASR: {accuracy:.2f}%')
                if accuracy >= 80.9 or (epoch >=800 and accuracy >= 70.0) or (epoch >=1400 and accuracy >= 60.0):
                    count = count + 1
                if count >=10:
                    count1 = count
                    break

        G_trigger.train()
    else:
        raise NameError

    graphs_train = []
    graphs_test = []
    if args.trigger_type == "gta":
        for  i, data in enumerate(train_trigger_graphs):
            g_copy = data[0].clone().to(device)
            g_copy = dgl.add_self_loop(g_copy)
                
            
            trigger_nodes = trigger_list[i]
            # trigger_nodes.to(device)
            # trojan_feat, trojan_weights = client[num_ID].backdoor.trojan(g_copy.ndata['feat'][trigger_nodes], args.thrd)
            trojan_feat, trojan_weights = G_trigger(g_copy.ndata['feat'][trigger_nodes])
            trojan_feat = trojan_feat.view([-1, data[0].ndata['feat'].shape[1]])
            g_copy.ndata['feat'][trigger_nodes] = trojan_feat.to(device)
            for node1 in range(num_trigger_nodes):
                for node2 in range(node1,num_trigger_nodes):  # 每个触发器节点有5个潜在的边
                        # print('node2,',node2)
                    if trojan_weights[node1][node2] > args.weight_threshold and trigger_nodes[node1] != trigger_nodes[node2]:
                            # 添加边，权重设置为 trojan_weights[node][edge]
                            # g_copy = dgl.add_edges(g_copy, trigger_nodes[node1], trigger_nodes[node2])
                        node1_tensor = torch.tensor([trigger_nodes[node1]], device = device)
                        node2_tensor = torch.tensor([trigger_nodes[node2]], device = device)
                        g_copy.add_edges(node1_tensor, node2_tensor)
                        g_copy.add_edges(node2_tensor, node1_tensor)
            g_copy = g_copy.to('cpu:0')
            graphs_train.append(g_copy)
        print('graphs_train',len(graphs_train))

    
    # graphs = [data[0] for data in test_changed_graphs]
        labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
        print('labels',len(labels))
        train_trigger_graphs = DGLFormDataset(graphs_train, labels)

        for  i, data in enumerate(test_changed_graphs):
            g_copy = data[0].clone().to(device)
            g_copy = dgl.add_self_loop(g_copy)
                
            
            trigger_nodes = test_trigger_list[i]
            # trigger_nodes.to(device)
            # trojan_feat, trojan_weights = client[num_ID].backdoor.trojan(g_copy.ndata['feat'][trigger_nodes], args.thrd)
            trojan_feat, trojan_weights = G_trigger(g_copy.ndata['feat'][trigger_nodes])
            trojan_feat = trojan_feat.view([-1, data[0].ndata['feat'].shape[1]])
            g_copy.ndata['feat'][trigger_nodes] = trojan_feat.to(device)
            for node1 in range(num_trigger_nodes):
                for node2 in range(node1,num_trigger_nodes):  # 每个触发器节点有5个潜在的边
                        # print('node2,',node2)
                    if trojan_weights[node1][node2] > args.weight_threshold and trigger_nodes[node1] != trigger_nodes[node2]:
                            # 添加边，权重设置为 trojan_weights[node][edge]
                            # g_copy = dgl.add_edges(g_copy, trigger_nodes[node1], trigger_nodes[node2])
                        node1_tensor = torch.tensor([trigger_nodes[node1]], device = device)
                        node2_tensor = torch.tensor([trigger_nodes[node2]], device = device)
                        g_copy.add_edges(node1_tensor, node2_tensor)
                        g_copy.add_edges(node2_tensor, node1_tensor)
            g_copy = g_copy.to('cpu:0')
            graphs_test.append(g_copy)
        print('graphs_test',len(graphs_test))
        labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
        print('labels',len(labels))
        test_trigger_graphs = DGLFormDataset(graphs_test, labels)

        test_clean_data = [copy.deepcopy(graph) for graph in testset]

        test_clean_graphs = [data[0] for data in test_clean_data]
        test_clean_labels = [data[1] for data in test_clean_data]
        test_clean_data = DGLFormDataset(test_clean_graphs, test_clean_labels)


        test_unchanged_data = [copy.deepcopy(graph) for graph in testset if graph[1].item() == args.target_label]
        test_unchanged_graphs = [data[0] for data in test_unchanged_data]
        test_unchanged_labels = [data[1] for data in test_unchanged_data]
        test_unchanged_data = DGLFormDataset(test_unchanged_graphs, test_unchanged_labels)

        # test_clean_data = [copy.deepcopy(graph) for graph in testset]
        # test_clean_graphs = [move_to_device(data[0], device) for data in test_clean_data]
        # test_clean_labels = [data[1].to(device) for data in test_clean_data]
        # test_clean_data = DGLFormDataset(test_clean_graphs, test_clean_labels)

        # test_unchanged_data = [copy.deepcopy(graph) for graph in testset if graph[1].item() == args.target_label]
        # test_unchanged_graphs = [move_to_device(data[0], device) for data in test_unchanged_data]
        # test_unchanged_labels = [data[1].to(device) for data in test_unchanged_data]
        # test_unchanged_data = DGLFormDataset(test_unchanged_graphs, test_unchanged_labels)


        ############################Adaptive trigger##########################################
        return train_trigger_graphs, test_trigger_graphs, G_trigger, final_idx, test_clean_data, test_unchanged_data


    else:
        ############################Heuristic trigger##########################################
        print("Start injecting trigger into the poisoned train datasets")
        for  i, data in enumerate(tqdm(train_trigger_graphs)):
            for j in range(len(trigger_list[i])-1):
                for k in range(j+1, len(trigger_list[i])):
                    if (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) \
                        and G_trigger.has_edge(j, k) is False:
                        ids = data[0].edge_ids(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
                        data[0].remove_edges(ids)
                    elif (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) is False \
                        and G_trigger.has_edge(j, k):
                        data[0].add_edges(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))

        ######################################################################
        print("Start injecting trigger into the poisoned test datasets")
        # evaluation: randomly inject the trigger into the graph
        for ith, graph in tqdm(enumerate(test_changed_graphs)):
            trigger_idx = test_trigger_list[ith]
            for i in range(len(trigger_idx)-1):
                for j in range(i+1, len(trigger_idx)):
                    if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                        and G_trigger.has_edge(i, j) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                        graph[0].remove_edges(ids)
                    elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                        and G_trigger.has_edge(i, j):
                        graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))


    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)



    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)

    #### Construct the clean data
    test_clean_data = [copy.deepcopy(graph) for graph in testset]

    test_clean_graphs = [data[0] for data in test_clean_data]
    test_clean_labels = [data[1] for data in test_clean_data]
    test_clean_data = DGLFormDataset(test_clean_graphs, test_clean_labels)
    #### Construct the unchaged data and changed data into the same datsets [unchanged data, changed data]
    test_unchanged_data = [copy.deepcopy(graph) for graph in testset if graph[1].item() == args.target_label]
    test_unchanged_graphs = [data[0] for data in test_unchanged_data]
    test_unchanged_labels = [data[1] for data in test_unchanged_data]
    test_unchanged_data = DGLFormDataset(test_unchanged_graphs, test_unchanged_labels)


    return train_trigger_graphs, test_trigger_graphs, G_trigger, final_idx, test_clean_data, test_unchanged_data

def transform_dataset_same_local_trigger(trainset, testset, avg_nodes, args, G_trigger):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)

    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    tmp_graphs = []
    tmp_idx = []
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    for idx, graph in enumerate(train_untarget_graphs):
        if graph[0].num_nodes() > num_trigger_nodes:
            tmp_graphs.append(graph)
            tmp_idx.append(train_untarget_idx[idx])
    n_trigger_graphs = int(args.poisoning_intensity*len(trainset))
    final_idx = []
    if n_trigger_graphs <= len(tmp_graphs):
        train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
        final_idx = tmp_idx[:n_trigger_graphs]

    else:
        train_trigger_graphs = tmp_graphs
        final_idx = tmp_idx
    trigger_list = []
    for data in train_trigger_graphs:
        trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
        trigger_list.append(trigger_num)

    for  i, data in enumerate(train_trigger_graphs):
        for j in range(len(trigger_list[i])-1):
            for k in range(j+1, len(trigger_list[i])):
                if (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) \
                    and G_trigger.has_edge(j, k) is False:
                    ids = data[0].edge_ids(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
                    data[0].remove_edges(ids)
                elif (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) is False \
                    and G_trigger.has_edge(j, k):
                    data[0].add_edges(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
    ## rebuild data with target label
    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)



    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]


    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    print("num_of_test_changed_graphs is: %d"%len(test_changed_graphs_final))
    for graph in test_changed_graphs:
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for i in range(len(trigger_idx)-1):
            for j in range(i+1, len(trigger_idx)):
                if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                    and G_trigger.has_edge(i, j) is False:
                    ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                    graph[0].remove_edges(ids)
                elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                    and G_trigger.has_edge(i, j):
                    graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)
    #### Construct the clean data
    test_clean_graphs = [copy.deepcopy(graph) for graph in testset]
    test_clean_graphs = [data[0] for data in test_clean_graphs]
    test_clean_labels = [torch.tensor([data[1]]) for data in test_clean_graphs]
    test_clean_data = DGLFormDataset(test_clean_graphs, test_clean_labels)
    #### Construct the unchaged data and changed data into the same datsets [unchanged data, changed data]
    test_unchanged_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() == args.target_label]
    test_unchanged_graphs = [data[0] for data in test_unchanged_graphs]
    test_unchanged_labels = [torch.tensor([data[1]]) for data in test_unchanged_graphs]

    test_poison_graphs = graphs + test_unchanged_graphs
    test_poison_labels = labels + test_unchanged_labels
    test_poison_data = DGLFormDataset(test_poison_graphs, test_poison_labels)

    return train_trigger_graphs, test_trigger_graphs, final_idx, test_clean_data, test_poison_data

def inject_global_trigger_test(testset, avg_nodes, args, triggers):
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu:0')

    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]
    
    num_mali = len(triggers)
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg) * num_mali
    default_min_num_trigger_nodes = 3
    if num_trigger_nodes < default_min_num_trigger_nodes* num_mali:
        num_trigger_nodes = default_min_num_trigger_nodes* num_mali   
    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    if len(test_changed_graphs) == 0:
        raise ValueError('num_trigger_nodes are larger than all the subgraphs!!! Please resize the num_mali and frac_of_avg')
    print("num_of_test_changed_graphs is: %d"%len(test_changed_graphs_final))
    each_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    trigger_list = []
    if args.trigger_position == "random":
        for data in test_changed_graphs:
            # print("data[0].nodes().tolist()",len(data[0].nodes().tolist()))
            # print("num trigger nodes", num_trigger_nodes)
            if len(data[0].nodes().tolist()) < num_trigger_nodes:
                trigger_num = data[0].nodes().tolist()
            else:
                trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
            trigger_list.append(trigger_num)
    elif args.trigger_position == "degree":
        for data in test_changed_graphs:
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            # sort according to degree
            degree_dict = dict(g.degree())
            sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    elif args.trigger_position == "cluster":
        for data in test_changed_graphs:
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            #  sort according to cluster
            simple_g = nx.Graph(g)
            clustering_dict = nx.clustering(simple_g,weight='weight')
            sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)

            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    else:
        raise NameError
    if args.trigger_type == "gta":
        ############################Adaptive trigger##########################################
        print("Start injecting the global trigger into the poisoned test datasets")
        print('gta_device',device)
        num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
        default_min_num_trigger_nodes = 3
        if num_trigger_nodes < default_min_num_trigger_nodes:
                num_trigger_nodes = default_min_num_trigger_nodes   
        graphs = []
        for  i, data in enumerate(test_changed_graphs):
            
            g_copy = data[0].clone().to(device)
            g_copy = dgl.add_self_loop(g_copy)
                
                
            trigger_nodes_list = []

            # 生成所有点集
            # all_trigger_nodes = torch.randperm(g_copy.number_of_nodes())[:args.num_mali * num_trigger_nodes].to(device)
            for j in range(0, len(trigger_list[i]), num_trigger_nodes):
                trigger_nodes_list.append(trigger_list[i][j:j+num_trigger_nodes])
            # print('trigger_nodes_list',trigger_nodes_list)


            for j, trigger in enumerate(triggers):    
                trigger_nodes = trigger_nodes_list[j]
                # print('trigger_nodes',trigger_nodes)
                # print('g_copy.ndata[\'feat\'][trigger_nodes]',g_copy.ndata['feat'][trigger_nodes].shape)
                # trojan_feat, trojan_weights = client[j].backdoor.trojan(g_copy.ndata['feat'][trigger_nodes], args.thrd)
                trojan_feat, trojan_weights = trigger(g_copy.ndata['feat'][trigger_nodes])
                # print('g_copy.ndata[\'feat\'][trigger_nodes]',g_copy.ndata['feat'][trigger_nodes].shape)
                trojan_feat = trojan_feat.view([-1, data[0].ndata['feat'].shape[1]])
                # # g_copy.ndata['feat'][trigger_nodes] = trojan_feat.to(device)
                # print('trojan_feat',trojan_feat.shape)
                # print('g_copy.ndata[\'feat\'][trigger_nodes]',g_copy.ndata['feat'][trigger_nodes].shape)
                g_copy.ndata['feat'][trigger_nodes] = trojan_feat
                for node1 in range(num_trigger_nodes):
                    for node2 in range(node1,num_trigger_nodes):  # 每个触发器节点有5个潜在的边
                            # print('node2,',node2)
                        if trojan_weights[node1][node2] > args.weight_threshold and trigger_nodes[node1] != trigger_nodes[node2]:
                                # 添加边，权重设置为 trojan_weights[node][edge]
                                # g_copy = dgl.add_edges(g_copy, trigger_nodes[node1], trigger_nodes[node2])
                            node1_tensor = torch.tensor([trigger_nodes[node1]], device = device)
                            node2_tensor = torch.tensor([trigger_nodes[node2]], device = device)
                            g_copy.add_edges(node1_tensor, node2_tensor)
                            g_copy.add_edges(node2_tensor, node1_tensor)
            graphs.append(g_copy)
        
        # graphs = [data[0] for data in test_changed_graphs]
        
        labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
        test_trigger_graphs = DGLFormDataset(graphs, labels)


        return test_trigger_graphs
    else:
        for graph in test_changed_graphs:
            trigger_idx = trigger_list[test_changed_graphs.index(graph)]
            for idx, trigger in enumerate(triggers):
                start = each_trigger_nodes * idx
                for i in range(start, start+each_trigger_nodes-1):
                    for j in range(i+1, start+each_trigger_nodes):
                        if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                            and trigger.has_edge(i, j) is False:
                            ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                            graph[0].remove_edges(ids)
                        elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                            and trigger.has_edge(i, j):
                            graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)
    return test_trigger_graphs

def inject_global_trigger_train(trainset, avg_nodes, args, triggers):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)
   
    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    tmp_graphs = []
    tmp_idx = []
    num_mali = len(triggers)
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg) * num_mali

    for idx, graph in enumerate(train_untarget_graphs):
        if graph[0].num_nodes() > num_trigger_nodes:
            tmp_graphs.append(graph)
            tmp_idx.append(train_untarget_idx[idx])

    n_trigger_graphs = int(args.poisoning_intensity*len(trainset))
    final_idx = []
    if n_trigger_graphs <= len(tmp_graphs):
        train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
        final_idx = tmp_idx[:n_trigger_graphs]
    else:
        train_trigger_graphs = tmp_graphs
        final_idx = tmp_idx
    print("num_of_train_trigger_graphs is: %d"%len(train_trigger_graphs))
    each_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    for graph in train_trigger_graphs:
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for idx, trigger in enumerate(triggers):
            start = each_trigger_nodes * idx
            for i in range(start, start+each_trigger_nodes-1):
                for j in range(i+1, start+each_trigger_nodes):
                    if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                        and trigger.has_edge(i, j) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                        graph[0].remove_edges(ids)
                    elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                        and trigger.has_edge(i, j):
                        graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)
    return train_trigger_graphs, final_idx


def save_object(obj, filename):
    savedir = os.path.split(filename)[0]
    if not os.path.exists(savedir):
        os.makedirs(savedir)
  
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(filename, 'rb') as input:
        graphs = pickle.load(input)
    return graphs

def check_graph_type(dataset):
    graph = dataset.train[0][0]
    edges = graph.edges()
    edges_0 = edges[0].tolist()
    edges_1 = edges[1].tolist()
    count = 0
    for i in range(len(edges_0)):
        for j in range(i, len(edges_0)):
            if edges_0[j] == edges_1[i] and edges_1[j] == edges_0[i]:
                count += 2
    if count == len(edges_0):
        flag = True
    else:
        flag = False
    return flag

def p_degree_non_iid_split(trainset, args, num_classes):
    #sort trainset
    sorted_trainset = []
    for i in range(num_classes):
        indices = [idx for idx in range(len(trainset)) if trainset[idx][1] == i]
        tmp = [trainset[j] for j in indices]
        print("len tmp",len(tmp))
        sorted_trainset.append(tmp)

    p = args.p_degree
    #split data for every class
    # if num_classes == 2:
    #     p = 0.7
    # else:
    #     p = 0.5
    length_list = []
    for i in range(num_classes):
        n = len(sorted_trainset[i])
                                                                                                                                                                                                                                                                    
        p_list = [((1-p)*num_classes)/((num_classes-1)*args.num_workers)] * args.num_workers

        if i*args.num_workers % num_classes != 0:
            start_idx = int(i*args.num_workers/num_classes) + 1
            p_list[start_idx-1] = ((1-p)*num_classes)/((num_classes-1)*args.num_workers)*(i*args.num_workers/num_classes-start_idx+1) + \
                p*num_classes/args.num_workers * (start_idx - i*args.num_workers/num_classes)
        else:
            start_idx = int(i*args.num_workers/num_classes)

        if (i+1)*args.num_workers % num_classes != 0:
            end_idx = int((i+1)*args.num_workers/num_classes)
            p_list[end_idx] = p*num_classes/args.num_workers * ((i+1)*args.num_workers/num_classes-end_idx) + \
                ((1-p)*num_classes)/((num_classes-1)*args.num_workers)*(1 - (i+1)*args.num_workers/num_classes + end_idx)
        else:
            end_idx = int(start_idx + args.num_workers/num_classes)
        
        for k in range(start_idx, end_idx):
            p_list[k] = p*num_classes/args.num_workers



        length = [pro * n for pro in p_list]
        length = [int(e) for e in length]
        if sum(length) > n:
            length = (np.array(length) - int( (sum(length) - n)/args.num_workers ) -1).tolist()
        length_list.append(length)

    partition = []
    for i in range(args.num_workers):
        dataset = []
        for j in range(num_classes):
            start_idx = sum(length_list[j][:i])
            end_idx = start_idx + length_list[j][i]

            dataset += [sorted_trainset[j][k] for k in range(start_idx, end_idx)]
        partition.append(dataset)
    return partition


def num_noniid_split(dataset, args,min_num,max_num):
    """
    Sample non-I.I.D client data from dataset
    -> Different clients can hold vastly different amounts of data
    :param dataset:
    :param num_users:
    :return:
    """
    num_dataset = len(dataset)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(args.num_workers)}

    random_num_size = np.random.randint(min_num, max_num + 1, size=args.num_workers)
    print(f"Total number of datasets owned by clients : {sum(random_num_size)}")

    # total dataset should be larger or equal to sum of splitted dataset.
    assert num_dataset >= sum(random_num_size)

    # divide and assign
    partition = []
    for i, rand_num in enumerate(random_num_size):
        rand_set = set(np.random.choice(idx, rand_num, replace=False))
        idx = list(set(idx) - rand_set)
        dict_users[i] = rand_set
        #my_dict = {val: idx for idx, val in enumerate(rand_set)}
        #indices = [my_dict[val] for val in rand_set]
        # print("rand_set",rand_set)
        # print("indices", list(rand_set))
        # print("dataset", dataset[0])
        partition.append([dataset[i] for i in rand_set])
    return partition





def split_dataset(args, dataset):
    """

    Parameters
    ----------
    args: ags for datasets
    dataset: TUDatasets [graph,labels]

    Returns
    -------
    participation data for each clients:
    [train_client_0,train_client_1,...,test_client_0,test_client_1,...]
    """
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #########################################
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    dataset_all = dataset.train[0] + dataset.val[0] + dataset.test[0]

    graph_sizes = []
    for data in dataset_all:
        graph_sizes.append(data[0].num_nodes())
    graph_sizes.sort()
    n = int(0.3*len(graph_sizes))
    graph_size_normal = graph_sizes[n:len(graph_sizes)-n]
    count = 0
    for size in graph_size_normal:
        count += size
    avg_nodes = count / len(graph_size_normal)
    avg_nodes = round(avg_nodes)
    ###################precious version all the cleient has the same test datasets
    # total_size = len(dataset_all)
    # test_size = int(total_size/(4*args.num_workers+1)) # train size : test size = 4 : 1
    # train_size = total_size - test_size
    # client_num = int(train_size/args.num_workers)
    # length = [client_num]*(args.num_workers-1)
    #
    # length.append(train_size-(args.num_workers-1)*client_num)
    #
    # length.append(test_size)
    #####################changed each client has a different test data different from the precious version that each version has the same testdata
    #length: [client_1_train, client_2_train,...,client_1_test_,client_3_test,...]

    total_size = len(dataset_all)
    test_size = int(total_size/(4*args.num_workers+1*args.num_workers)) # train size : test size = 4 : 1
    train_size = total_size - test_size*args.num_workers
    client_num = int(train_size/args.num_workers)
    length = [client_num]*(args.num_workers-1)
    length.append(train_size-(args.num_workers-1)*client_num)
    for i in range(args.num_workers-1):
        length.append(test_size)
    length.append(total_size - train_size - test_size*(args.num_workers-1))


    ##################################
    # return the adverage degree of nodes among all graphs
    sum_avg_degree = 0
    for data in dataset_all:
        # Get the degree of each node
        degrees = data[0].in_degrees()
        # Calculate the average degree
        avg_degree = degrees.float().mean().item()
        sum_avg_degree += avg_degree
    args.avg_degree = int(sum_avg_degree / len(graph_sizes))
    if args.is_iid == "iid":
        # iid splitq
        partition_data = random_split(dataset_all, length) # split training data and test data
    elif args.is_iid == "p-degree-non-iid":
        # p-degree-non-iid: Local Model Poisoning Attacks to Byzantine-Robust Federated Learning
        # non-iid split
        total_size = len(dataset_all)
        test_size = int(total_size / (4 * args.num_workers + 1 * args.num_workers))  # train size : test size = 4 : 1
        total_train_size = total_size - test_size * args.num_workers
        total_test_size = test_size * args.num_workers
        length = [total_train_size, total_test_size]
        trainset, testset = random_split(dataset_all, length)
        train_partition_data = p_degree_non_iid_split(trainset, args, num_classes)
        test_partition_data = p_degree_non_iid_split(testset, args, num_classes)
        for k in range(len(test_partition_data)):
            train_partition_data.append(test_partition_data[k])
        partition_data = train_partition_data
    elif args.is_iid == "num-non-iid":
        # p-degree-non-iid: Local Model Poisoning Attacks to Byzantine-Robust Federated Learning
        # non-iid split
        total_size = len(dataset_all)
        test_size = int(total_size / (4 * args.num_workers + 1 * args.num_workers))  # train size : test size = 4 : 1
        total_train_size = total_size - test_size * args.num_workers
        total_test_size = test_size * args.num_workers
        length = [total_train_size, total_test_size]
        trainset, testset = random_split(dataset_all, length)
        train_min_num = int(0.6 * (total_train_size / args.num_workers))
        train_max_num = int(0.9 * (total_train_size / args.num_workers))
        test_min_num = int(0.6 * (test_size))
        test_max_num = int(0.9 * (test_size))
        train_partition_data = num_noniid_split(trainset, args, min_num= train_min_num, max_num= train_max_num)
        test_partition_data = num_noniid_split(testset, args, min_num= test_min_num, max_num= test_max_num)
        for k in range(len(test_partition_data)):
            train_partition_data.append(test_partition_data[k])
        partition_data = train_partition_data
    else:
        raise  NameError

    return partition_data, avg_nodes

def adj_to_edge_index(adj):
    N = adj.shape[0]  # get number of nodes

    # find non-zero entries in the adjacency matrix
    rows, cols = torch.where(adj != 0)

    # create edge index tensor
    edge_index = torch.stack([rows, cols], dim=0)

    self_loops = False
    # add self-loops (if desired)
    if self_loops:
        self_loop_index = torch.arange(N)
        edge_index = torch.cat([edge_index, torch.stack([self_loop_index, self_loop_index])], dim=1)
    return edge_index

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.nn import GraphConv
class Trojan_GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Trojan_GCN, self).__init__()
        if output_dim < 3:
            output_dim = 3
        self.conv1 = GraphConv(input_dim, 2*input_dim)
        self.conv2 = GraphConv(2*input_dim, input_dim)
        self.feat = nn.Linear(input_dim,input_dim)
        self.feat1 = nn.Linear(input_dim,input_dim)
        # self.edge_weights = nn.Parameter(torch.randn(nfeat, 1, device=device))  # 初始化为随机值
        self.edge_weights = nn.Linear(input_dim, output_dim)
        self.edge_weights1 = nn.Linear(output_dim, output_dim)
        
    def forward(self, g, features, trigger_nodes):
        # 第一层GCN
        h = self.conv1(g, features)
        h = F.relu(h)
        # 第二层GCN
        h = self.conv2(g, h)
        h = F.relu(h)
        feat = self.feat(h[trigger_nodes])
        feat = F.relu(feat)
        feat = self.feat1(feat)
        edge_weight = self.edge_weights(h[trigger_nodes])
        edge_weight = F.relu(edge_weight)
        edge_weight = self.edge_weights1(edge_weight)

        return feat, edge_weight




class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()
        # print('nfeattt,',nfeat)
        if nout < 3:
            nout = 3
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)
        # print(f"nout: {nout}, type(nout): {type(nout)}")
        self.feat = nn.Linear(nfeat,nfeat)
        # self.edge_weights = nn.Parameter(torch.randn(nfeat, 1, device=device))  # 初始化为随机值
        self.edge_weights = nn.Linear(nfeat, nout)
        self.device = device

    def forward(self, input):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """


        self.layers = self.layers
        h = self.layers(input)
        # print('input,',input.shape)
        # print('hhhhhh,',h.shape)
        feat = self.feat(h)
        edge_weight = self.edge_weights(h)
        # feat = GW(feat, thrd, self.device)
        # edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight


class HomoLoss(nn.Module):
    def __init__(self, args, device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self, trigger_edge_index, x, thrd):
        # trigger_edge_index 是一个二维张量，其中包含触发器边的源节点和目标节点索引
        # x 是节点的特征矩阵
        # thrd 是用于计算损失的阈值

        # 使用触发器边索引来计算边的相似度
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]], x[trigger_edge_index[1]])
        # print("Edge similarity: ",edge_sims)
        
        # 计算损失，只有当边的相似度小于阈值时，损失才为正
        loss = torch.relu(thrd - edge_sims).mean()
        
        return loss

class Backdoor:
    def __init__(self, args, device, pre_trained_model, nfeat, avg_nodes, net_params1):
        self.args = args
        self.device = device
        self.model = pre_trained_model
        # self.trojan = GraphTrojanNet(self.device, nfeat, int(avg_nodes * args.frac_of_avg), layernum=2).to(self.device)
        self.trojan = Trojan_GCN(nfeat, int(avg_nodes * args.frac_of_avg)).to(self.device)
        self.homo_loss = HomoLoss(self.args, self.device)
        self.optimizer = optim.Adam(self.trojan.parameters(), lr=args.lr/10, weight_decay=args.weight_decay)

    def fit(self, model, trainset, args, avg_nodes):
        torch.autograd.set_detect_anomaly(True)
        self.trojan.train()
        train_untarget_idx = []
        for i in range(len(trainset)):
            if trainset[i][1].item() != args.target_label:
                train_untarget_idx.append(i)

        train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
        train_labels = [graph[1] for graph in trainset]
        num_classes = torch.max(torch.tensor(train_labels)).item() + 1
        
        tmp_graphs = []
        tmp_idx = []
        num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
        # print('num_trigger_nodes,',num_trigger_nodes)
        # print('average nodes in graph,',avg_nodes)
        for idx, graph in enumerate(train_untarget_graphs):
            if graph[0].num_nodes() > num_trigger_nodes:
                tmp_graphs.append(graph)
                tmp_idx.append(train_untarget_idx[idx])
        if args.dataset == "COLORS-3":
            args.poisoning_intensity = 0.01
        n_trigger_graphs = int(args.poisoning_intensity*len(trainset))
        final_idx = []
        if n_trigger_graphs <= len(tmp_graphs):
            train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
            final_idx = tmp_idx[:n_trigger_graphs]

        else:
            train_trigger_graphs = tmp_graphs
            final_idx = tmp_idx

        print("Start generating trigger position by {}".format(args.trigger_position))
        default_min_num_trigger_nodes = 3
        if num_trigger_nodes < default_min_num_trigger_nodes:
            num_trigger_nodes = default_min_num_trigger_nodes            
        
        trigger_list = []

        if args.trigger_position == "random":
            for data in train_trigger_graphs:
                # print("data[0].nodes().tolist()",len(data[0].nodes().tolist()))
                # print("num trigger nodes", num_trigger_nodes)
                if len(data[0].nodes().tolist()) < num_trigger_nodes:
                    trigger_num = data[0].nodes().tolist()
                else:
                    trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
                trigger_list.append(trigger_num)
        elif args.trigger_position == "degree":
            for data in train_trigger_graphs:
                #  transfer data to Network graph
                g = dgl.to_networkx(data[0].cpu())
                # sort according to degree
                degree_dict = dict(g.degree())
                sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
                trigger_num = sorted_nodes[:num_trigger_nodes]
                trigger_list.append(trigger_num)
        elif args.trigger_position == "cluster":
            for data in train_trigger_graphs:
                #  transfer data to Network graph
                g = dgl.to_networkx(data[0].cpu())
                #  sort according to cluster
                simple_g = nx.Graph(g)
                clustering_dict = nx.clustering(simple_g,weight='weight')
                sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)

                trigger_num = sorted_nodes[:num_trigger_nodes]
                trigger_list.append(trigger_num)
        else:
            raise NameError

        print("Start preparing for the poisoned test datasets")
        count = 0
        for epoch in range(self.args.trojan_epochs):
            loss = 0
            correct = 0  # 用于跟踪正确预测的数量
            total = 0  # 用于跟踪总样本数量
            for  i, data in enumerate(train_trigger_graphs):

                # 创建图的副本
                g_copy = data[0].clone().to(self.device)
                g_copy = dgl.add_self_loop(g_copy)
                labels = torch.tensor([args.target_label]).to(self.device)

                #随机选取一部分节点作为触发节点
                # trigger_nodes = torch.randperm(g_copy.number_of_nodes())[:num_trigger_nodes].to(self.device)
                trigger_nodes = trigger_list[i]
                # print('trigger_nodes,',trigger_nodes)


                # 生成触发器特征和权重
                # trojan_feat, trojan_weights = self.trojan(g_copy.ndata['feat'][trigger_nodes], self.args.thrd)

                # trojan_feat, trojan_weights = self.trojan(g_copy, g_copy.ndata['feat'], trigger_nodes)
                trojan_feat, trojan_weights= self.trojan(g_copy, g_copy.ndata['feat'], trigger_nodes)                
                # print("trojan_weights shape: ",trojan_weights.shape)
                # print("g_copy.ndata['feat']",g_copy.ndata['feat'].shape)
                # print(f"trojan_feat shape: {trojan_feat.shape}")
                # 将触发器特征注入到副本图中
                trojan_feat = trojan_feat.view([-1, g_copy.ndata['feat'].shape[1]])
                # print(f"trigger_nodes shape: {trigger_nodes.shape}")
                # print(f"trojan_feat shape: {trojan_feat.shape}")
                g_copy.ndata['feat'][trigger_nodes] = trojan_feat

                trojan_edge = []
                # print("Start injecting trigger into the poisoned train datasets")

                for node1 in range(num_trigger_nodes):
                    for node2 in range(node1,num_trigger_nodes):  # 每个触发器节点有5个潜在的边
                        # print('node2,',node2)
                        if trojan_weights[node1][node2] > args.weight_threshold and trigger_nodes[node1] != trigger_nodes[node2]:
                            # 添加边，权重设置为 trojan_weights[node][edge]
                            # g_copy = dgl.add_edges(g_copy, trigger_nodes[node1], trigger_nodes[node2])
                            node1_tensor = torch.tensor([trigger_nodes[node1]], device=self.device)
                            node2_tensor = torch.tensor([trigger_nodes[node2]], device=self.device)
                            g_copy.add_edges(node1_tensor, node2_tensor)
                            g_copy.add_edges(node2_tensor, node1_tensor)
                            trojan_edge.append((trigger_nodes[node1], trigger_nodes[node2]))
                            trojan_edge.append((trigger_nodes[node2], trigger_nodes[node1]))
                trojan_edges = torch.tensor(trojan_edge, device=self.device).t()
                # print('trojan_edges,',len(trojan_edges[0]),trojan_edges)
                


                # 前向传播和计算损失
                output = model.forward(g_copy, g_copy.ndata['feat'], g_copy.edata['feat']).to(self.device)
                # print('output,',output)
                # loss_target = self.args.target_loss_weight * F.nll_loss(output, labels)
                loss_target = self.args.target_loss_weight * model.loss(output, labels)
                
                # 计算准确率
                output = F.softmax(output, dim=1)
                # print('output,',output)
                _, predicted = torch.max(output, 1)  # 使用exp()来获得概率分布
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # print('loss_target,',loss_target)
                loss_homo = 0.0
                if self.args.homo_loss_weight > 0 and len(trojan_edge) > 0:
                    loss_homo = self.homo_loss(trojan_edges, g_copy.ndata['feat'], self.args.homo_boost_thrd)
                # print('loss_homo,',loss_homo)
                loss = loss_target + self.args.homo_loss_weight * loss_homo + loss
                # print(i,loss)
                # print("num_of_test_trigger_graphs is: %d"%len(train_trigger_graphs))

            # 反向传播和优化
            # print(f"Before backward, loss: {loss.item()}")
            self.optimizer.zero_grad()
            loss.backward()  # 在这里只调用一次backward
            # print(f"After backward, loss: {loss.item()}")
            self.optimizer.step()

            accuracy = 100 * correct / total
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}%')
            if accuracy >= 99.9:
                count = count + 1
                if count >=5:
                    break


class Backdoor1:
    def __init__(self, args, device, pre_trained_model, nfeat, avg_nodes, net_params1):
        self.args = args
        self.device = device
        self.model = pre_trained_model
        self.trojan = GraphTrojanNet(self.device, nfeat, int(avg_nodes * args.frac_of_avg), layernum=2).to(self.device)
        # self.trojan = Trojan_GAT(net_params1, int(avg_nodes * args.frac_of_avg)).to(self.device)
        self.homo_loss = HomoLoss(self.args, self.device)
        self.optimizer = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def fit(self, model, trainset, args, avg_nodes):
        torch.autograd.set_detect_anomaly(True)
        self.trojan.train()
        train_untarget_idx = []
        for i in range(len(trainset)):
            if trainset[i][1].item() != args.target_label:
                train_untarget_idx.append(i)

        train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
        train_labels = [graph[1] for graph in trainset]
        num_classes = torch.max(torch.tensor(train_labels)).item() + 1
        
        tmp_graphs = []
        tmp_idx = []
        num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
        # print('num_trigger_nodes,',num_trigger_nodes)
        # print('average nodes in graph,',avg_nodes)
        for idx, graph in enumerate(train_untarget_graphs):
            if graph[0].num_nodes() > num_trigger_nodes:
                tmp_graphs.append(graph)
                tmp_idx.append(train_untarget_idx[idx])
        if args.dataset == "COLORS-3":
            args.poisoning_intensity = 0.005
        n_trigger_graphs = int(args.poisoning_intensity*len(trainset))
        final_idx = []
        if n_trigger_graphs <= len(tmp_graphs):
            train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
            final_idx = tmp_idx[:n_trigger_graphs]

        else:
            train_trigger_graphs = tmp_graphs
            final_idx = tmp_idx

        print("Start generating trigger position by {}".format(args.trigger_position))
        default_min_num_trigger_nodes = 3
        if num_trigger_nodes < default_min_num_trigger_nodes:
            num_trigger_nodes = default_min_num_trigger_nodes            
        
        trigger_list = []

        if args.trigger_position == "random":
            for data in train_trigger_graphs:
                # print("data[0].nodes().tolist()",len(data[0].nodes().tolist()))
                # print("num trigger nodes", num_trigger_nodes)
                if len(data[0].nodes().tolist()) < num_trigger_nodes:
                    trigger_num = data[0].nodes().tolist()
                else:
                    trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
                trigger_list.append(trigger_num)
        elif args.trigger_position == "degree":
            for data in train_trigger_graphs:
                #  transfer data to Network graph
                g = dgl.to_networkx(data[0].cpu())
                # sort according to degree
                degree_dict = dict(g.degree())
                sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
                trigger_num = sorted_nodes[:num_trigger_nodes]
                trigger_list.append(trigger_num)
        elif args.trigger_position == "cluster":
            for data in train_trigger_graphs:
                #  transfer data to Network graph
                g = dgl.to_networkx(data[0].cpu())
                #  sort according to cluster
                simple_g = nx.Graph(g)
                clustering_dict = nx.clustering(simple_g,weight='weight')
                sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)

                trigger_num = sorted_nodes[:num_trigger_nodes]
                trigger_list.append(trigger_num)
        else:
            raise NameError

        print("Start preparing for the poisoned test datasets")
        count = 0
        for epoch in range(self.args.trojan_epochs):
            loss = 0
            correct = 0  # 用于跟踪正确预测的数量
            total = 0  # 用于跟踪总样本数量
            for  i, data in enumerate(train_trigger_graphs):
                # 创建图的副本
                g_copy = data[0].clone().to(self.device)
                
                labels = torch.tensor([args.target_label]).to(self.device)

                #随机选取一部分节点作为触发节点
                # trigger_nodes = torch.randperm(g_copy.number_of_nodes())[:num_trigger_nodes].to(self.device)
                trigger_nodes = trigger_list[i]
                # print('trigger_nodes,',trigger_nodes)


                # 生成触发器特征和权重
                # trojan_feat, trojan_weights = self.trojan(g_copy.ndata['feat'][trigger_nodes], self.args.thrd)

                # trojan_feat, trojan_weights = self.trojan(g_copy, g_copy.ndata['feat'], trigger_nodes)
                trojan_feat, trojan_weights= self.trojan(g_copy.ndata['feat'][trigger_nodes])                
                # print("trojan_weights shape: ",trojan_weights.shape)
                # print(f"trojan_feat shape: {trojan_feat.shape}")
                # 将触发器特征注入到副本图中
                trojan_feat = trojan_feat.view([-1, g_copy.ndata['feat'].shape[1]])
                # print(f"trigger_nodes shape: {trigger_nodes.shape}")
                # print(f"trojan_feat shape: {trojan_feat.shape}")
                g_copy.ndata['feat'][trigger_nodes] = trojan_feat

                trojan_edge = []
                # print("Start injecting trigger into the poisoned train datasets")

                for node1 in range(num_trigger_nodes):
                    for node2 in range(node1,num_trigger_nodes):  # 每个触发器节点有5个潜在的边
                        # print('node2,',node2)
                        if trojan_weights[node1][node2] > args.weight_threshold and trigger_nodes[node1] != trigger_nodes[node2]:
                            # 添加边，权重设置为 trojan_weights[node][edge]
                            # g_copy = dgl.add_edges(g_copy, trigger_nodes[node1], trigger_nodes[node2])
                            node1_tensor = torch.tensor([trigger_nodes[node1]], device=self.device)
                            node2_tensor = torch.tensor([trigger_nodes[node2]], device=self.device)
                            g_copy.add_edges(node1_tensor, node2_tensor)
                            g_copy.add_edges(node2_tensor, node1_tensor)
                            trojan_edge.append((trigger_nodes[node1], trigger_nodes[node2]))
                            trojan_edge.append((trigger_nodes[node2], trigger_nodes[node1]))
                trojan_edges = torch.tensor(trojan_edge, device=self.device).t()
                # print('trojan_edges,',len(trojan_edges[0]),trojan_edges)
                


                # 前向传播和计算损失
                output = model.forward(g_copy, g_copy.ndata['feat'], g_copy.edata['feat']).to(self.device)
                # print('output,',output)
                # loss_target = self.args.target_loss_weight * F.nll_loss(output, labels)
                loss_target = self.args.target_loss_weight * model.loss(output, labels)
                
                # 计算准确率
                output = F.softmax(output, dim=1)
                # print('output,',output)
                _, predicted = torch.max(output, 1)  # 使用exp()来获得概率分布
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # print('loss_target,',loss_target)
                loss_homo = 0.0
                # if self.args.homo_loss_weight > 0 and len(trojan_edge) > 0:
                #     loss_homo = self.homo_loss(trojan_edges, g_copy.ndata['feat'], self.args.homo_boost_thrd)
                # print('loss_homo,',loss_homo)
                loss = loss_target + self.args.homo_loss_weight * loss_homo + loss
                # print(i,loss)
                # print("num_of_test_trigger_graphs is: %d"%len(train_trigger_graphs))

            # 反向传播和优化
            # print(f"Before backward, loss: {loss.item()}")
            self.optimizer.zero_grad()
            loss.backward()  # 在这里只调用一次backward
            # print(f"After backward, loss: {loss.item()}")
            self.optimizer.step()

            accuracy = 100 * correct / total
            print(f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy:.2f}%')
            if accuracy >= 97.9 or (epoch >=800 and accuracy >= 95.0) or (epoch >=1400 and accuracy >= 93.0):
                count = count + 1
                if count >=10:
                    break

def get_poisoned_dataloader_with_global_trigger(dataloader, avg_nodes, args, client, device):

    
    test_changed_graphs = [copy.deepcopy(graph) for graph in dataloader if graph[1].item() != args.target_label]
        
    num_mali = args.num_mali
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg) * num_mali
    default_min_num_trigger_nodes = 3 * num_mali
    if num_trigger_nodes < default_min_num_trigger_nodes:
            num_trigger_nodes = default_min_num_trigger_nodes   
    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    print("num_of_test_changed_graphs is: %d"%len(test_changed_graphs_final))
    each_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    poisoned_loader = []
    # num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    graphs = []
    trigger_list = []
    if args.trigger_position == "random":
        for data in test_changed_graphs:
                # print("data[0].nodes().tolist()",len(data[0].nodes().tolist()))
                # print("num trigger nodes", num_trigger_nodes)
            if len(data[0].nodes().tolist()) < num_trigger_nodes:
                trigger_num = data[0].nodes().tolist()
            else:
                trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
            trigger_list.append(trigger_num)
    elif args.trigger_position == "degree":
        for data in test_changed_graphs:
                #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            # sort according to degree
            degree_dict = dict(g.degree())
            sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    elif args.trigger_position == "cluster":
        for data in test_changed_graphs:
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
                #  sort according to cluster
            simple_g = nx.Graph(g)
            clustering_dict = nx.clustering(simple_g,weight='weight')
            sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)

            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    else:
        raise NameError
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    default_min_num_trigger_nodes = 3
    if num_trigger_nodes < default_min_num_trigger_nodes:
            num_trigger_nodes = default_min_num_trigger_nodes   
    for  i, data in enumerate(test_changed_graphs):
        
        g_copy = data[0].clone().to(device)
        g_copy = dgl.add_self_loop(g_copy)
            
            
        trigger_nodes_list = []

        # 生成所有点集
        # all_trigger_nodes = torch.randperm(g_copy.number_of_nodes())[:args.num_mali * num_trigger_nodes].to(device)
        for j in range(0, len(trigger_list[i]), num_trigger_nodes):
            trigger_nodes_list.append(trigger_list[i][j:j+num_trigger_nodes])
        # print('trigger_nodes_list',trigger_nodes_list)


        for j in range(num_mali):    
            trigger_nodes = trigger_nodes_list[j]
            # print('trigger_nodes',trigger_nodes)
            # print('g_copy.ndata[\'feat\'][trigger_nodes]',g_copy.ndata['feat'][trigger_nodes].shape)
            # trojan_feat, trojan_weights = client[j].backdoor.trojan(g_copy.ndata['feat'][trigger_nodes], args.thrd)
            trojan_feat, trojan_weights = client[j].backdoor.trojan(g_copy.ndata['feat'][trigger_nodes])
            # print('g_copy.ndata[\'feat\'][trigger_nodes]',g_copy.ndata['feat'][trigger_nodes].shape)
            trojan_feat = trojan_feat.view([-1, data[0].ndata['feat'].shape[1]])
            # # g_copy.ndata['feat'][trigger_nodes] = trojan_feat.to(device)
            # print('trojan_feat',trojan_feat.shape)
            # print('g_copy.ndata[\'feat\'][trigger_nodes]',g_copy.ndata['feat'][trigger_nodes].shape)
            g_copy.ndata['feat'][trigger_nodes] = trojan_feat
            for node1 in range(num_trigger_nodes):
                for node2 in range(node1,num_trigger_nodes):  # 每个触发器节点有5个潜在的边
                        # print('node2,',node2)
                    if trojan_weights[node1][node2] > args.weight_threshold and trigger_nodes[node1] != trigger_nodes[node2]:
                            # 添加边，权重设置为 trojan_weights[node][edge]
                            # g_copy = dgl.add_edges(g_copy, trigger_nodes[node1], trigger_nodes[node2])
                        node1_tensor = torch.tensor([trigger_nodes[node1]], device = device)
                        node2_tensor = torch.tensor([trigger_nodes[node2]], device = device)
                        g_copy.add_edges(node1_tensor, node2_tensor)
                        g_copy.add_edges(node2_tensor, node1_tensor)
        graphs.append(g_copy)
    
    # graphs = [data[0] for data in test_changed_graphs]
    
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)


    return test_trigger_graphs

def get_poisoned_dataloader_with_local_trigger(dataloader, avg_nodes, args, client, device, num_ID):

    
    test_changed_graphs = [copy.deepcopy(graph) for graph in dataloader if graph[1].item() != args.target_label]
        
    num_mali = args.num_mali
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    print("num_of_test_changed_graphs is: %d"%len(test_changed_graphs_final))
    each_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    poisoned_loader = []
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    graphs = []
    trigger_list = []
    if args.trigger_position == "random":
        for data in test_changed_graphs:
                # print("data[0].nodes().tolist()",len(data[0].nodes().tolist()))
                # print("num trigger nodes", num_trigger_nodes)
            if len(data[0].nodes().tolist()) < num_trigger_nodes:
                trigger_num = data[0].nodes().tolist()
            else:
                trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
            trigger_list.append(trigger_num)
    elif args.trigger_position == "degree":
        for data in test_changed_graphs:
                #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
            # sort according to degree
            degree_dict = dict(g.degree())
            sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    elif args.trigger_position == "cluster":
        for data in test_changed_graphs:
            #  transfer data to Network graph
            g = dgl.to_networkx(data[0].cpu())
                #  sort according to cluster
            simple_g = nx.Graph(g)
            clustering_dict = nx.clustering(simple_g,weight='weight')
            sorted_nodes = sorted(clustering_dict, key=clustering_dict.get, reverse=True)

            trigger_num = sorted_nodes[:num_trigger_nodes]
            trigger_list.append(trigger_num)
    else:
        raise NameError
    for  i, data in enumerate(test_changed_graphs):
        
        g_copy = data[0].clone().to(device)
        g_copy = dgl.add_self_loop(g_copy)
            
          
        trigger_nodes = trigger_list[i]
        # trigger_nodes.to(device)
        # trojan_feat, trojan_weights = client[num_ID].backdoor.trojan(g_copy.ndata['feat'][trigger_nodes], args.thrd)
        trojan_feat, trojan_weights = client[num_ID].backdoor.trojan(g_copy.ndata['feat'][trigger_nodes])
        trojan_feat = trojan_feat.view([-1, data[0].ndata['feat'].shape[1]])
        g_copy.ndata['feat'][trigger_nodes] = trojan_feat.to(device)
        for node1 in range(num_trigger_nodes):
            for node2 in range(node1,num_trigger_nodes):  # 每个触发器节点有5个潜在的边
                    # print('node2,',node2)
                if trojan_weights[node1][node2] > args.weight_threshold and trigger_nodes[node1] != trigger_nodes[node2]:
                        # 添加边，权重设置为 trojan_weights[node][edge]
                        # g_copy = dgl.add_edges(g_copy, trigger_nodes[node1], trigger_nodes[node2])
                    node1_tensor = torch.tensor([trigger_nodes[node1]], device = device)
                    node2_tensor = torch.tensor([trigger_nodes[node2]], device = device)
                    g_copy.add_edges(node1_tensor, node2_tensor)
                    g_copy.add_edges(node2_tensor, node1_tensor)
        graphs.append(g_copy)
    print('graphs',len(graphs))

    
    # graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    print('labels',len(labels))
    test_trigger_graphs = DGLFormDataset(graphs, labels)


    return test_trigger_graphs


