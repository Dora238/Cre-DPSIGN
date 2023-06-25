import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
# from torch_kmeans import KMeans
# from Config import sgdConfig, device
from LoadData import getData_mnist
from MainModel import MLP, flatten_list, unflatten_vector, caL_loss_acc, mean, gm
from Attack import same_value, sign_flipping, zero_gradient, sample_duplicating,gauss_attack
from options import args_parser, exp_details
# from Sample import a_res
import numpy as np
from sklearn.cluster import KMeans
from torch.nn.functional import cosine_similarity


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("code_neural_network") + len("code_neural_network")]


def centralized_sgd(setting, attack=None, save_model=False, save_results=False):
    """

    :options.py -> paraparam setting: 'iid','attack','learning rate','num of workers','num of byzantine workers'
    :Attack.py -> all types of attack
    """

    # initialize the global model and data loader
    # epsilon:privacy budget
    args = args_parser()
    exp_details(args)
    lambda0 = 0.01
    u = 0.004
    epsilon = args.eps
    Gmax = 0.01
    du = 2 * args.lr * Gmax
    # gamma = np.exp(epsilon) / (np.exp(epsilon) + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = [MLP(784, 50, 50, 10).to(device)
             for _ in range(args.num_users)]
    model0 = MLP(784, 50, 50, 10).to(device)
    train_dataset_subset, train_loader, test_loader = getData_mnist(setting, args)
    train_loaders_splited = [
        torch.utils.data.DataLoader(dataset=subset, batch_size=args.batchsize, shuffle=True, pin_memory=True)
        for subset in train_dataset_subset
    ]
    train_loaders_splited_iter = [iter(loader) for loader in train_loaders_splited]

    train_loss_list = []
    test_acc_list = []
    byzantine = []
    regular = []
    sum_credit = []
    worker_credit = []

    for i in range(args.num_users):
        worker_credit.append([i, 1, 0])

    for i in range(args.num_users):
        sum_credit.append([worker_credit[i][0], worker_credit[i][1]])

    # initialize the set of gradients
    worker_grad = [
        [torch.zeros_like(para, requires_grad=False) for para in model0.parameters()]
        for _ in range(args.num_users)
    ]
    worker_model = [
        [torch.zeros_like(para, requires_grad=False) for para in model0.parameters()]
        for _ in range(args.num_users)
    ]
    master_grad = [torch.zeros_like(para, requires_grad=False) for para in model0.parameters()]

    signinfo = [
        [torch.zeros_like(para, requires_grad=False) for para in model0.parameters()]
        for _ in range(args.num_users)
    ]

    #generate byzantine workers
    workers = [i for i in range(args.num_users)]
    byzantine = random.sample(range(args.num_users), args.byzantinue_users)
    regular = list(set(range(args.num_users)).difference(byzantine))
    print(byzantine)
    # print(regular)

    # start training
    for iteration in range(1, args.iterations + 1):
        print('Train iteration: {}'.format(iteration))

        # The total training rounds are 5000 rounds.
        # Starting from the first round, sample and global aggregate every 50 rounds.
        if (iteration == 1) or (iteration % 50 == 0):
            count =0
            model0.train()
            for id in regular:
                model[id].train()
                if setting == 'iid':
                    try:
                        batch_iterator = train_loaders_splited_iter[id]
                        data, target = next(batch_iterator)
                    except StopIteration:
                        train_loaders_splited_iter = [iter(loader) for loader in train_loaders_splited]
                        batch_iterator = train_loaders_splited_iter[id]
                        data, target = next(batch_iterator)
                elif setting == 'noniid':
                    try:
                        batch_iterator = train_loaders_splited_iter[count]
                        data, target = next(batch_iterator)
                    except StopIteration:
                        train_loaders_splited_iter = [iter(loader) for loader in train_loaders_splited]
                        batch_iterator = train_loaders_splited_iter[count]
                        data, target = next(batch_iterator)
                    count += 1
                data, target = data.to(device), target.to(device)
                output = model[id](data)
                loss = F.nll_loss(output, target)
                # autograd
                model[id].zero_grad()
                loss.backward()
                # obtain the sign information
                for index, (para, para0) in enumerate(zip(model[id].parameters(), model0.parameters())):
                    signinfo[id][index].data.zero_()
                    # add gaussian noise to the sign gradient(CreGau)
                    sigma = torch.maximum(2 / 3 * (para0.data - para.data), 4 * du / epsilon * torch.ones_like(para))
                    gauss = torch.randn_like(para) * sigma
                    signinfo[id][index].data.add_(torch.sign(para0.data - para.data + gauss), alpha=1)
                for index, (para, signvalue) in enumerate(zip(model[id].parameters(), signinfo[id])):
                    worker_grad[id][index].data.zero_()
                    worker_grad[id][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[id][index].data.add_(para, alpha=args.decayWeight)
                    worker_grad[id][index].data.add_(signvalue.data, alpha=-lambda0)
                for index, para in enumerate(model[id].parameters()):
                    worker_model[id][index].data.zero_()
                    worker_model[id][index].data.add_(para.data, alpha=1)
                for para, grad in zip(model[id].parameters(), worker_grad[id]):
                    para.data.add_(grad, alpha=-args.lr)
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_model_flat = flatten_list(worker_model)
            if attack != None:
                worker_model_flat = attack(worker_model_flat, regular, byzantine)
            for id in byzantine:
                worker_model[id] = unflatten_vector(worker_model_flat[id], model0)
                for index, (para0, paraby) in enumerate(zip(model0.parameters(), worker_model[id])):
                    signinfo[id][index].data.zero_()
                    signinfo[id][index].data.add_(torch.sign(para0.data - paraby.data), alpha=1)

            signinfo_flat = flatten_list(signinfo)
            print(signinfo_flat)
            # cluster based on cos_similarity of worker's gradient
            cluster_num = 65
            kmeans = KMeans(n_clusters=cluster_num)
            cos_sim = cosine_similarity(signinfo_flat.unsqueeze(1),signinfo_flat.unsqueeze(0),dim=2)
            kmeans.fit(cos_sim.cpu())
            cre_cluster_sum = [0 for i in range(cluster_num)]
            # compress clusters from 65 to 60
            clusters = []
            for i in range(kmeans.n_clusters):
                idxs = np.where(kmeans.labels_ == i)[0]
                cluster = []
                for idx in idxs:
                    cre_cluster_sum[i] += worker_credit[idx][1]
                    cluster.append(idx)
                clusters.append(cluster)
            max_index = list(np.argsort(cre_cluster_sum, axis=-1, kind='quicksort', order=None))[-1:-61:-1]

            # computer total credits of each cluster
            for _index in range(len(clusters)):
                if _index not in max_index:
                    clusters[max_index[0]] = clusters[max_index[0]] + clusters[_index]
                    break
            # sample on each cluster
            sample_user = []
            for i in max_index:
                idxs = np.where(kmeans.labels_ == i)[0]
                weight_user = []
                for idx in idxs:
                    weight_user.append([idx, worker_credit[idx][1]])
                sum_weight = np.sum(weight_user, axis=0)

                weights = [_weight_user[1] for _weight_user in weight_user]
                probs = [weight / sum_weight[1] for weight in weights]
                choice = np.random.choice(idxs, size=1, p=probs)
                sample_user.append(choice[0])
            print(sample_user)

            # Reward Worker based on samesum
            signinfo_flat = torch.sum(signinfo_flat, dim=0)
            signmaster = unflatten_vector(signinfo_flat, model0)
            flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
            sum_list = [0 for i in range(args.num_users)]
            for user in workers:
                usrsigninfolist = []
                for p in signinfo[user]:
                    usrsigninfolist.append(p.tolist())
                usrsigninfolist = flatten(usrsigninfolist)
                total_signinfo_list = signinfo_flat.tolist()
                samesum = 0
                for i in range(len(usrsigninfolist)):
                    if np.sign(usrsigninfolist[i]) == np.sign(total_signinfo_list[i]):
                        samesum = samesum + 1
                worker_credit[user][2] = samesum
                sum_list[user] = samesum

            # update_credit
            max_index = []
            sum_list = np.asarray(sum_list)
            print("sum_list", sum_list)
            max_index = sum_list.argsort()[-1:-21:-1]
            # print("max_index", max_index)

            # write sum_credit
            for i in range(args.num_users):
                if i in max_index:
                    if iteration == 1:
                        worker_credit[i][1] = worker_credit[i][1] + 0.2
                    else:
                        worker_credit[i][1] = worker_credit[i][1] + 0.4
                    sum_credit[i].append(worker_credit[i][1])
                else:
                    sum_credit[i].append(worker_credit[i][1])
            # print("sum_credit", sum_credit)
            # print("workerCredit", worker_credit)

            # clear samesum
            for usr in range(args.num_users):
                worker_credit[usr][2] = 0

            # model aggregate
            for index, (para, signvalue) in enumerate(zip(model0.parameters(), signmaster)):
                master_grad[index].data.zero_()
                master_grad[index].data.add_(para.data, alpha=u)
                master_grad[index].data.add_(signvalue.data, alpha=lambda0)
                para.data.add_(master_grad[index].data, alpha=-args.lr)

            # calculate loss and accuracy of the testing data.
            test_loss, test_acc = caL_loss_acc(model0, device, test_loader)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_acc))
            test_acc_list.append(test_acc)
            continue
        count = 0

        #Other rounds
        model0.train()
        for id in sample_user:
            if id in byzantine:
                worker_model_flat = flatten_list(worker_model)
                worker_model_flat = attack(worker_model_flat, regular, byzantine)
                worker_model[id] = unflatten_vector(worker_model_flat[id], model0)
                for index, (para0, paraby) in enumerate(zip(model0.parameters(), worker_model[id])):
                    signinfo[id][index].data.zero_()
                    signinfo[id][index].data.add_(torch.sign(para0.data - paraby.data), alpha=1)
                continue
            model[id].train()
            if setting == 'iid':
                try:
                    batch_iterator = train_loaders_splited_iter[id]
                    data, target = next(batch_iterator)
                except StopIteration:
                    train_loaders_splited_iter = [iter(loader) for loader in train_loaders_splited]
                    batch_iterator = train_loaders_splited_iter[id]
                    data, target = next(batch_iterator)
            elif setting == 'noniid':
                try:
                    batch_iterator = train_loaders_splited_iter[count]
                    data, target = next(batch_iterator)
                except StopIteration:
                    train_loaders_splited_iter = [iter(loader) for loader in train_loaders_splited]
                    batch_iterator = train_loaders_splited_iter[count]
                    data, target = next(batch_iterator)
                count += 1

            data, target = data.to(device), target.to(device)
            output = model[id](data)
            loss = F.nll_loss(output, target)

            # autograd
            model[id].zero_grad()
            loss.backward()

            # obtain the sign information
            for index, (para, para0) in enumerate(zip(model[id].parameters(), model0.parameters())):
                signinfo[id][index].data.zero_()

                # add the Gaussian noise to the sign information
                sigma = torch.maximum(2/3*(para0.data - para.data), 4*du/epsilon * torch.ones_like(para))
                gauss = torch.randn_like(para) * sigma
                signinfo[id][index].data.add_(torch.sign(para0.data - para.data + gauss), alpha=1)

                # signinfo[id][index].data.add_(torch.sign(para0.data - para.data), alpha=1)

            for index, (para, signvalue) in enumerate(zip(model[id].parameters(), signinfo[id])):
                worker_grad[id][index].data.zero_()
                worker_grad[id][index].data.add_(para.grad.data, alpha=1)
                worker_grad[id][index].data.add_(para, alpha=args.decayWeight)
                worker_grad[id][index].data.add_(signvalue.data, alpha=-lambda0)

            for index, para in enumerate(model[id].parameters()):
                worker_model[id][index].data.zero_()
                worker_model[id][index].data.add_(para.data, alpha=1)

            for para, grad in zip(model[id].parameters(), worker_grad[id]):
                para.data.add_(grad, alpha=-args.lr)

        # based on clusters to sample
        worker_model_flat = flatten_list(worker_model)
        signinfo_flat = flatten_list(signinfo)
        sample_user = []
        clusters = []
        for i in range(kmeans.n_clusters):
            idxs = np.where(kmeans.labels_ == i)[0]
            cluster = []
            for idx in idxs:
                cre_cluster_sum[i] += worker_credit[idx][1]
                cluster.append(idx)
            clusters.append(cluster)
        max_index = list(np.argsort(cre_cluster_sum, axis=-1, kind='quicksort', order=None))[-1:-61:-1]

        for i in max_index:
            idxs = clusters[i]
            weight_user = []
            for idx in idxs:
                weight_user.append([idx, worker_credit[idx][1]])
            sum_weight = np.sum(weight_user, axis=0)
            weights = [_weight_user[1] for _weight_user in weight_user]
            probs = [weight / sum_weight[1] for weight in weights]
            choice = np.random.choice(idxs, size=1, p=probs)
            sample_user.append(choice[0])
        print('sample_user', sample_user)

        signinfo_flat = torch.sum(signinfo_flat, dim=0)
        signmaster = unflatten_vector(signinfo_flat, model0)
        flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
        sum_list = [0 for i in range(args.num_users)]
        # Reward Worker based on samesum
        for user in sample_user:
            usrsigninfolist = []
            for p in signinfo[user]:
                usrsigninfolist.append(p.tolist())
            usrsigninfolist = flatten(usrsigninfolist)
            total_signinfo_list = signinfo_flat.tolist()
            # yihuo = usrsigninfolist^total_signinfo_list
            samesum = 0
            for i in range(len(usrsigninfolist)):
                if np.sign(usrsigninfolist[i]) == np.sign(total_signinfo_list[i]):
                    samesum = samesum + 1
            worker_credit[user][2] = samesum
            sum_list[user] = samesum

        # update_credit
        max_index = []
        sum_list = np.asarray(sum_list)
        # print("sum_list", sum_list)
        max_index = sum_list.argsort()[-1:-21:-1]
        # print("max_index", max_index)

        # write sum_credit
        for i in range(args.num_users):
            if i in max_index:
                worker_credit[i][1] = worker_credit[i][1] + 0.1
                sum_credit[i].append(worker_credit[i][1])
            else:
                sum_credit[i].append(worker_credit[i][1])
        # print("sum_credit", sum_credit)
        # print("workerCredit", worker_credit)

        # clear samesum
        for usr in range(args.num_users):
            worker_credit[usr][2] = 0

        for index, (para, signvalue) in enumerate(zip(model0.parameters(), signmaster)):
            master_grad[index].data.zero_()
            master_grad[index].data.add_(para.data, alpha=u)
            master_grad[index].data.add_(signvalue.data, alpha=lambda0)
            para.data.add_(master_grad[index].data, alpha=-args.lr)

        # calculate loss and accuracy of the testing data.
        test_loss, test_acc = caL_loss_acc(model0, device, test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_acc))
        test_acc_list.append(test_acc)

    # save model
    if save_model:
        torch.save(model0.state_dict(), "RSA.pt")

    # save experiment results
    if args.attack == sign_flipping:
        file_name = 'sign_flipping/CreGau_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
        file_name_cre = 'sign_flipping/CreGau_xinyu_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
    elif args.attack == same_value:
        file_name = 'same_value/CreGau_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
        file_name_cre = 'same_value/CreGau_xinyu_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
    elif args.attack == gauss_attack:
        file_name = 'gauss_attack/CreGau_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
        file_name_cre = 'gauss_attack/CreGau_xinyu_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
    elif args.attack == zero_gradient:
        file_name = 'zero_gradient/CreGau_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
        file_name_cre = 'zero_gradient/CreGau_xinyu_corrflip_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
    elif args.attack == sample_duplicating:
        file_name = 'sample_duplicating/CreGau_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
        file_name_cre = 'sample_duplicating/CreGau_xinyu_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)

    with open(file_name, 'wb') as f:
        pickle.dump((args, train_loss_list, test_acc_list), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(file_name_cre, 'wb') as f:
        pickle.dump((byzantine, sum_credit), f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    args = args_parser()
    centralized_sgd(setting=args.iid, attack=args.attack,
                    save_model=False, save_results=True)








