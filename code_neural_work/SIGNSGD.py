import os
import pickle
import random
import torch
import torch.nn.functional as F
from options import args_parser, exp_details
from LoadData import getData_mnist
from MainModel import MLP, flatten_list, unflatten_vector, caL_loss_acc, mean, gm
from Attack import same_value, sign_flipping, zero_gradient, sample_duplicating,gauss_attack

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("code_neural_network")+len("code_neural_network")]


def centralized_sgd(setting, aggregate, attack=None, save_model=False, save_results=False):
    """

    :param setting: 'iid' or 'noniid'
    :param aggregate: mean or gm(geometric median)
    :param attack: Byzantine attacks, like sample-duplicating attacks
    :param resampling_flag: True for RS-Byrd-SGD, False for Byrd-SGD
    :param save_model: whether to save the global model
    :param save_results: whether to save the experiment results
    """
    # initialize the global model and data loader
    args = args_parser()
    exp_details(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(784, 50, 50, 10).to(device)
    train_dataset_subset, train_loader, test_loader = getData_mnist(setting, args)
    train_loaders_splited = [
        torch.utils.data.DataLoader(dataset=subset, batch_size=args.batchsize, shuffle=False, pin_memory=True)
        for subset in train_dataset_subset
    ]
    train_loaders_splited_iter = [iter(loader) for loader in train_loaders_splited]

    train_loss_list = []
    test_acc_list = []
    byzantine = []
    regular = []

    # initialize the set of gradients
    worker_grad = [
        [torch.zeros_like(para, requires_grad=False) for para in model.parameters()]
        for _ in range(args.num_users)
    ]

    # start training
    for iteration in range(1, args.iterations + 1):
        print('Train iteration: {}'.format(iteration))

        # randomly generate the set of Byzantine workers and regular workers.
        if iteration == 1:
            byzantine = random.sample(range (args.num_users), args.byzantinue_users)
            regular = list(set(range(args.num_users)).difference (byzantine))
            print(byzantine)
            print(regular)

        count = 0

        # regular workers compute the corrected stochastic gradients using SAGA
        for id in regular:

            model.train()
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
            output = model(data)
            loss = F.nll_loss(output, target)

            # autograd
            model.zero_grad()
            loss.backward()

            # store the workers' gradients
            for index, para in enumerate(model.parameters()):
                worker_grad[id][index].data.zero_()
                worker_grad[id][index].data.add_(1, para.grad.data)
                worker_grad[id][index].data.add_(args.decayWeight, para)

        # the master node aggregate the stochastic gradients under Byzantine attacks
        worker_grad_flat = flatten_list(worker_grad)

        if attack != None:
            worker_grad_flat = attack(worker_grad_flat, regular, byzantine)


        sign_worker_grad = torch.sign(worker_grad_flat)

        aggrGrad_flat = aggregate(sign_worker_grad)
        aggrGrad = unflatten_vector(aggrGrad_flat, model)

        # the master node update the global model
        for para, grad in zip(model.parameters(), aggrGrad):
            para.data.add_(-args.lr, grad)

        # calculate loss and accuracy of the testing data.
        test_loss, test_acc = caL_loss_acc(model, device, test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_acc))
        test_acc_list.append(test_acc)

    # save model
    if save_model:
        torch.save(model.state_dict(), "sgd.pt")

    # save experiment results
    if args.attack == sign_flipping:
        file_name = 'sign_flipping/SIGNSGD_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
    elif args.attack == same_value:
        file_name = 'same_value/SIGNSGD_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
    elif args.attack == gauss_attack:
        file_name = 'gauss_attack/SIGNSGD_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
    elif args.attack == zero_gradient:
        file_name = 'zero_gradient/SIGNSGD_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)
    elif args.attack == sample_duplicating:
        file_name = 'sample_duplicating/SIGNSGD_user[{}]_byzanusers[{}]_duli{}_lr[{}]_eps[{}].pkl'. \
            format(args.num_users, args.byzantinue_users, args.iid, args.lr,
                   args.eps)

    with open(file_name, 'wb') as f:
        pickle.dump((args, train_loss_list, test_acc_list), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(file_name, 'wb') as f:
        pickle.dump((args, train_loss_list, test_acc_list), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = args_parser()
    centralized_sgd(setting='iid', aggregate=mean, attack=args.attack,
                    save_model=False, save_results=True)








