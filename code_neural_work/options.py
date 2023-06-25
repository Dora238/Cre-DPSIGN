import argparse
from Attack import same_value, sign_flipping, zero_gradient, sample_duplicating,gauss_attack
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--byzantinue_users', type=int, default=10,
                        help="number of byzantinue users: K")

    parser.add_argument('--iterations', type=int, default=5000,
                        help="interations: K")
    parser.add_argument('--decayWeight', type=float, default=0.00,
                        help="decayWeight: K")

    parser.add_argument('--batchsize', type=int, default=1,
                        help="batchsize: K")
    parser.add_argument('--testbatchsize', type=int, default=1000,
                        help="batchsize: K")

    parser.add_argument('--attack', type=str, default=gauss_attack,
                        help="attack: same_value, sign_flipping, zero_gradient, sample_duplicating")

    parser.add_argument('--iid', type=str, default='iid',
                        help="iid,noniid")
    parser.add_argument('--eps', type=float, default=0.4,
                        help="epsilon:0.2;0.5;0.8;1")
    parser.add_argument('--lr', type=float, default=0.04,
                        help="learning rate")

    parser.add_argument('--iter', type=int, default=1000,
                        help="learning rate")
    args = parser.parse_args()
    return args

def exp_details(args):
    print('\nExperimental details:')
    print(f'    attack     : {args.attack}')
    print(f'    iid : {args.iid}')
    print(f'    Learning  : {args.lr}')
    # print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Num of users  : {args.num_users}')
    print(f'    Num of byzantinue users   : {args.byzantinue_users}')
    print(f'    epsilon      : {args.eps}\n')
    return