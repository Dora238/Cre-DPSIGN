import torch


def same_value(worker_grad, regular, byzantine):
    """
    Same-value attacks
    """
    malicious_grad = 100 * torch.ones_like(worker_grad[0])
    for id in byzantine:
        worker_grad[id].copy_(malicious_grad)
    return worker_grad


def sign_flipping(worker_grad, regular, byzantine):
    """
    Sign-flipping attacks
    """
    for id in byzantine:
        worker_grad[id] *= -5
    return worker_grad


def gauss_attack(worker_grad, regular, byzantine):
    """
    Gaussian attacks
    """
    malicious_grad = 10000 * torch.randn_like(worker_grad[0])
    for id in byzantine:
        worker_grad[id].copy_(malicious_grad)
    return worker_grad


def zero_gradient(worker_grad, regular, byzantine):
    """
    Zero-gradient attacks
    """
    regular_sum = torch.zeros_like(worker_grad[0])
    for id in regular:
        regular_sum.add_(1, worker_grad[id])

    for id in byzantine:
        worker_grad[id].copy_(-1 * regular_sum / len(byzantine))

    return worker_grad


def norm_zero_gradient(worker_grad, regular, byzantine):
    """
    Normalized zero-gradient attacks
    """
    regular_sum = torch.zeros_like(worker_grad[0])
    for id in regular:
        regular_sum.add_(1 / torch.norm(worker_grad[id]),
                         worker_grad[id])

    for id in byzantine:
        worker_grad[id].copy_(-1 * regular_sum / len(byzantine))

    return worker_grad


def sample_duplicating(worker_grad, regular, byzantine):
    """
    Sample-duplicating attacks where Byzantine workers send the gradient of the first regular worker
    """
    malicious_grad = worker_grad[regular[0]]
    for id in byzantine:
        worker_grad[id].copy_(malicious_grad)
    return worker_grad