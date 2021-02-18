import random
import torch.nn as nn


def calculate_kernel(input, output, padding=0, stride=1):
    return -output * stride + 2 * padding + stride + input


def calculate_padding(input, output, kernel, stride=1):
    return 0.5 * (kernel + (output - 1) * stride - input)


def calculate_stride(input, output, kernel, padding=0):
    return (-kernel + 2 * padding + input) / (output - 1)


def calculate_output(input, kernel, padding=0, stride=1):
    return (input + 2 * padding - kernel)/stride + 1


activation_functions = [nn.Tanh, nn.Sigmoid, nn.ELU, nn.PReLU, nn.SELU, nn.LogSigmoid, nn.CELU, nn.Softsign,
                        nn.Tanhshrink]


def get_random_activation_function():
    func = random.choice(activation_functions)
    return func()
