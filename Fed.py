import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
import time

from Config import Config
from Logger import Logger
from Preprocessing import *
from Model import *

import argparse

def train(config, model, optimizer, data, local_control_variate, global_control_variate, device):
    num_local_steps = config.num_local_steps
    criterion = config.criterion
    fed_algorithm = config.fed_algorithm
    mu = config.mu
    learning_rate = config.learning_rate
    model_type = config.model_type

    model.to(device)

    model_dict = model.state_dict()
    global_model = model_type(num_classes=10).to(device)
    global_model.load_state_dict(model_dict)

    model.train()
    losses = 0

    for step in range(num_local_steps):
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if fed_algorithm == 'FedProx':
                # print('Using FedProx')
                proximal_term = 0
                for w, w_global in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (mu / 2) * torch.norm(w - w_global)**2
                loss += proximal_term

            loss.backward()

            # SCAFFOLD adds additional control on changes to local updates
            if fed_algorithm == 'SCAFFOLD':
                # local_control_variate = [cv.to(device) for cv in local_control_variate]
                for param, local_cv, global_cv in zip(model.parameters(), local_control_variate, global_control_variate):
                    param.grad.data += local_cv - global_cv

            optimizer.step()

            losses += loss.item()
            # print(f"Rank {i}, Step {step}, Loss: {loss.item()}")

    # Update local control variate for SCAFFOLD AFTER local steps
    if fed_algorithm == 'SCAFFOLD':
        with torch.no_grad():
            for param, global_param, local_cv, global_cv in zip(model.parameters(), global_model.parameters(), local_control_variate, global_control_variate):
                local_cv.data = local_cv.data - global_cv.data + (global_param.data - param.data) / (learning_rate * num_local_steps)

    return losses

def average_models(models, fed_algorithm, local_control_variates=None):
    model_params = [model.state_dict() for model in models]
    averaged_params = {}
    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name] for i in range(len(models))])
        averaged_params[param_name] = torch.mean(params.float(), dim=0)  # Cast to float before computing mean

    global_control_variate = None
    if fed_algorithm == "SCAFFOLD":
        global_control_variate = [
            torch.mean(torch.stack([lcv[i] for lcv in local_control_variates]), dim=0)
            for i in range(len(local_control_variates[0]))
        ]
    return averaged_params, global_control_variate

def duplicate_model(config, model, num_duplicates, device):
    model_type = config.model_type

    model_dicts = [model.state_dict() for _ in range(num_duplicates)]
    duplicated_models = [model_type(num_classes=10).to(device) for _ in range(num_duplicates)]
    for i, model_dict in enumerate(model_dicts):
        duplicated_models[i].load_state_dict(model_dict)
    return duplicated_models

def communication_round(index, config, model, train_loaders, global_control_variate, all_losses, logger, device, test_loader):
    num_clients = config.num_clients
    fed_algorithm = config.fed_algorithm
    learning_rate = config.learning_rate
    optimizer = config.optimizer
    num_gpus = torch.cuda.device_count()

    global_model = model

    models = duplicate_model(config, global_model, num_clients, device)
    if optimizer == 'adam':
        optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in models]
    elif optimizer == 'sgd':
        optimizers = [optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) for model in models]
    else:
        optimizers = [None]
        logger.log('Optimizer NOT SUPPORTED')

    processes = []
    data_this_round = []
    losses = 0

    if fed_algorithm == 'SCAFFOLD':
        local_control_variates = [[gcv.clone() for gcv in global_control_variate] for i in range(num_clients)]
    else:
        local_control_variates = [None for _ in range(num_clients)]

    # prep data for this round
    # for i in range(len(models)):
    #     images, labels = next(iter(train_loaders[i]))
    #     data_this_round.append((images, labels))

    ### === TRAINING ===

    for i in range(len(models)):
        losses += train(config, models[i], optimizers[i], train_loaders[i], local_control_variates[i], global_control_variate, device)

    ### === END OF TRAINING ===

    # average models and prep models for next round
    # logger.log('    Averaging client models...')
    if fed_algorithm == 'SCAFFOLD':
        ensemble_model_params, global_control_variate = average_models(models, fed_algorithm, local_control_variates)
    else:
        ensemble_model_params, global_control_variate = average_models(models, fed_algorithm)

    global_model.load_state_dict(ensemble_model_params)

    all_losses.append(losses)
    # periodically print the loss
    if (index+1) % 10 == 0:
        logger.log(f'    *** Round {index+1} loss = {losses}')
        # for model in models:
        #     model.eval()
        #     with torch.no_grad():
        #         correct = 0
        #         total = 0
        #         for images, labels in test_loader:
        #             images, labels = images.to(device), labels.to(device)
        #             outputs = model(images)
        #             # logger.log(outputs)
        #             _, predicted = torch.max(outputs.data, 1)
        #             total += labels.size(0)
        #             correct += (predicted == labels).sum().item()
        #             break
        #         logger.log(f'Before aggregation test accuracy: {100 * correct / total}%')
        global_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                # logger.log(outputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                break
            logger.log(f'After aggregation test accuracy: {100 * correct / total}%')

    return global_model

def federated_learning(config, logger, device):
    num_communications = config.num_communications
    optimizer = config.optimizer
    fed_algorithm = config.fed_algorithm
    model_type = config.model_type

    logger.log("Loading dataset...")

    # change False to True if download needed
    train_dataset, test_dataset = load_dataset(True)

    train_loaders, test_loader = partition_dataset(config, train_dataset, test_dataset, logger)

    logger.log("Creating models, optimizers, GCVs...")
    model = model_type(num_classes=10).to(device)
    all_losses = []

    ### === SAFETY CHECKS ===
    # available optimizer check
    if optimizer not in ('adam', 'sgd'):
        logger.log('!!! Optimizer not supported !!!', level='warning')
        return None, None
    # available algorithm check
    if fed_algorithm not in ('FedAvg', 'FedProx', 'SCAFFOLD'):
        logger.log('!!! Federated learning algorithm not supported !!!', level='warning' )
        return None, None
    ### === END OF SAFETY CHECKS ===

    if fed_algorithm == 'SCAFFOLD':
        global_control_variate = [torch.zeros_like(param) for param in model.parameters()]
    else:
        global_control_variate = [None]

    for round in range(num_communications):
        # print(f"Communication Round {round+1}")
        model = communication_round(round, config, model, train_loaders, global_control_variate, all_losses, logger, device, test_loader)

    # final evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # logger.log(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            break
        logger.log(f'Test Accuracy: {100 * correct / total}%')

    return 100 * correct / total, all_losses

if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description='Federated Learning Experiment')
    parser.add_argument('--fl_algorithm', type=str, nargs='+', default=['FedAvg'], help='List of Federated learning algorithm (default: FedAvg)')
    parser.add_argument('--log', type=str, default='results.log', help='Name of log file')
    parser.add_argument('--distribution', type=str, default='non-IID', help='Type of data distribution')
    parser.add_argument('--alpha', type=float, nargs='+', default=[0, 0.05, 0.1], help='List of alpha values (default: [0.05])')
    parser.add_argument('--local_step', type=int, nargs='+', default=[1, 3, 5], help='List of local steps (default: [5])')
    parser.add_argument('--iteration', type=int, default=5, help='Number of iterations (default: 5)')
    parser.add_argument('--device_id', type=int, default=1, help='CUDA device ID (default: 1)')

    '''
    python Fed.py --fl_algorithm FedAvg --device_id 1 --log results.log
    python Fed.py --fl_algorithm FedProx --device_id 2 --log results2.log
    python Fed.py --fl_algorithm SCAFFOLD --device_id 3 --log results3.log
    '''

    args = parser.parse_args()

    logger = Logger(args.log)
    logger.initialize()

    # logger.log('#######################################')
    # logger.log('Federated Learning experimental results')
    # logger.log('#######################################')

    # logger.log('Beginning training...')

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    s = 'Training on: ' + str(device)
    logger.log(s)

    for fl_algorithm in args.fl_algorithm:
        for a in args.alpha:
            for local_step in args.local_step:
                logger.log('----------------------------------------')
                logger.log(f'Currently using: {fl_algorithm}')
                logger.log(f"Using alpha = {a}")
                logger.log(f"Local step = {local_step}")

                accuracies = []
                losses = []
                times = []

                for i in range(args.iteration):
                    logger.log("---------------------------")
                    logger.log(f"Iteration {i}:")

                    config = Config(alpha=a, fed_algorithm=fl_algorithm, num_local_steps=local_step, 
                                    distribution=args.distribution, model_type=ResNet50)

                    tic = time.perf_counter()
                    accuracy, loss = federated_learning(config, logger, device)
                    toc = time.perf_counter()

                    accuracies.append(accuracy)
                    losses.append(loss)
                    times.append(toc - tic)
                
                logger.log(f"{accuracies}")
                logger.log(f"Average accuracy: {np.average(accuracies)}")
                logger.log(f"Average loss per 10 iterations: {np.average(losses, axis=0)}")
                logger.log(f"Average time: {np.average(times)}")
