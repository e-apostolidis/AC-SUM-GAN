from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    config = get_config(mode='train')
    test_config = get_config(mode='test')

    print(config)
    print(test_config)

    train_loader = get_loader(config.mode, config.split_index, config.action_state_size)
    test_loader = get_loader(test_config.mode, test_config.split_index, test_config.action_state_size)
    solver = Solver(config, train_loader, test_loader)

    solver.build()
    solver.evaluate(-1)  # evaluates the summaries generated using the initial random weights of the network
    solver.train()
