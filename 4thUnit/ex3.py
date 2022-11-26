def create_train_test_splits(data: list, train_size: float) -> tuple:
    trained = (list(), list())
    if len(data) == 0:
        return trained
    train_size = round(int(str(train_size).split(".")[1][0]) * 0.1, 1)
    trained = (list(data[i] for i in range(0, int(len(data)*train_size))) , list(data[i] for i in range(int(len(data)*(train_size-1)), 0)))
    return trained


