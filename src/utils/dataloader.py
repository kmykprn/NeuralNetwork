import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def loadFashionMNIST():
    """
    FashionMNISTのデータをロードする関数

    Args: None

    Returns:
        train_dataloader: バッチサイズ64でミニバッチを作成する学習用DataLoader
        test_dataloader: バッチサイズ64でミニバッチを作成するテスト用DataLoader
    """

    # 学習用データセットをダウンロードし"data"ディレクトリに格納
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),    # ndarrayをFloatTensorに変換し、ピクセルの値を0~1の範囲に変換。
    )

    # テスト用データセットをダウンロードし"data"ディレクトリに格納
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # epochごとにデータをシャッフルしてミニバッチを作成
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader