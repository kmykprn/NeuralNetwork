import os
import torch
from torch import nn
from models.simple_fc import SimpleFC
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from core.train import train
from core.test import test


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


if __name__ == "__main__":

    train_dataloader, test_dataloader = loadFashionMNIST()

    # gpuかcpuを使用。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # NNモデルを別ファイルから呼び出し
    model = SimpleFC().to(device)

    # 重みが存在している場合は、重みをロード
    weights_path = 'weights/model_weights.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))

    # 学習率, バッチサイズ, 学習回数を定義
    learning_rate = 1e-3
    batch_size = 64
    epochs = 10

    # ロス関数の定義
    loss_fn = nn.CrossEntropyLoss()

    # optimizer(SGD: θ-lr * Δθ)の定義
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 学習・評価
    for epoc in range(epochs):
        print(f"Epoch {epoc+1}\n-------------------------------")
        train(device=device, dataloader=train_dataloader, model=model, loss_func=loss_fn, optimizer=optimizer)
        test(device=device, dataloader=test_dataloader, model=model, loss_func=loss_fn)

    # モデルを保存
    torch.save(model.state_dict(), weights_path)
    print("Done!")

