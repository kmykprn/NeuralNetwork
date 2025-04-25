import os
import torch
from torch import nn
from models.simpleNN import SimpleCNN
from core.train import train
from core.test import test
from utils.dataloader import loadFashionMNIST


if __name__ == "__main__":

    # gpuかcpuを使用。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # データセットを取得
    train_dataloader, test_dataloader = loadFashionMNIST()

    # NNモデルを別ファイルから呼び出し
    model = SimpleCNN(input_shape=(1, 28, 28), num_output_class=10).to(device)

    # 重みが存在している場合は、重みをロード
    weights_path = 'weights/simple_cnn_weights.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))

    # 学習率, バッチサイズ, 学習回数を定義
    learning_rate = 1e-3
    batch_size = 64
    epochs = 50

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