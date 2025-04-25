from torch import nn


class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()

        # 1次元に変換
        self.flatten = nn.Flatten()

        # 入力: 712次元, 出力: 10次元で処理
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        NNの順方向の処理を行なう関数。
        モデルを呼び出したときに自動で呼び出されるため、直接呼び出してはいけない。

        Args:
            x:
                (バッチ数, 28, 28)

        Returns:
            logits:
                (バッチ数, 10)
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
