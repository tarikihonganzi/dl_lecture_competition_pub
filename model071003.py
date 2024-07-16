import re
import random
import time
from statistics import mode

from PIL import Image, ImageStat
from tqdm import tqdm
import numpy as np
import pandas
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import torch.nn.functional as F

from torchvision.models import resnet34, ResNet34_Weights, Wide_ResNet50_2_Weights, wide_resnet50_2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text_adv(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't", "whats": "what's",
        "thats": "that's", "whos": "who's", "wheres": "where's", "whens": "when's",
        "please":"", "could you": "can you", "could i": "can i", "could we": "can we",
    }

    contractions_2 = {
        "theatre" : "theater", "colour" : "color", "centre" : "center", "favourite" : "favorite",
        "travelling" : "traveling", "counselling" : "counseling", "metre" : "meter",
        "cancelled" : "canceled", "labour" : "labor", "organisation" : "organization",
        "calibre" : "caliber", "cheque" : "check", "manoeuvre" : "maneuver",
        "neighbour" : "neighbor", "grey" : "gray", "dialogue" : "dialog",
    }

    contractions_3 = {
        "what is": "what's", "who is": "who's", "where is": "where's", "when is": "when's",
        "how is": "how's", "it is": "it's", "he is": "he's", "she is": "she's",
        "that is": "that's", "there is": "there's", "here is": "here's",
        "i am": "i'm", "you are": "you're", "we are": "we're", "they are": "they're",
        "i have": "i've", "you have": "you've", "we have": "we've", "they have": "they've",
        "i will": "i'll", "you will": "you'll",
    }
    # contractions_3 = {
    #     "what's" : "what is", "who's" : "who is", "where's" : "where is", "when's" : "when is",
    #     "how's" : "how is", "it's" : "it is", "he's" : "he is", "she's" : "she is",
    #     "that's" : "that is", "there's" : "there is", "here's" : "here is",
    #     "i'm" : "i am", "you're" : "you are", "we're" : "we are", "they're" : "they are",
    #     "i've" : "i have", "you've" : "you have", "we've" : "we have", "they've" : "they have",
    #     "i'll" : "i will", "you'll" : "you will",
    # }
 
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    for contraction, correct in contractions_2.items():
        text = text.replace(contraction, correct)
    for contraction, correct in contractions_3.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, transform_adv=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.transform_adv = transform_adv  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        print(f"df shape: {self.df.shape}")
        
        self.answer = answer

        # question / answerの辞書を作成
        self.question2idx = {}
        self.answer2idx = {}
        self.idx2question = {}
        self.idx2answer = {}
        c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0

        # 質問文の最大長を取得
        self.max_question_length = 0

        # 質問文に含まれる単語を辞書に追加
        for question in self.df["question"]:
            question = process_text_adv(question)
            words = question.split(" ")

            if len(words) > self.max_question_length:
                self.max_question_length = len(words)
            
            for idex, word in enumerate(words):            
                if word not in self.question2idx:
                    self.question2idx[word] = len(self.question2idx)

        
        self.idx2question = {v: k for k, v in self.question2idx.items()}  # 逆変換用の辞書(question)

        self.answer2idx["<unk>"] = 0

        if self.answer:
            # 回答に含まれる単語を辞書に追加
            for answers in self.df["answers"]:
                for answer in answers:
                    word = answer["answer"]
                    word = process_text(word)
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}  # 逆変換用の辞書(answer)

            # print(f"answer{self.idx2answer}")

    def update_dict(self, dataset):
        """
        検証用データ，テストデータの辞書を訓練データの辞書に更新する．

        Parameters
        ----------
        dataset : Dataset
            訓練データのDataset
        """
        self.question2idx = dataset.question2idx
        self.answer2idx = dataset.answer2idx
        self.idx2question = dataset.idx2question
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx, isKL=False):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (vocab_size)
            質問文をone-hot表現に変換したもの
        answers : torch.Tensor  (n_answer)
            10人の回答者の回答のid
        mode_answer_idx : torch.Tensor  (1)
            10人の回答者の回答の中で最頻値の回答のid
        """
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}").convert("RGB")
        
        # image = self.transform(image)
        question = np.zeros(len(self.idx2question) + 1)  # 未知語用の要素を追加
        question_vector = []
        # question_words = self.df["question"][idx].split(" ")
        question_text = process_text_adv(self.df["question"][idx])
        question_words = question_text.split(" ")

        is_positional_question = False

        for i in range(self.max_question_length - len(question_words)):
            question_vector.insert(0, 0)
    
        for word in question_words:
            if not is_positional_question and (word == "where" or word == "right" or word == "left" or word == "top" or word == "bottom" or word == "position"):
                is_positional_question = True  
            try:
                question[self.question2idx[word]] = 1  # one-hot表現に変換
                question_vector.append(self.question2idx[word])
            except KeyError:
                question[-1] = 1  # 未知語
                question_vector.append(0)
        
        # if is_positional_question:
        #     print(f"positional question: {question_words}")
        # else:
        #     print(f"NOT positional question: {question_words}")

        if self.answer:
            answers_prevector = np.zeros(len(self.answer2idx))
                
            answers = [self.answer2idx[process_text(answer["answer"])]  if process_text(answer["answer"]) in self.answer2idx 
                       else 0 for answer in self.df["answers"][idx]]
            for answer in answers:
                answers_prevector[answer] += 1
            sum_answers = max(np.sum(answers_prevector), 1e-10)
            answers_vector = answers_prevector / sum_answers
            mode_answer_idx = mode(answers)  # 最頻値を取得（正解ラベル）

            if is_positional_question or self.transform_adv is None:
                image = self.transform(image)
            else:
                image = self.transform_adv(image)

            return image, torch.Tensor(question), torch.Tensor(answers), int(mode_answer_idx), torch.Tensor(answers_vector), torch.LongTensor(question_vector)

        else:

            if is_positional_question or self.transform_adv is None:
                image = self.transform(image)
            else:
                image = self.transform_adv(image)

            return image, torch.Tensor(question), torch.LongTensor(question_vector)

    def __len__(self):
        return len(self.df)

def loadVQAData(df_path, image_dir, transform=None, answer=True):
    return VQADataset(df_path, image_dir, transform, answer)

# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)

# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

class VQAModel(nn.Module):
    def __init__(self, vocab_size: int, n_answer: int):
        super().__init__()
        # self.resnet = ResNet18()
        self.resnet = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        # self.resnet = torchvision.models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(512, 512)
        self.text_encoder = nn.Linear(vocab_size, 512)

        self.word_embeddings = nn.Embedding(vocab_size, 512, padding_idx=0)
        # batch_first=Trueが大事！
        self.lstm = nn.LSTM(512, 512, batch_first=True, dropout=0.20)

        self.hidden2tag = nn.Linear(512, 512)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_answer)

        self.dropout = nn.Dropout(0.15)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量
        # question_feature = self.text_encoder(question)  # テキストの特徴量

        embeds = self.word_embeddings(question)
        #embeds.size() = (batch_size × len(sentence) × embedding_dim)
        _, lstm_out = self.lstm(embeds)
        # x = torch.cat([image_feature, lstm_out[0].squeeze()], dim=1)

        tag_space = self.hidden2tag(lstm_out[0])
        question_feature = F.leaky_relu(torch.squeeze((tag_space + lstm_out[0]), dim=0))

        # x = torch.cat([image_feature, tag_space.squeeze()], dim=1)
        x = torch.cat([image_feature, question_feature], dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        y = self.fc2(x)
        x = F.leaky_relu((x + y))
        x = self.fc3(x)

        return x

def softmax(x, axis=1):
    x -= x.max(axis, keepdims=True) # expのoverflowを防ぐ
    x_exp = np.exp(x)
    return x_exp / x_exp.sum(axis, keepdims=True)

# 4. 学習の実装
def train_KL(model, dataloader, optimizer, criterion, device):
    model.train()
    eps = 1e-8

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer, answers_vector , question_vector in tqdm(dataloader):
        image, question, answer, mode_answer, answers_vector, question_vector = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device), answers_vector.to(device), question_vector.to(device)

        pred = model(image, question_vector)
        nn_softmax = nn.Softmax(dim=1)
        softmax_pred = nn_softmax(pred)

        loss = criterion(torch.log(softmax_pred + eps), answers_vector + eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

# 4. 学習の実装_normal
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer, answers_vector in tqdm(dataloader):
        image, question, answer, mode_answer, answers_vector = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device), answers_vector.to(device)

        pred = model(image, question)
        loss = criterion(pred, mode_answer.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

def eval(model, dataloader, optimizer, criterion, device):
    model.eval()
    eps = 1e-8

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mode_answer, answers_vector , question_vector in tqdm(dataloader):
        image, question, answer, mode_answer, answers_vector, question_vector = \
            image.to(device), question.to(device), answers.to(device), mode_answer.to(device), answers_vector.to(device), question_vector.to(device)
        

        pred = model(image, question_vector)
        nn_softmax = nn.Softmax(dim=1)
        softmax_pred = nn_softmax(pred)

        loss = criterion(torch.log(softmax_pred + eps), answers_vector + eps)

        total_loss += loss.item()
        total_acc += VQA_criterion(pred.argmax(1), answers)  # VQA accuracy
        simple_acc += (pred.argmax(1) == mode_answer).float().mean().item()  # simple accuracy

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start

set_seed(42)

# dataloader / model
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_train_adv = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform_train, transform_adv=transform_train_adv, answer=True)

valid_dataset = VQADataset(df_path="./data/test.json", image_dir="./data/train", transform=transform_test, transform_adv=transform_test, answer=True)
valid_dataset.update_dict(train_dataset)

test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform_test, answer=False)
test_dataset.update_dict(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# cosine scheduler
class CosineScheduler:
    def __init__(self, epochs, lr, warmup_length=5):
        """
        Arguments
        ---------
        epochs : int
            学習のエポック数．
        lr : float
            学習率．
        warmup_length : int
            warmupを適用するエポック数．
        """
        self.epochs = epochs
        self.lr = lr
        self.warmup = warmup_length

    def __call__(self, epoch):
        """
        Arguments
        ---------
        epoch : int
            現在のエポック数．
        """
        progress = (epoch - self.warmup) / (self.epochs - self.warmup)
        progress = np.clip(progress, 0.0, 1.0)
        lr = self.lr * 0.5 * (1. + np.cos(np.pi * progress))

        if self.warmup:
            lr = lr * min(1., (epoch+1) / self.warmup)

        return lr

def set_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
model = VQAModel(vocab_size=len(train_dataset.question2idx)+1, n_answer=len(train_dataset.answer2idx)).to(device)

# optimizer / criterion
num_epoch = 20
warmup_length = 4
lr = 0.001
criterion = nn.CrossEntropyLoss()
criterion_KL = nn.KLDivLoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

scheduler = CosineScheduler(num_epoch, lr, warmup_length)

train_loss_hist = []
train_acc_hist = []
train_lr_hist = []

test_loss_hist = []
test_acc_hist = []

best_test_acc = 0.
best_epoch = 0

# train model
for epoch in range(num_epoch):
    new_lr = scheduler(epoch)
    set_lr(new_lr, optimizer)
    train_lr_hist.append(new_lr)
    
    # train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
    train_loss, train_acc, train_simple_acc, train_time = train_KL(model, train_loader, optimizer, criterion_KL, device)
    test_loss, test_acc, test_simple_acc, test_time = eval(model, valid_loader, optimizer, criterion_KL, device)
    print(f"【{epoch + 1}/{num_epoch}】\n"
            f"train time: {train_time:.2f} [s]\n"
            f"train loss: {train_loss:.4f}\n"
            f"train acc: {train_acc:.4f}\n"
            f"train simple acc: {train_simple_acc:.4f}\n"
            f"train lr: {new_lr:.6f}\n")
    print(f"test time: {test_time:.2f} [s]\n"
          f"test loss: {test_loss:.4f}\n"
          f"test acc: {test_acc:.4f}\n"
          f"test simple acc: {test_simple_acc:.4f}")
    
    if test_acc > best_test_acc:
      best_test_acc = test_acc
      best_epoch = epoch
    torch.save(model.state_dict(), f"./model_data/071003/model_{epoch}.pth")
    
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)

    test_loss_hist.append(test_loss)
    test_acc_hist.append(test_acc)

# modelの読み込み
model.load_state_dict(torch.load(f"./model_data/071003/model_{best_epoch}.pth"))
print(f"best epoch: {best_epoch + 1}")

# 提出用ファイルの作成
model.eval()
submission = []
for image, question, question_vector in tqdm(test_loader):
    image, question, question_vector = image.to(device), question.to(device), question_vector.to(device)
    pred = model(image, question_vector)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)

submission = [train_dataset.idx2answer[id] for id in submission]
submission = np.array(submission)
torch.save(model.state_dict(), "model_071003_best.pth")
np.save("submission_071003_best.npy", submission)

# modelの読み込み
model.load_state_dict(torch.load(f"./model_data/071003/model_{num_epoch - 1}.pth"))

# 提出用ファイルの作成
model.eval()
submission = []
for image, question, question_vector in tqdm(test_loader):
    image, question, question_vector = image.to(device), question.to(device), question_vector.to(device)
    pred = model(image, question_vector)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)

submission = [train_dataset.idx2answer[id] for id in submission]
submission = np.array(submission)
torch.save(model.state_dict(), "model_071003_normal.pth")
np.save("submission_071003_normal.npy", submission)

print("finish!")
print(f"train loss: {train_loss_hist}")
print(f"train acc: {train_acc_hist}")
print(f"test loss: {test_loss_hist}")
print(f"test acc: {test_acc_hist}")