from torch.utils.data import DataLoader, Subset, random_split
from EmbryoVision.preprocessing.dataset_creation import EmbryoImageDataset
from torchvision.transforms import v2
from cnn_model import EmbryoModel
from loss import CombinedLoss
import torch


class Trainer:
    def __init__(self):
        self.dataset = torch.load("C:\\Work\\EmbryoVision\\data\\torch_type\\first_dataset")
        self.indices = torch.randperm(len(self.dataset)).tolist()
        self.num_batches = 16
        self.shuffled_dataset = Subset(self.dataset, self.indices)
        self.train_size = int(0.8 * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EmbryoModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.008)
        self.loss = CombinedLoss(lam=0.5)
        self.CEL = torch.nn.CrossEntropyLoss()
        self.MSE = torch.nn.MSELoss()

    def preparing_to_train(self):
        train, val = random_split(self.shuffled_dataset, [self.train_size, self.val_size])
        return train, val

    def train_loop(self):
        train, val = self.preparing_to_train()
        dataloader = DataLoader(train, shuffle=True, batch_size=16)
        train_loss = 0

        for images, labels in dataloader:
            # Compute prediction and loss
            reg_pred, cls_pred = self.model(images.to(self.device))
            MSE = self.MSE(reg_pred, labels[0].to(self.device))
            CEL = self.CEL(cls_pred, labels[1].to(self.device))
            x = torch.argmax(cls_pred, dim=2)
            print("Losses are loaded.")

            self.optimizer.zero_grad()
            MSE.backward(retain_graph=True)
            print("MSE counted")
            CEL.backward(retain_graph=True)
            print("CEL counted")
            self.optimizer.step()

            print("MSE Current: " + str(float(MSE)))
            print("CEL Current: " + str(float(CEL)))
            train_loss += (MSE + CEL)
            print(train_loss)

        train_loss /= self.num_batches
        print(f"Train loss: {train_loss:>8f}")

        return train_loss

    @staticmethod
    def main():
        start = Trainer()
        start.train_loop()


if __name__ == "__main__":
    Trainer.main()
