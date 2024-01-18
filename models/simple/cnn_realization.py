from torch.utils.data import DataLoader, Subset, random_split
from EmbryoVision.preprocessing.dataset_creation import EmbryoImageDataset
from torch.nn.functional import one_hot as hot
from torchvision.transforms import v2
from cnn_model import EmbryoModel
from EmbryoVision.utils import EarlyStopping
import EmbryoVision.models.attention_based.attention_network as att_net
from EmbryoVision.utils.tools import Tools
from loss import CombinedLoss

import torch


class Trainer:
    def __init__(self):
        self.dataset = torch.load("C:\\Work\\EmbryoVision\\data\\torch_type\\first_dataset")
        self.tools = Tools()
        self.num_epochs = 10
        self.indices = torch.randperm(len(self.dataset)).tolist()
        self.num_batches = 16
        self.shuffled_dataset = Subset(self.dataset, self.indices)
        self.train_size = int(0.8 * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cnn_model = EmbryoModel().to(self.device)
        self.att_model = att_net.EmbryoModelWithAttention().to(self.device)
        self.train, self.test = self.preparing_to_train()
        self.optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=0.008)
        self.loss = CombinedLoss(lam=0.5)
        self.CEL = torch.nn.CrossEntropyLoss()
        self.MSE = torch.nn.MSELoss()
        self.BCE = torch.nn.BCELoss()

    def preparing_to_train(self):
        train, test = random_split(self.shuffled_dataset, [self.train_size, self.val_size])
        return train, test

    def train_loop(self):
        dataloader = DataLoader(self.train, shuffle=True, batch_size=16)
        train_loss = 0
        count = 0

        for images, labels in dataloader:
            # Compute prediction and loss
            reg_pred, cls_pred, hole_pred = self.cnn_model(images.to(self.device))
            MSE = self.MSE(reg_pred, labels[0].to(self.device))
            CEL = self.CEL(cls_pred, labels[1].to(self.device))
            if hole_pred.size(dim=0) != 16:
                break
            BCE = self.BCE(hole_pred.view(16), labels[2].to(self.device))

            #print("Losses are loaded.")

            self.optimizer.zero_grad()
            MSE.backward(retain_graph=True)
            #print("MSE counted")
            BCE.backward(retain_graph=True)
            #print("BCE counted")
            CEL.backward()
            #print("CEL counted")
            self.optimizer.step()

            cls_loss = self.tools.compute_preds(cls_pred, labels)
            #print("MSE Current: " + str(float(MSE)))
            #print("CEL Current: " + str(float(cls_loss)))
            #print("BCE Current: " + str(float(BCE)))
            train_loss += cls_loss
            count += 1
            #print("Overall Train Loss: " + str(train_loss / count))

        print(f"Train loss: {train_loss / count:>8f}")
        #torch.save(self.cnn_model.state_dict(), "C:\\Work\\EmbryoVision\\data\\torch_type\\att_model")
        return train_loss / count

    def test_loop(self):
        test = DataLoader(self.test, batch_size=self.num_batches, shuffle=True)
        test_loss, correct = 0, 0

        with torch.no_grad():
            count = 0
            for images, labels in test:
                # Compute prediction and loss
                reg_pred, cls_pred, hole_pred = self.cnn_model(images.to(self.device))
                MSE = self.MSE(reg_pred, labels[0].to(self.device))
                CEL = self.CEL(cls_pred, labels[1].to(self.device))
                if hole_pred.size(dim=0) != 16:
                    break
                BCE = self.BCE(hole_pred.view(16), labels[2].to(self.device))

                cls_loss = self.tools.compute_preds(cls_pred, labels)
                #print("MSE Current: " + str(float(MSE)))
                #print("CEL Current: " + str(float(cls_loss)))
                #print("BCE Current: " + str(float(BCE)))
                test_loss += cls_loss
                count += 1
                print("Overall Test Loss: " + str(test_loss / count))




        print(f"Test loss: {test_loss:>8f}, test accuracy: {(100 * test_loss):>0.1f}% \n")

        return test_loss / count

    def start_training(self):
        train_error_list = []
        test_error_list = []
        count = 0
        for i in range(self.num_epochs):
            train_error = self.train_loop()
            train_error_list.append(train_error)
            print(f"The train error for {i} epoch' is :" + str(train_error))
            test_error = self.test_loop()
            test_error_list.append(test_error)
            print(f"The test error for {i} epoch' is :" + str(test_error))

            if test_error > test_error_list[-1]: count += 1
            if count >= 3: break

        torch.save(train_error_list, "C:\\Work\\EmbryoVision\\data\\torch_type\\train_list")
        torch.save(test_error_list, "C:\\Work\\EmbryoVision\\data\\torch_type\\test_list")
        torch.save(self.cnn_model.state_dict(), "C:\\Work\\EmbryoVision\\data\\torch_type\\final_cnn_model")

    @staticmethod
    def main():
        start = Trainer()
        start.start_training()


if __name__ == "__main__":
    Trainer.main()
