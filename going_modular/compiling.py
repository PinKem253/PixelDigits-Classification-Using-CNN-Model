from tqdm.auto import tqdm
import torch
from torch import nn
import os


def install(package):
  os.system(f"pip install {package}")

try:
  import torchmetrics
  from torchmetrics import Accuracy
  from torchmetrics import Precision
  from torchmetrics import Recall
  from torchmetrics import F1Score
except:
  install("torchmetrics")
  import torchmetrics
  from torchmetrics import Accuracy
  from torchmetrics import Precision
  from torchmetrics import Recall
  from torchmetrics import F1Score

try:
  from early_stopping_pytorch import EarlyStopping
except:
  install("early_stopping_pytorch")
  from early_stopping_pytorch import EarlyStopping

def accuracy(y_pred,y_true,num_classes:int,device:torch.device):
  acc = Accuracy(task="multiclass",num_classes = num_classes).to(device)
  return acc(y_pred,y_true)
def precision(y_pred,y_true,num_classes:int,device:torch.device):
  precision = Precision(task="multiclass",average="micro",num_classes = num_classes).to(device)
  return precision(y_pred,y_true)
def recall(y_pred,y_true,num_classes:int,device:torch.device):
  recall = Recall(task="multiclass",average="micro",num_classes= num_classes).to(device)
  return recall(y_pred,y_true)
def f1(y_pred,y_true,num_classes:int,device:torch.device):
  f1 = F1Score(task="multiclass",average="micro",num_classes=num_classes).to(device)
  return f1(y_pred,y_true)

def train_step(dataloader:torch.utils.data.DataLoader,
               model:torch.nn.Module,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               num_classes:int,
               device:torch.device
               ):
  model.train()
  train_loss,train_acc = 0,0

  for batch,(X,y) in enumerate(dataloader):
    X,y = X.to(device),y.to(device)
    y_pred = model(X)

    loss = loss_fn(y_pred,y)
    train_loss += loss.item()

    y_pred_label = y_pred.argmax(dim=1)
    acc = accuracy(y_pred_label,y,num_classes,device)
    train_acc += acc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss/=len(dataloader)
  train_acc /=len(dataloader)

  return train_acc,train_loss

def test_step(dataloader:torch.utils.data.DataLoader,
              model:torch.nn.Module,
              loss_fn: torch.nn.Module,
              optimizer:torch.optim.Optimizer,
              num_classes:int,
              device:torch.device,
              ):
  model.eval()
  test_loss,test_acc,precision_score,recall_score,f1_score = 0,0,0,0,0

  with torch.inference_mode():
    for batch,(X,y) in enumerate(dataloader):
      X,y = X.to(device),y.to(device)
      y_pred = model(X)

      loss = loss_fn(y_pred,y)
      test_loss+= loss

      y_pred_label = y_pred.argmax(dim=1)
      acc = accuracy(y_pred_label,y,num_classes,device)
      test_acc += acc

      pre = precision(y_pred_label,y,num_classes,device)
      precision_score+= pre.item()

      re = recall(y_pred_label,y,num_classes,device)
      recall_score+= re.item()

      f1score = f1(y_pred_label,y,num_classes,device)
      f1_score+= f1score.item()

    test_loss/=len(dataloader)
    test_acc /=len(dataloader)
    precision_score/=len(dataloader)
    recall_score /= len(dataloader)
    f1_score /= len(dataloader)

  return test_acc,test_loss,precision_score,recall_score,f1_score


def compile(model:torch.nn.Module,
            train_dataloader:torch.utils.data.DataLoader,
            test_dataloader:torch.utils,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            device:torch.device,
            num_classes:int,
            epochs:int,
            ):

  history={
      "epochs":[],
      "train_acc":[],
      "train_loss":[],
      "test_acc":[],
      "test_loss":[],
      "precision":[],
      "recall":[],
      "f1":[],

  }
  early_stopping = EarlyStopping(patience=5,delta=1e-3,verbose=True)

  for epoch in tqdm(range(epochs)):
    train_acc,train_loss = train_step(dataloader=train_dataloader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      num_classes=num_classes,
                                      device=device,
                                      )

    test_acc,test_loss,precision,recall,f1 = test_step(dataloader=test_dataloader,
                                   model=model,
                                   loss_fn = loss_fn,
                                   optimizer=optimizer,
                                   num_classes=num_classes,
                                   device=device,
                                   )

    print('-'*100)
    print(f"Epochs: {epoch+1}")
    print(f"Train acc: {train_acc:.3f} | Train loss: {train_loss:.3f} | Test acc: {test_acc:.3f} | Test loss: {test_loss:.3f}")

    history["epochs"].append(epoch)
    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)
    history["test_acc"].append(test_acc)
    history["test_loss"].append(test_loss)
    history["precision"].append(precision)
    history["recall"].append(recall)
    history["f1"].append(f1)


    early_stopping(test_loss.item(),model)
    if early_stopping.early_stop:
      print('-'*100)
      print(f"Stopping at epoch: {epoch}")
      break

  print("Saving best model state")
  model.load_state_dict(torch.load('checkpoint.pt',weights_only=True))


  return history
