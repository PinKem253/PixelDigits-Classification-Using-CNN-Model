from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import torch
import matplotlib.pyplot as plt
#%matplotlib inline
device = "cuda" if torch.cuda.is_available() else "cpu"
def cm_plot(test_dataloader:torch.utils.data.DataLoader,
            model: torch.nn.Module(),
            ):
  y_true,y_pred = [],[]
  model.eval()
  with torch.inference_mode():
    for batch,(X,y) in enumerate(test_dataloader):
      X,y = X.to(device), y.to(device)

      pred_logits = model(X)
      pred_label = pred_logits.argmax(dim=1)

      y_true.append(y.cpu())
      y_pred.append(pred_label.cpu())

  y_true = torch.cat(y_true).numpy()
  y_pred = torch.cat(y_pred).numpy()

  cm= confusion_matrix(y_true,y_pred)
  ConfusionMatrixDisplay(cm).plot()
plt.show()

