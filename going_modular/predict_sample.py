import torch
import matplotlib.pyplot as plt
import torchvision
def predict_and_plot(nrows:int,
                     ncols:int,
                     dataset: torchvision.datasets,
                     class_names:list,
                     model:torch.nn.Module,
                     device= "cuda" if torch.cuda.is_available() else "cpu",
                     ):
  model_0.eval()
  #width,tall
  plt.figure(figsize=(nrows,ncols))

  for i in range(1,nrows*ncols+1):
    idx = torch.randint(0,len(test_dataset),size=[1]).item()
    img,label = dataset[idx]
    img = img.to(device)

    y_pred = model(img.unsqueeze(dim=0))
    y_pred_label = y_pred.argmax(dim=1)

    plt.subplot(nrows,ncols,i)
    plt.imshow(img.cpu().squeeze())

    if(y_pred_label == label):
      plt.title(class_names[label],c="green")
    else:
      plt.title(f"Predict:{class_names[y_pred_label]} \n True:{class_names[label]}", c="red")

    plt.axis(False)
    plt.tight_layout()
plt.show()
