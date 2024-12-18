import os
try:
  import gradio as gr
except:
  os.system("pip install gradio")
  import gradio as gr
import torch
from model import create_model
#from demos.my_folder.model import create_model
from timeit import default_timer as timer

with open("class_names.txt","r") as f:
  class_names = [string.strip() for string in f.readlines()]

model,deploy_transform = create_model(device="cpu")

model.load_state_dict(
    torch.load(f="models/cnn_model.pth",
               map_location=torch.device("cpu"),
               )
)

def predict(img):
  start_time = timer()
  img = deploy_transform(img).unsqueeze(dim=0)

  model.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(model(img),dim=1)
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  end_time = timer()
  pred_time = round(end_time-start_time,3)
  return pred_labels_and_probs,pred_time

example_list = [["examples/" + example]for example in os.listdir("examples")]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3,label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    title="Pixel Digits Classification App",
)
demo.launch(share=True)
