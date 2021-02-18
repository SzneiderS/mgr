import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from torchvision.transforms import transforms
import torch

from Models.BaseModel import BaseModule

import base64
from io import BytesIO
import random

tensorize = transforms.ToTensor()
to_img = transforms.Compose([
    transforms.ToPILImage()
])

app = dash.Dash()

app.layout = html.Div(className="container", children=[
    html.Button(id='random-vector-button', n_clicks=0, children='Wylosuj wektor'),
    html.Div(id="sliders_div", className="sliders"),
    html.Div(className="flex width100 overflow-hidden result-div", children=[
        html.Img(src="", id="result_image", className="height100 float-right")
    ]),
    html.Div(id="drop", style={"display": "hidden"})
])

net_file = "/sdb7/seb/ready/vae3"

net = BaseModule.load(net_file)

variational = net.encoder.variational
dec = net.decoder
vector_size = dec.fc.in_features

del net


vec = torch.zeros(vector_size)
if torch.cuda.is_available:
    dec = dec.cuda()

img = app.layout["result_image"]
sliders = app.layout["sliders_div"]
sliders.children = []


sliders_num = int(vector_size / 4)
slider_inputs = []
slider_states = []
range_val = 3 if variational else 1
for i in range(sliders_num):
    slider = dcc.Slider(min=-range_val, max=range_val, step=0.01, className="slider")
    slider.value = 0.0
    slider.id = "slider" + str(i)
    slider.updatemode = "drag"
    sliders.children.append(slider)
    slider_inputs.append(Input(slider.id, 'value'))
    slider_states.append(State(slider.id, 'value'))

import torch.nn.functional as F

@app.callback(Output('result_image', 'src'),
              slider_inputs)
def val_changed(*args):
    data = torch.tensor(args).float()
    data.squeeze_()
    # data = data.view(1, 1, data.size(0))
    # data = F.interpolate(data, vector_size)
    # data.squeeze_()
    # return get_str_from_vec(data)
    repeats = int(vector_size / data.size()[0]) + 1
    return get_str_from_vec(data.repeat(repeats).resize_(vector_size))


@app.callback(Output("sliders_div", "children"),
              [Input("random-vector-button", "n_clicks")])
def random_vector_button(n_clicks):
    if variational:
        data = torch.randn(sliders_num)
    else:
        data = torch.rand(sliders_num) * 2 * range_val - range_val
    repeats = int(sliders_num / data.size()[0]) + 1
    data = data.repeat(repeats).resize_(sliders_num)
    for slider, val in zip(sliders.children, data):
        slider.value = val.item()
    val_changed(data.tolist())
    return sliders.children


def get_str_from_vec(v):
    buffered = BytesIO()
    if torch.cuda.is_available:
        v = v.cuda()
    with torch.no_grad():
        image = torch.sigmoid(dec.forward(v)[0]).cpu()
        to_img(image).save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return "data:image/png;base64," + img_str


if __name__ == '__main__':
    app.run_server(debug=True, port=27016, host="0.0.0.0")
