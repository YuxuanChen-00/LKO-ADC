import torch
from src.DRKO.lko_lstm_network import LKO_lstm_Network

state_size = 6
hidden_size_lstm = 128
hidden_size_mlp = 128
output_size =225
control_size = 6
delay_step = 9

net = LKO_lstm_Network(state_size, hidden_size_lstm, hidden_size_mlp, output_size, control_size, delay_step)
net.load_state_dict(torch.load("DRKO/models/LKO_hyperparameter_search_denorm_eval/delay_9_phi_225/model_delay_9_phi_225.pth"))
net.eval()

state_current = torch.zeros(1, state_size)
control_current = torch.zeros(1, control_size)
state_sequence = torch.zeros(1, delay_step-1, state_size)
control_sequence = torch.zeros(1, delay_step-1, control_size)

torch.onnx.export(
    net,
    (state_current, control_current, state_sequence, control_sequence),
    "model.onnx",
    input_names=["state_current", "control_current", "state_sequence", "control_sequence"],
    output_names=["phi_current", "phi_pred", 'state_pred'],
    opset_version=13,
    dynamic_axes={"state_current": {0: "batch_size"}, "control_current": {0: "batch_size"}, "state_sequence":
        {0: "batch_size"}, "control_sequence": {0: "batch_size"}}
)