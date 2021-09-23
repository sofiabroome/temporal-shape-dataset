from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    encoder_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
        if not 'linear' in name:
            encoder_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return encoder_params, total_params


