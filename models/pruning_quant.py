import torch
import torch.nn.utils.prune as prune
import torch.quantization as quantization

def prune_model(model, amount=0.2, structured=False):
    """
    对模型所有Conv2d层进行剪枝。
    structured: True为按通道结构化剪枝，False为L1非结构化剪枝。
    剪枝后自动移除掩码。
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    if structured:
        for module, param in parameters_to_prune:
            prune.ln_structured(module, name=param, amount=amount, n=2, dim=0)  # 按通道剪枝
            prune.remove(module, param)
    else:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        for module, param in parameters_to_prune:
            prune.remove(module, param)
    return model

def quantize_model(model, static=False, data_loader=None):
    """
    对模型进行量化。
    static: True为静态量化（需校准数据），False为动态量化。
    """
    model.eval()
    if static:
        if data_loader is None:
            raise ValueError("静态量化需提供校准数据data_loader")
        model.qconfig = quantization.get_default_qconfig('fbgemm')
        quantization.prepare(model, inplace=True)
        # 校准
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'] if isinstance(batch, dict) else batch[0]
                model(images)
        quantization.convert(model, inplace=True)
        return model
    else:
        quantized_model = quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        return quantized_model 