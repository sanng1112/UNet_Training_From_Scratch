import torch
import torch.nn as nn
from collections import OrderedDict

DEVICES = {
    "jetson_nano": {"name": "Jetson Nano", "gflops": 472},
    "jetson_xavier_nx": {"name": "Jetson Xavier NX", "gflops": 21000},
    "jetson_orin_nano": {"name": "Jetson Orin Nano", "gflops": 40000},
    "raspberry_pi_5": {"name": "Raspberry Pi 5", "gflops": 50},
    "smartphone_mid": {"name": "Mid-range Smartphone", "gflops": 1500},
}

def compute_parameters(model: nn.Module) -> int:
    """Returns the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_flops(flops: float) -> str:
    """Formats FLOPs count into human-readable string (G or M)."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} M"
    else:
        return f"{flops:,}"

def compute_flops(model: nn.Module, input_size=(320, 320)):
    """
    Computes model FLOPs using forward hooks on Conv2d, ConvTranspose2d, and Linear layers.
    Also returns a breakdown by layer class names.
    """
    flops = 0
    breakdown = {}
    
    def conv_hook(module, input, output):
        nonlocal flops
        # output shape: [B, C_out, H_out, W_out]
        b, c_out, h_out, w_out = output.shape
        c_in = module.in_channels
        kh, kw = module.kernel_size
        groups = module.groups
        # Multiply-adds count as 2 FLOPs
        f = 2 * b * c_in * (c_out // groups) * kh * kw * h_out * w_out
        flops += f
        name = module.__class__.__name__
        breakdown[name] = breakdown.get(name, 0) + f

    def linear_hook(module, input, output):
        nonlocal flops
        c_in = module.in_features
        num_elements = output.numel()
        f = 2 * num_elements * c_in
        flops += f
        name = module.__class__.__name__
        breakdown[name] = breakdown.get(name, 0) + f
        
    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
            
    device = next(model.parameters()).device
    x = torch.randn(1, 3, *input_size, device=device)
    model.eval()
    with torch.no_grad():
        model(x)
        
    for h in handles:
        h.remove()
        
    return flops, breakdown

def compute_flops_by_component(model: nn.Module, input_size=(320, 320)):
    """
    Computes model FLOPs broken down by structural groups: Encoder, Decoder, Bottleneck, SegHead.
    """
    groups = {"Encoder": 0, "Decoder": 0, "Bottleneck": 0, "SegHead": 0}
    
    def get_group(name):
        name_lower = name.lower()
        if "encoder" in name_lower:
            return "Encoder"
        elif "decoder" in name_lower:
            return "Decoder"
        elif "bottleneck" in name_lower:
            return "Bottleneck"
        elif "final_conv" in name_lower or "seg_head" in name_lower:
            return "SegHead"
        else:
            # Fallback based on component structure
            if any(p in name_lower for p in ["stem", "downsample"]):
                return "Encoder"
            elif any(p in name_lower for p in ["upsample", "cross"]):
                return "Decoder"
            return "Encoder"

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            group_name = get_group(name)
            
            def make_hook(g_name):
                def hook(module, input, output):
                    b, c_out, h_out, w_out = output.shape
                    c_in = module.in_channels
                    kh, kw = module.kernel_size
                    groups_val = module.groups
                    f = 2 * b * c_in * (c_out // groups_val) * kh * kw * h_out * w_out
                    groups[g_name] += f
                return hook
                
            handles.append(m.register_forward_hook(make_hook(group_name)))
            
    device = next(model.parameters()).device
    x = torch.randn(1, 3, *input_size, device=device)
    model.eval()
    with torch.no_grad():
        model(x)
        
    for h in handles:
        h.remove()
        
    return groups

def estimate_edge_latency(flops: float, params: int, device_key: str, precision: str = "fp32") -> dict:
    """
    Estimates latency and FPS on edge hardware based on model GFLOPs, parameter count,
    and device characteristics.
    """
    device = DEVICES[device_key]
    peak_gflops = device["gflops"]
    
    # Efficiency factor accounts for memory bandwidth limitations, cache misses, and framework overhead
    efficiency = 0.08 if "jetson" in device_key else 0.05
    effective_gflops = peak_gflops * efficiency
    
    # Compute latency
    comp_latency = flops / (effective_gflops * 1e9)
    
    # Memory bandwidth latency (for loading parameters)
    bandwidths = {
        "jetson_nano": 25.6,
        "jetson_xavier_nx": 51.2,
        "jetson_orin_nano": 68.0,
        "raspberry_pi_5": 15.0,
        "smartphone_mid": 30.0,
    }
    # FP32 weights = 4 bytes per param
    mem_size = params * 4
    mem_latency = mem_size / (bandwidths.get(device_key, 20.0) * 1e9)
    
    total_latency = max(comp_latency, mem_latency)
    fps = 1.0 / total_latency
    
    return {"latency_ms": total_latency * 1000, "fps": fps}
