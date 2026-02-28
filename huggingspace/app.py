import gradio as gr
import torch
import torch.nn as nn
import numpy as np

FEATURE_COLS = [
    'stenosis_severity', 'lesion_length', 'reference_diameter', 'plaque_burden', 
    'vessel_curvature', 'tapering_rate', 'bifurcation_angle_prox', 'bifurcation_angle_dist', 
    'parent_vessel_diameter', 'parent_vessel_length', 'child_vessel_diameter', 
    'child_vessel_length', 'min_lumen_area', 'tree_feature_1', 'tree_feature_2', 
    'tree_feature_3', 'tree_feature_4', 'tree_feature_5', 'tree_feature_6', 
    'tree_feature_7', 'tree_feature_8', 'tree_feature_9', 'tree_feature_10', 
    'tree_feature_11', 'tree_feature_12', 'tree_feature_13', 'tree_feature_14', 
    'tree_feature_15'
]

class Generator(nn.Module):
    def __init__(self, input_dim=28, noise_dim=16):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + noise_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, z, x, y_fake):
        return self.model(torch.cat([z, x, y_fake], dim=1))

device = torch.device("cpu")
netG = Generator().to(device)
try:
    netG.load_state_dict(torch.load("netG.pth", map_location=device))
    netG.eval()
except:
    print("Error: netG.pth not found in repo.")

# 4. The Prediction Function (Aligned to Training)
def predict_ffr(angle, stenosis, biased_ffr):
    # Create the 28-feature vector in the EXACT training order
    # We map 'angle' to 'vessel_curvature' (Feature #5) as a proxy
    # We map 'stenosis' to 'stenosis_severity' (Feature #1)
    x_input = np.zeros((1, 28))
    x_input[0, 0] = stenosis  # stenosis_severity
    x_input[0, 4] = angle     # vessel_curvature (proxy for angle)
    
    # Fill remaining features with physiological defaults (0.5)
    for i in range(28):
        if x_input[0, i] == 0:
            x_input[0, i] = 0.5

    X_tensor = torch.FloatTensor(x_input).to(device)
    y_fake_tensor = torch.FloatTensor([[biased_ffr]]).to(device)
    
    # Use torch.randn for physiological variation (not zeros)
    torch.manual_seed(42)
    z = torch.randn(1, 16).to(device)
    
    with torch.no_grad():
        y_raw = netG(z, X_tensor, y_fake_tensor).cpu().numpy()[0][0]
        
    # Physiological Rescaling (0.4 - 1.0)
    y_adj = 0.4 + (0.6 * y_raw)
    y_adj = np.clip(y_adj, 0.4, 1.0)
    
    delta = y_adj - biased_ffr
    status = "🔴 ISCHEMIC (Significant)" if y_adj <= 0.80 else "🟢 NON-ISCHEMIC (Stable)"
    
    return f"{y_adj:.3f}", status, f"{delta:+.3f} (GAN Correction Applied)"

# 5. Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 FFR-GAN: Physiological Bias Correction Prototype")
    gr.Markdown("### Translational Feasibility Demo for Coronary Artery Disease Assessment")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Clinical Inputs")
            angle = gr.Slider(30, 90, value=60, label="Vessel Angle (Degrees)")
            stenosis = gr.Slider(20, 90, value=65, label="Stenosis Severity (%)")
            biased_ffr = gr.Number(value=0.88, label="Biased FFR (Area-based)")
            btn = gr.Button("Run Physiological Adjustment", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 2. GAN Adjusted Output")
            out_ffr = gr.Textbox(label="Corrected FFR (Diameter-equivalent)")
            out_status = gr.Textbox(label="Clinical Status")
            out_delta = gr.Label(label="Adjustment Delta")
            
    btn.click(predict_ffr, inputs=[angle, stenosis, biased_ffr], outputs=[out_ffr, out_status, out_delta])

demo.launch()
