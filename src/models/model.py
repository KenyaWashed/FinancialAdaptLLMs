# from transformers import AutoModelForCausalLM

# def get_model(**kwargs):
#     if kwargs.pop("model_parallel")==True:
#         model = AutoModelForCausalLM.from_pretrained(device_map="auto", **kwargs)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(**kwargs)
#     return model

import torch
from huggingface_hub import hf_hub_download

# MÀY PHẢI PASTE TOÀN BỘ ĐỊNH NGHĨA CLASS TinyRecursiveModel CỦA MÀY VÀO ĐÂY
# (Hoặc import nó từ file code của mày, ví dụ: from my_custom_code import TinyRecursiveModel)
# class TinyRecursiveModel(torch.nn.Module):
#     ... (code của mày) ...

def get_model(pretrained_model_name_or_path, **kwargs):
    # --- ĐOẠN CHẶN ĐẦU CHO MODEL CUSTOM CỦA MÀY ---
    if pretrained_model_name_or_path == "KenyaWashed/trm-convfinqa":
        print("====== ĐANG BYPASS ĐỂ LOAD TINY RECURSIVE MODEL ======")
        
        # 1. Khởi tạo cái khung model của mày (Nhớ điền param cho đúng)
        model = TinyRecursiveModel() 
        
        # 2. Tải cái file .pt từ Hugging Face về
        weight_path = hf_hub_download(repo_id="KenyaWashed/trm-convfinqa", filename="pytorch_model.bin")
        
        # 3. Mở file checkpoint ra
        checkpoint = torch.load(weight_path, map_location="cpu")
        
        # 4. Trích xuất đúng cái state_dict (sửa chữ 'model_state_dict' nếu lúc train mày đặt tên key khác)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint # Nếu nó là weights thuần
            
        # 5. Load vào model
        model.load_state_dict(state_dict)
        
        # Chuyển vào GPU nếu có
        if torch.cuda.is_available():
            model = model.cuda()
            
        model.eval()
        return model

    # --- ĐOẠN CODE GỐC CỦA ADAPTLLMS GIỮ NGUYÊN BÊN DƯỚI ---
    # (Khúc này thường là auto_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, ...))
    ...
