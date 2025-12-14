import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models 
import gdown
import os

# ==============================================================================
# 1. KONFIGURASI DAN HYPERPARAMETER KRITIS
# ==============================================================================

# HARUS DIGANTI: FILE ID GOOGLE DRIVE Anda
DRIVE_FILE_ID = '1j8bnvJhnDB4N0bzn2CmBQQqXj05LpG6-' 
MODEL_PATH_LOCAL = 'OCR_model_downloaded.pth' 

# CHAR_LIST berdasarkan Colab (36 karakter: 0-9 dan a-z)
CHAR_LIST = "0123456789abcdefghijklmnopqrstuvwxyz" 

OUTPUT_CLASSES = len(CHAR_LIST) + 1 
HIDDEN_SIZE = 256  
RNN_LAYERS = 2     

# ==============================================================================
# 2. DEFINISI ARSITEKTUR MODEL (FINAL)
# ==============================================================================

class ReconstructedOCRModel(nn.Module):
    def __init__(self, num_classes, hidden_size, rnn_layers):
        super(ReconstructedOCRModel, self).__init__()
        
        mobilenet = models.mobilenet_v2(weights=None)
        self.cnn = mobilenet.features 
        
        first_conv = self.cnn[0][0]
        # Input CNN diatur ke 3 channel (karena di Colab menggunakan transforms.Grayscale(num_output_channels=3))
        self.cnn[0][0] = nn.Conv2d(3, first_conv.out_channels, 
                                   kernel_size=first_conv.kernel_size, 
                                   stride=first_conv.stride, 
                                   padding=first_conv.padding, 
                                   bias=first_conv.bias is not None)
        
        C_out = mobilenet.last_channel 
        
        # Memisahkan LSTM dan Linear untuk mencocokkan nama key 'rnn' dan 'fc' di checkpoint
        self.rnn = nn.LSTM(C_out, hidden_size, rnn_layers, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        x = self.cnn(x) 
        
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.view(B, W, C * H)
        x = x.permute(1, 0, 2)
        
        recurrent_features, _ = self.rnn(x)
        output = self.fc(recurrent_features)
        
        return output

# ==============================================================================
# 3. FUNGSI PEMUATAN MODEL DENGAN DOWNLOAD EXTERNAL
# ==============================================================================

@st.cache_resource
def load_model_and_weights(file_id, local_path, output_classes, hidden_size, rnn_layers):
    
    if not os.path.exists(local_path):
        st.info("📦 Memulai unduhan model dari Google Drive...")
        try:
            gdown.download(id=file_id, output=local_path, quiet=False)
            st.success("✅ Model berhasil diunduh ke disk lokal.")
        except Exception as e:
            st.error(f"❌ Gagal mengunduh model dari Google Drive. Error: {e}")
            return None
    else:
        st.info("📂 File model sudah ada di disk. Melanjutkan ke pemuatan.")

    model = ReconstructedOCRModel(output_classes, hidden_size, rnn_layers) 
    
    try:
        state_dict = torch.load(local_path, map_location=torch.device('cpu'))
        
        # Ekstraksi bobot dari sub-key 'model_state' (sesuai kode Colab Anda)
        if 'model_state' in state_dict:
             state_dict = state_dict['model_state'] 
        
        model.load_state_dict(state_dict, strict=True) 
        model.eval() 
        st.success("🎉 Model berhasil dimuat dan siap digunakan!")
        return model
        
    except Exception as e:
        st.error(f"❌ Gagal memuat arsitektur model. Error: {e}")
        st.warning("Periksa ulang: CHAR_LIST (Output Classes), HIDDEN_SIZE, dan RNN_LAYERS.")
        return None

# ==============================================================================
# 4. FUNGSI UTILITY (PRE/POST-PROCESSING FINAL)
# ==============================================================================

def preprocess_image(image: Image.Image):
    # KOREKSI: Gunakan Dimensi Colab (W=256)
    target_width = 256
    target_height = 32
    
    # 1. Resize
    image = image.resize((target_width, target_height))
    
    # 2. Grayscale lalu ubah ke RGB (sesuai transforms.Grayscale(num_output_channels=3) di Colab)
    image = image.convert('L').convert('RGB')
    
    # 3. To NumPy Array dan Normalisasi (RGB: H, W, C)
    image_array = np.array(image, dtype=np.float32) / 255.0 
    
    # 4. Transpose ke PyTorch (C, H, W)
    image_array = image_array.transpose((2, 0, 1)) 
    
    # 5. Normalisasi ke [-1, 1] (sesuai transforms.Normalize(mean=0.5, std=0.5) di Colab)
    mean = np.array([0.5, 0.5, 0.5]).reshape(-1, 1, 1)
    std = np.array([0.5, 0.5, 0.5]).reshape(-1, 1, 1)
    image_array = (image_array - mean) / std
    
    # 6. To Tensor dan Tambahkan Batch Dimension
    image_tensor = torch.from_numpy(image_array).unsqueeze(0) 
    
    return image_tensor

def postprocess_output(output_tensor):
    probs = output_tensor.softmax(2)
    _, preds_index = probs.max(2)
    preds_index = preds_index.squeeze(1).tolist()
    
    raw_text = []
    for i in range(len(preds_index)):
        idx = preds_index[i]
        # Hapus Blank (0) dan Duplikasi
        if idx != 0 and (i == 0 or idx != preds_index[i-1]):
            raw_text.append(CHAR_LIST[idx - 1]) 
    
    # Model dilatih hanya untuk huruf kecil
    return "".join(raw_text).lower()

# ==============================================================================
# 5. APLIKASI STREAMLIT UTAMA
# ==============================================================================

st.set_page_config(page_title="MobileNet+CRNN OCR", layout="wide")
st.title("📖 Word-level OCR Deployment (MobileNet + CRNN)")
st.markdown("Aplikasi untuk menguji model Word-level OCR.")
st.divider()

model = load_model_and_weights(
    DRIVE_FILE_ID, 
    MODEL_PATH_LOCAL, 
    OUTPUT_CLASSES, 
    HIDDEN_SIZE, 
    RNN_LAYERS
)

if model:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Unggah Gambar")
        st.markdown("**(Disarankan: Unggah gambar kata tunggal yang sudah di-*crop* untuk hasil terbaik)**")
        uploaded_file = st.file_uploader(
            "Pilih gambar kata (.jpg, .png)", 
            type=["jpg", "png", "jpeg"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
            
            st.markdown("---")
            if st.button('🚀 Jalankan OCR', type="primary", use_container_width=True):
                with st.spinner('Memproses gambar dan menjalankan model di CPU...'):
                    try:
                        input_tensor = preprocess_image(image)
                        
                        with torch.no_grad():
                            output_tensor = model(input_tensor)
                        
                        result_text = postprocess_output(output_tensor)
                        
                        st.session_state['ocr_result'] = result_text
                        st.session_state['processed'] = True

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat inferensi: {e}")

    with col2:
        st.subheader("Hasil OCR")
        if 'processed' in st.session_state and st.session_state['processed']:
            st.success("✅ OCR Selesai!")
            st.code(st.session_state['ocr_result'], language='text')
            
            st.markdown(f"""
            <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; font-size: 1.2em;">
                <p><strong>Hasil Teks Akhir:</strong></p>
                <p style="font-weight: bold; color: #1e88e5;">{st.session_state['ocr_result']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif uploaded_file is None:
             st.info("Silakan unggah gambar di sebelah kiri dan klik 'Jalankan OCR'.")


else:
    st.warning("Aplikasi menunggu model dimuat dari Drive. Cek log konsol jika proses terhenti.")
