import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models # Digunakan untuk MobileNet

# --- Konfigurasi Model dan Hyperparameter ---

MODEL_PATH = 'OCR_model.pth' 

# GANTI INI DENGAN KAMUS KARAKTER ASLI ANDA
# Ini adalah tebakan umum. Sesuaikan jika model Anda punya karakter khusus (misal: aksara, simbol)
CHAR_LIST = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?'-:;()[]{}*&^%$#@~`|+=\\/" 

# +1 untuk CTC 'Blank' token. Ini WAJIB.
OUTPUT_CLASSES = len(CHAR_LIST) + 1 
HIDDEN_SIZE = 256  # Tebakan umum untuk LSTM hidden size. Sesuaikan jika Anda tahu nilai pastinya.
RNN_LAYERS = 2     # Tebakan umum jumlah layer LSTM. Sesuaikan jika Anda tahu nilai pastinya.

# --- Definisi Arsitektur Model (REKONSTRUKSI) ---
# Berdasarkan informasi: MobileNet + CRNN + CTC

class ReconstructedOCRModel(nn.Module):
    def __init__(self, num_classes, hidden_size, rnn_layers):
        super(ReconstructedOCRModel, self).__init__()
        
        # 1. CNN Backbone: MobileNet (Digunakan sebagai fitur ekstraktor)
        mobilenet = models.mobilenet_v2(weights=None)
        
        self.cnn = mobilenet.features 
        
        first_conv = self.cnn[0][0]
        self.cnn[0][0] = nn.Conv2d(1, first_conv.out_channels, 
                                   kernel_size=first_conv.kernel_size, 
                                   stride=first_conv.stride, 
                                   padding=first_conv.padding, 
                                   bias=first_conv.bias is not None)
        
        # 2. Sequential Layer untuk menyesuaikan output CNN ke RNN
        C_out = mobilenet.last_channel 
        
        # 3. RNN (LSTM)
        self.rnn = nn.Sequential(
            nn.LSTM(C_out, hidden_size, rnn_layers, bidirectional=True, batch_first=False),
            nn.Linear(2 * hidden_size, num_classes) # *2 karena Bidirectional
        )

    def forward(self, x):
        # 1. CNN
        x = self.cnn(x) 
        
        # 2. Ubah Feature Map menjadi Sequence untuk RNN
        B, C, H, W = x.size()
        
        x = x.permute(0, 3, 1, 2)    
        x = x.view(B, W, C * H)      
        x = x.permute(1, 0, 2)       
        
        # 3. RNN/LSTM dan Linear
        recurrent_features, _ = self.rnn[0](x)
        output = self.rnn[1](recurrent_features)
        
        return output

# --- Fungsi Pemuatan Model dan Utility ---

@st.cache_resource
def load_model(path):
    model = ReconstructedOCRModel(OUTPUT_CLASSES, HIDDEN_SIZE, RNN_LAYERS) 
    
    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        if 'model_state' in state_dict:
             state_dict = state_dict['model_state'] 
        
        model.load_state_dict(state_dict, strict=True) 
        model.eval() 
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Silakan periksa arsitektur `ReconstructedOCRModel` dan hyperparameter (HIDDEN_SIZE/RNN_LAYERS). Error: {e}")
        return None

def preprocess_image(image: Image.Image):
    target_width = 128
    target_height = 32
    
    image = image.convert('L') # Grayscale (1 Channel)
    image = image.resize((target_width, target_height))
    
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0) 
    return image_tensor

# Fungsi Postprocessing (CTC Greedy Decode)
def postprocess_output(output_tensor):
    """
    Melakukan CTC Greedy Decoding (mengubah output tensor menjadi string teks).
    """
    # 1. Ubah ke probabilitas
    probs = output_tensor.softmax(2) 
    
    # 2. Ambil indeks dengan probabilitas tertinggi di setiap step (Greedy)
    _, preds_index = probs.max(2) 
    preds_index = preds_index.squeeze(1).tolist() # [Sequence Length]
    
    # 3. Decode: Hapus duplikasi dan Blank token (Indeks 0)
    raw_text = []
    for i in range(len(preds_index)):
        idx = preds_index[i]
        if idx != 0 and (i == 0 or idx != preds_index[i-1]):
            raw_text.append(CHAR_LIST[idx - 1]) 
    
    return "".join(raw_text)


# --- Aplikasi Streamlit Utama ---

st.set_page_config(page_title="MobileNet+CRNN OCR", layout="wide")
st.title("📖 Word-level OCR Deployment (MobileNet + CRNN)")
st.markdown("Aplikasi untuk menguji model Word-level OCR dengan arsitektur MobileNet dan CTC Decode.")
st.divider()

model = load_model(MODEL_PATH)

if model:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Unggah Gambar")
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
                        st.info("Pastikan fungsi `preprocess_image` sudah menggunakan dimensi input yang benar.")

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
            
        elif 'processed' not in st.session_state and uploaded_file is None:
             st.info("Silakan unggah gambar di sebelah kiri dan klik 'Jalankan OCR'.")


else:
    st.warning("Model belum dapat dimuat. Silakan periksa konsol untuk pesan error dari PyTorch.")
