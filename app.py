import os
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import pickle

from src.models import RGBFrequencyFusionModel, SpatioTemporalVideoModel, extract_fft_features_for_svm
from src.preprocessing import detect_and_crop_face, compute_fft, extract_video_frames
from src.dataset import get_transforms
from src.explain import GradCAM, generate_cam_overlay
# python -m streamlit run app.py
# Thiết lập trang Streamlit
st.set_page_config(
    page_title="AI Fake Content Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected Custom CSS for Premium Design (Dark Mode, Glassmorphism, Neon glow)
st.markdown("""
<style>
    /* Import font Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Thiết kế Header Gradient cực đẹp */
    .header-container {
        background: linear-gradient(135deg, #1e0034 0%, #0d001a 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #3c0068;
        box-shadow: 0 8px 32px 0 rgba(107, 0, 179, 0.2);
        text-align: center;
    }
    
    .header-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        background: linear-gradient(to right, #00f2fe, #4facfe, #b92b27);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        color: #b3b3b3;
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    /* Thiết kế thẻ Card Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, border 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 242, 254, 0.3);
    }
    
    .metric-title {
        color: #b3b3b3;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Phân biệt nhãn Real/Fake bằng màu Neon */
    .badge-real {
        background-color: rgba(0, 230, 115, 0.15);
        color: #00e673;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        border: 1px solid rgba(0, 230, 115, 0.3);
        display: inline-block;
    }
    
    .badge-fake {
        background-color: rgba(255, 77, 77, 0.15);
        color: #ff4d4d;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        border: 1px solid rgba(255, 77, 77, 0.3);
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Đọc cấu hình
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()
IMAGE_SIZE = tuple(config['preprocessing']['image_size'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiêu đề ứng dụng
st.markdown("""
<div class="header-container">
    <div class="header-title">🛡️ AI FAKE CONTENT DETECTOR</div>
    <div class="header-subtitle">Hệ thống phân cấp phát hiện nội dung giả mạo bằng phân tích Không gian, Tần số và Thời gian</div>
</div>
""", unsafe_allow_html=True)

# Tải mô hình
@st.cache_resource
def load_models():
    models_dict = {'svm': None, 'image': None, 'video': None}
    
    # 1. Load SVM
    svm_path = 'checkpoints/svm_baseline.pkl'
    if os.path.exists(svm_path):
        with open(svm_path, 'rb') as f:
            models_dict['svm'] = pickle.load(f)
            
    # 2. Load Image Fusion Model
    img_path = 'checkpoints/best_image_fusion.pth'
    if os.path.exists(img_path):
        # Inference loads every weight from our checkpoint, so avoid a needless
        # EfficientNet download and keep the web app fully usable offline.
        img_model = RGBFrequencyFusionModel(
            num_classes=config['model']['num_classes'],
            pretrained=False,
        )
        img_model.load_state_dict(torch.load(img_path, map_location='cpu'))
        img_model.to(device)
        img_model.eval()
        models_dict['image'] = img_model
        
    # 3. Load Video Model
    vid_path = 'checkpoints/best_video_temporal.pth'
    if os.path.exists(vid_path) and models_dict['image'] is not None:
        vid_model = SpatioTemporalVideoModel(image_model=models_dict['image'], num_classes=config['model']['num_classes'])
        vid_model.load_state_dict(torch.load(vid_path, map_location='cpu'))
        vid_model.to(device)
        vid_model.eval()
        models_dict['video'] = vid_model
        
    return models_dict

models_loaded = load_models()

# Sidebar cấu hình
st.sidebar.markdown("### ⚙️ CẤU HÌNH HỆ THỐNG")
threshold = st.sidebar.slider("Ngưỡng phân loại Fake (Threshold)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("### 📊 TRẠNG THÁI MÔ HÌNH")
def show_status(name, is_loaded):
    if is_loaded:
        st.sidebar.markdown(f"🟢 **{name}**: Đã sẵn sàng")
    else:
        st.sidebar.markdown(f"🔴 **{name}**: Chưa tìm thấy checkpoint (Sử dụng ngẫu nhiên)")

show_status("FFT + SVM Baseline", models_loaded['svm'] is not None)
show_status("RGB-Frequency Fusion (Ảnh)", models_loaded['image'] is not None)
show_status("Spatio-Temporal Model (Video)", models_loaded['video'] is not None)

# Thiết lập Tab
tab_img, tab_vid = st.tabs(["🖼️ PHÂN TÍCH ẢNH", "🎥 PHÂN TÍCH VIDEO"])

# ==========================================
# TAB 1: PHÂN TÍCH ẢNH
# ==========================================
with tab_img:
    st.markdown("### Phân tích đặc trưng không gian và miền tần số trên ảnh")
    img_file = st.file_uploader("Tải lên ảnh của bạn (JPG, PNG, JPEG)", type=['jpg', 'jpeg', 'png'])
    
    if img_file is not None:
        # Load image
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Tiền xử lý: resize trực tiếp về 224x224 (khớp với pipeline training CIFAKE)
        # Lưu ý: model được train trên CIFAKE (ảnh chung) nên KHÔNG dùng face detection
        proc_img = cv2.resize(img_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        fft_arr = compute_fft(proc_img)
        
        # Bố cục hiển thị ảnh
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Ảnh tải lên gốc", use_container_width=True)
            
        with col2:
            st.image(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB), caption=f"Ảnh đã resize ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]})", use_container_width=True)
            
        with col3:
            # Hiển thị phổ tần số FFT
            fig_fft, ax_fft = plt.subplots(figsize=(4, 4))
            ax_fft.imshow(fft_arr, cmap='gray')
            ax_fft.axis('off')
            st.pyplot(fig_fft, use_container_width=True)
            plt.close(fig_fft)
            st.markdown("<p style='text-align: center; color: #b3b3b3; font-size: 0.8rem;'>Phổ tần số Log-Amplitude</p>", unsafe_allow_html=True)
            
        # CHẠY SUY LUẬN
        st.markdown("---")
        st.markdown("### 🧠 Kết quả phân tích mô hình")
        
        # 1. Kết quả mô hình Fusion ảnh
        st.markdown("#### Mô hình Đề xuất: RGB-Frequency Fusion")
        
        if models_loaded['image'] is not None:
            # Chuẩn bị tensor (dùng proc_img đã resize, không phải face crop)
            transform = get_transforms(split='test')
            proc_pil = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
            rgb_tensor = transform(proc_pil).unsqueeze(0).to(device)
            fft_tensor = torch.tensor(fft_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Khởi chạy Grad-CAM
            target_layer = models_loaded['image'].spatial_branch.backbone.features[-1]
            grad_cam = GradCAM(models_loaded['image'], target_layer)
            
            rgb_tensor.requires_grad = True
            cam_mask, logits = grad_cam(rgb_tensor, fft_tensor)
            
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            fake_prob = probs[1]
            
            # Xử lý nhãn và màu sắc
            if fake_prob >= threshold:
                label_html = f'<span class="badge-fake">GIẢ MẠO (FAKE) - {fake_prob*100:.2f}%</span>'
            else:
                label_html = f'<span class="badge-real">THẬT (REAL) - {(1-fake_prob)*100:.2f}%</span>'
                
            st.markdown(f"<h5>Trạng thái nhận diện: {label_html}</h5>", unsafe_allow_html=True)
            st.caption(f"📊 Xác suất REAL: {(1-fake_prob)*100:.1f}% | FAKE: {fake_prob*100:.1f}% | Ngưỡng: {threshold:.2f}")
            
            # Hiển thị Grad-CAM
            overlay = generate_cam_overlay(proc_img, cam_mask)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(overlay, caption="Bản đồ nhiệt Grad-CAM (Vùng đáng ngờ)", use_container_width=True)
            with col_res2:
                # Vẽ biểu đồ cột phân bố xác suất
                fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
                colors = ['#00e673', '#ff4d4d']
                bars = ax_bar.barh(['Thật (Real)', 'Giả mạo (Fake)'], [1 - fake_prob, fake_prob], color=colors)
                ax_bar.set_xlim(0, 1)
                ax_bar.axvline(x=threshold, color='yellow', linestyle='--', label=f'Ngưỡng ({threshold:.2f})')
                ax_bar.set_xlabel('Xác suất (Probability)')
                ax_bar.legend(fontsize=8)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)
                
            grad_cam.remove_hooks()
        else:
            st.warning("⚠️ Chưa nạp mô hình ảnh Fusion thực tế. Đang hiển thị kết quả ngẫu nhiên do thiếu checkpoints.")
            
        # 2. Kết quả mô hình SVM Baseline
        st.markdown("#### Mô hình Baseline: FFT + SVM")
        if models_loaded['svm'] is not None:
            svm_feats = extract_fft_features_for_svm(fft_arr).reshape(1, -1)
            svm_probs = models_loaded['svm'].predict_proba(svm_feats)[0]
            svm_fake_prob = svm_probs[1]
            
            if svm_fake_prob >= threshold:
                svm_label = f'<span class="badge-fake">GIẢ MẠO (FAKE) - {svm_fake_prob*100:.2f}%</span>'
            else:
                svm_label = f'<span class="badge-real">THẬT (REAL) - {(1-svm_fake_prob)*100:.2f}%</span>'
            st.markdown(f"Trạng thái (SVM): {svm_label}", unsafe_allow_html=True)
        else:
            st.info("Chưa huấn luyện hoặc chưa nạp mô hình SVM Baseline.")

# ==========================================
# TAB 2: PHÂN TÍCH VIDEO
# ==========================================
with tab_vid:
    st.markdown("### Phân tích tính nhất quán theo thời gian trên video")
    vid_file = st.file_uploader("Tải lên video của bạn (MP4, AVI, MOV)", type=['mp4', 'avi', 'mov'])
    
    if vid_file is not None:
        # Lưu file tạm thời để OpenCV đọc
        temp_path = "temp_uploaded_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(vid_file.read())
            
        # Đọc và lấy thông tin video
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        resolution = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Bố cục hiển thị thông tin video
        col_inf1, col_inf2, col_inf3 = st.columns(3)
        with col_inf1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Số lượng Frame</div>
                <div class="metric-value">{total_frames}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_inf2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Tỷ lệ khung hình (FPS)</div>
                <div class="metric-value">{fps:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_inf3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Độ phân giải</div>
                <div class="metric-value">{resolution}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Trích xuất frames
        with st.spinner("Đang trích xuất và xử lý 16 frame của video..."):
            frames = extract_video_frames(temp_path)
            
        if len(frames) == 16:
            st.success("Trích xuất và crop khuôn mặt 16 frame thành công!")
            
            # CHẠY SUY LUẬN VIDEO
            if models_loaded['video'] is not None:
                # Tiến hành tiền xử lý cho từng frame để đưa vào mạng video
                transform = get_transforms(split='test')
                
                rgb_list = []
                fft_list = []
                face_imgs_list = []
                fft_imgs_list = []
                
                for frame in frames:
                    face_img = detect_and_crop_face(frame)
                    face_imgs_list.append(face_img)
                    
                    fft_arr = compute_fft(face_img)
                    fft_imgs_list.append(fft_arr)
                    
                    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    rgb_list.append(transform(face_pil))
                    fft_list.append(torch.tensor(fft_arr, dtype=torch.float32).unsqueeze(0))
                    
                # Stack thành tensor shape: (1, 16, C, H, W)
                clip_rgb = torch.stack(rgb_list).unsqueeze(0).to(device)
                clip_fft = torch.stack(fft_list).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits, attention_weights = models_loaded['video'](clip_rgb, clip_fft)
                    
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                fake_prob = probs[1]
                
                # Biểu thị kết quả phân loại video
                st.markdown("---")
                st.markdown("### 🧠 Kết quả phân tích chuỗi thời gian video")
                
                if fake_prob >= threshold:
                    vid_label = f'<span class="badge-fake">VIDEO GIẢ MẠO (FAKE) - {fake_prob*100:.2f}%</span>'
                else:
                    vid_label = f'<span class="badge-real">VIDEO THẬT (REAL) - {(1-fake_prob)*100:.2f}%</span>'
                    
                st.markdown(f"<h4>Trạng thái video: {vid_label}</h4>", unsafe_allow_html=True)
                
                # Trực quan hóa attention weights theo timeline
                att_weights_np = attention_weights[0].cpu().numpy().flatten()
                
                col_gr1, col_gr2 = st.columns(2)
                with col_gr1:
                    # Vẽ timeline attention
                    fig_att, ax_att = plt.subplots(figsize=(6, 3))
                    ax_att.plot(range(1, 17), att_weights_np, marker='o', color='#8884d8', linewidth=2)
                    ax_att.set_xlabel('Frame Index trong Clip')
                    ax_att.set_ylabel('Attention Weight (Độ nghi vấn)')
                    ax_att.set_title('Timeline phân phối độ nghi vấn theo thời gian')
                    ax_att.set_xticks(range(1, 17))
                    st.pyplot(fig_att, use_container_width=True)
                    plt.close(fig_att)
                    
                with col_gr2:
                    # Tìm frame có attention cao nhất (đáng ngờ nhất)
                    suspicious_idx = np.argmax(att_weights_np)
                    st.markdown(f"**Khung hình đáng ngờ nhất:** Frame thứ **{suspicious_idx+1}** (Attention = {att_weights_np[suspicious_idx]:.4f})")
                    
                    # Chạy Grad-CAM cho frame đáng ngờ nhất để giải thích spatial
                    target_layer = models_loaded['video'].image_encoder.spatial_branch.backbone.features[-1]
                    grad_cam = GradCAM(models_loaded['video'].image_encoder, target_layer)
                    
                    # Chuẩn bị đầu vào cho frame đáng ngờ
                    rgb_frame_tensor = clip_rgb[:, suspicious_idx].clone()
                    fft_frame_tensor = clip_fft[:, suspicious_idx].clone()
                    
                    rgb_frame_tensor.requires_grad = True
                    cam_mask_susp, _ = grad_cam(rgb_frame_tensor, fft_frame_tensor)
                    overlay_susp = generate_cam_overlay(face_imgs_list[suspicious_idx], cam_mask_susp)
                    
                    st.image(overlay_susp, caption=f"Vùng bất thường trên khuôn mặt ở Frame {suspicious_idx+1}", use_container_width=True)
                    grad_cam.remove_hooks()
                    
                # Cho phép trượt và xem toàn bộ 16 frame
                st.markdown("#### 🎞️ Duyệt chi tiết 16 frame đã crop")
                frame_sel = st.slider("Chọn frame để xem chi tiết", 1, 16, 1) - 1
                
                col_fr1, col_fr2 = st.columns(2)
                with col_fr1:
                    st.image(cv2.cvtColor(face_imgs_list[frame_sel], cv2.COLOR_BGR2RGB), caption=f"Frame {frame_sel+1} khuôn mặt", use_container_width=True)
                with col_fr2:
                    fig_fft_sel, ax_fft_sel = plt.subplots(figsize=(4, 4))
                    ax_fft_sel.imshow(fft_imgs_list[frame_sel], cmap='gray')
                    ax_fft_sel.axis('off')
                    st.pyplot(fig_fft_sel, use_container_width=True)
                    plt.close(fig_fft_sel)
                    st.markdown("<p style='text-align: center; color: #b3b3b3; font-size: 0.8rem;'>Phổ FFT tương ứng</p>", unsafe_allow_html=True)
                    
            else:
                st.warning("⚠️ Chưa nạp mô hình video (BiGRU). Vui lòng chạy huấn luyện mô hình video trước.")
        else:
            st.error("Không thể trích xuất đủ 16 frame từ video này.")
            
        # Xóa file tạm
        if os.path.exists(temp_path):
            os.remove(temp_path)
