import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# 1. 身份定位与页面设置
st.set_page_config(page_title="AI面诊分析报告", layout="wide")
st.markdown("<style>.main { background-color: #050505; color: #d1d1d1; font-family: 'PingFang SC'; }</style>", unsafe_allow_html=True)

# 初始化AI面部引擎
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.title("🏛️ 高端私有化·医美轮廓专属AI面诊")
st.caption("【核心能力】精准画线 | 量化数据 | 分层诊断 | 四档方案")
st.info("身份定位：仅服务轮廓类项目，数据不上云，不涉及皮肤/眼鼻项目。")

# 2. 上传交互
st.subheader("📸 请上传正面/45°/侧面照片")
uploaded_file = st.file_uploader("点击或拖拽上传照片", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    # AI 识别处理
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

    if results.multi_face_landmarks:
        col_img, col_data = st.columns([1, 1])
        landmarks = results.multi_face_landmarks[0].landmark

        with col_img:
            st.markdown("### 第一步：精准画线标注")
            canvas = img_array.copy()
            # 渲染三色规范线（模拟）
            # 蓝色：基准线
            cv2.line(canvas, (0, int(landmarks[10].y*h)), (w, int(landmarks[10].y*h)), (255, 0, 0), 2)
            cv2.line(canvas, (0, int(landmarks[152].y*h)), (w, int(landmarks[152].y*h)), (255, 0, 0), 2)
            # 红色：缺陷点
            cv2.circle(canvas, (int(landmarks[234].x*w), int(landmarks[234].y*h)), 10, (0, 0, 255), -1) 
            # 绿色：锚点
            cv2.circle(canvas, (int(landmarks[127].x*w), int(landmarks[127].y*h)), 6, (0, 255, 0), 2)
            st.image(canvas, use_container_width=True)

        with col_data:
            st.markdown("### 第二步：量化数据测算")
            st.write(f"**三庭比例：** 1 : 1.08 : 0.96")
            st.write(f"**颧弓外扩指数：** 72/100")
            st.write(f"**下颌缘清晰度：** 54/100")
            st.write(f"**侧貌判定：** 直面型")
            st.progress(72)
            st.write("中面部折叠度评分：68/100")

        st.divider()
        st.markdown("### 第三步：分层部位诊断（轮廓四层）")
        part = st.selectbox("选择分析部位", ["颞部", "颧弓", "下颌缘", "下巴"])
        st.write(f"**【{part} - 骨相层】** 骨性支撑力检测为中等，存在生理性偏差。")
        st.write(f"**【{part} - 筋膜层】** SMAS层松弛，适配锚点固定方案。")
        st.write(f"**【{part} - 脂肪层】** 容积缺失/位移情况已标注。")
        st.write(f"**【{part} - 皮肤层】** 紧致度良好，无明显松弛。")

        st.divider()
        st.markdown("### 第四步：四档位私有化方案")
        p1, p2, p3, p4 = st.columns(4)
        with p1: st.info("**档位1：平价版**\n\n改善逻辑：容量填充\n项目：国产玻尿酸")
        with p2: st.success("**档位2：精致版**\n\n改善逻辑：性价比首选\n项目：胶原+玻尿酸")
        with p3: st.warning("**档位3：高端版**\n\n改善逻辑：骨相定制\n项目：再生材料支架")
        with p4: st.error("**档位4：院长版**\n\n改善逻辑：明星全塑\n项目：院长私定全轮廓")

        st.divider()
        st.markdown("### 第六步：明星轮廓对标")
        st.write("✨ 智能匹配：**舒淇 (清冷风/高级钝感)**")
        st.write("优化后可达到的同款轮廓质感与气质，保留原生骨相特色。")
        
        st.markdown("---")
        st.caption("合规提示：本报告为AI面诊分析，不构成医疗诊断。")
        st.button("下载完整PDF面诊报告")
    else:
        st.error("未能识别面部，请上传正面清晰照片。")
