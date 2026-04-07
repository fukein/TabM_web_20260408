# import streamlit as st
# import numpy as np
# import pandas as pd
# import torch
# import sklearn.metrics
# import sklearn.preprocessing
# import matplotlib.pyplot as plt
# from io import BytesIO  
# import warnings  
# warnings.filterwarnings('ignore')

# # ===================== 页面配置 =====================
# st.set_page_config(page_title="南疆复播大豆粒重预测", page_icon="🌱", layout="wide")
# # 字体设置（匹配指定绘图样式：中文宋体）
# plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['axes.linewidth'] = 1.2
# plt.rcParams['font.size'] = 12

# # ===================== 核心配置（100%匹配训练代码） =====================
# # 加载训练集，复用训练时的预处理参数（无偏差关键）
# train_df = pd.read_excel("6特征_训练集.xlsx")
# # 数值特征（4个核心）
# num_features = ['荚数', '粒数', '复叶位置', '累积日摄量']
# x_num_train = train_df[num_features].values.astype(np.float32)
# # 类别特征（2个）
# cat_features = ['品种', '生育期']
# SELECTED_FEATURES = num_features + cat_features
# # 均值/标准差（训练集真实值）
# y_mean = train_df["粒重"].mean()
# y_std = train_df["粒重"].std()

# # 类别映射（手动输入时用）
# 品种映射 = {
#     "新大豆23号":0,"龙垦324":1,"龙垦3092":2,"五豆188":3,
#     "新大豆23号翻耕":4,"新大豆26号":5,"新大豆23号稀植":6
# }
# 生育期映射 = {"花荚期":0,"始粒期":1,"鼓粒初期":2,"鼓粒末期":3}

# device = torch.device('cpu')

# # ===================== 数据预处理（完全复现训练逻辑） =====================
# # 加噪声+分位数归一化（和训练代码一致）
# noise = np.random.default_rng(0).normal(0.0, 1e-5, x_num_train.shape).astype(x_num_train.dtype)
# preprocessor = sklearn.preprocessing.QuantileTransformer(
#     n_quantiles=max(min(len(x_num_train)//30, 1000), 10),
#     output_distribution='normal'
# ).fit(x_num_train + noise)

# # ===================== 模型加载（匹配训练结构） =====================
# @st.cache_resource(show_spinner=False)
# def build_and_load_model():
#     import rtdl_num_embeddings
#     import tabm
    
#     # 用训练集数值特征维度构建嵌入层
#     dummy_x_num = torch.randn(100, len(num_features))
#     num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
#         rtdl_num_embeddings.compute_bins(dummy_x_num, n_bins=48),
#         d_embedding=16, activation=False, version='B'
#     )
#     # 类别基数：品种7类、生育期4类（和训练一致）
#     model = tabm.TabM.make(
#         n_num_features=len(num_features),
#         cat_cardinalities=[7, 4],
#         d_out=1,
#         num_embeddings=num_embeddings
#     ).to(device)

#     # 加载权重（删除版本冲突的mask键）
#     state_dict = torch.load("6特征_大豆粒重模型.pth", map_location=device)
#     state_dict.pop("num_module.impl.mask", None)
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model

# model = build_and_load_model()

# # ===================== 预测函数（无偏差核心） =====================
# @torch.no_grad()
# def predict(df_input, is_batch=False):
#     df = df_input.copy()
    
#     # 区分手动/批量：手动需映射名称→编码，批量已编码直接用
#     if not is_batch:
#         df["品种"] = df["品种"].map(品种映射)
#         df["生育期"] = df["生育期"].map(生育期映射)
#     else:
#         # 批量数据（测试集）过滤无效值，确保类别非负
#         df["品种"] = df["品种"].astype(int)
#         df["生育期"] = df["生育期"].astype(int)
#         df = df[(df["品种"] >= 0) & (df["生育期"] >= 0)].reset_index(drop=True)

#     # 数值特征预处理（复用训练预处理）
#     x_num = df[num_features].values.astype(np.float32)
#     x_num = preprocessor.transform(x_num)
#     # 类别特征转int
#     x_cat = df[cat_features].values.astype(np.int64)

#     # 模型预测（复现训练时的多分支平均）
#     x_num_t = torch.tensor(x_num, device=device)
#     x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=device)
#     pred = model(x_num_t, x_cat_t).cpu().numpy()
#     pred = pred.mean(axis=1)  # 多分支输出取平均
#     pred = pred * y_std + y_mean  # 还原真实粒重

#     if not is_batch:
#         return float(pred[0])  # 手动返回标量
#     else:
#         return df, pred  # 批量返回有效数据+预测值

# # ===================== 绘图函数（匹配指定测试集散点风格） =====================
# def plot_test_scatter(y_true, y_pred):
#     # 画布配置（长方形，匹配指定样式）
#     fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=100)
    
#     # 背景色：浅灰蓝
#     ax.set_facecolor('#F7F9FC')
    
#     # 测试集散点（黑色，不透明）
#     ax.scatter(y_true, y_pred, s=38, c='#2A2A2A', alpha=0.95, label="测试集")

#     # 拟合线（红色虚线，线宽2.3）
#     z = np.polyfit(y_true, y_pred, 1)
#     k, b = z
#     x_line = np.linspace(0, y_true.max() * 1.05, 200)
#     y_fit = k * x_line + b
#     ax.plot(x_line, y_fit, '#E63946', linestyle='--', linewidth=2.3, 
#             label=f"y = {k:.4f}x {b:+.4f}")

#     # 误差带（±10%青蓝系，±20%暖橙系）
#     theta = np.arctan(k) if k != 0 else np.pi/4
#     dy = np.cos(theta)
#     band_10 = y_true.mean() * 0.10
#     band_20 = y_true.mean() * 0.20

#     # ±10% 渐变色
#     y10_upper = y_fit + band_10 * dy
#     y10_lower = y_fit - band_10 * dy
#     ax.fill_between(x_line, y_fit, y10_upper, color='#457B9D', alpha=0.7)
#     ax.fill_between(x_line, y10_lower, y_fit, color='#A8DADC', alpha=0.7)
#     ax.fill_between(x_line, y10_lower, y10_upper, color='#C1E3E4', alpha=0.25)

#     # ±20% 渐变色
#     y20_upper = y_fit + band_20 * dy
#     y20_lower = y_fit - band_20 * dy
#     ax.fill_between(x_line, y10_upper, y20_upper, color='#FFB700', alpha=0.3)
#     ax.fill_between(x_line, y20_lower, y10_lower, color='#FFE156', alpha=0.3)

#     # 图例（误差带标识）
#     ax.plot([], [], color='#457B9D', lw=10, alpha=0.6, label='±10%')
#     ax.plot([], [], color='#FFB700', lw=10, alpha=0.5, label='±20%')

#     # 坐标轴配置
#     ax.set_xlabel('观测值（g）', fontsize=14, labelpad=8)
#     ax.set_ylabel('预测值（g）', fontsize=14, labelpad=8)
#     ax.grid(alpha=0.6, lw=0.8, color='white')  # 白色网格
#     ax.legend(fontsize=11, loc='upper left', frameon=False)
#     ax.set_aspect('auto')  # 取消正方形布局

#     # 坐标范围（包含所有数据点）
#     vmax = max(y_true.max(), y_pred.max()) * 1.05
#     ax.set_xlim(0, vmax)
#     ax.set_ylim(0, vmax)

#     # 调整布局
#     plt.tight_layout()
#     return fig

# # ===================== Excel下载函数（解决报错） =====================
# def df_to_excel(df):
#     output = BytesIO()
#     with pd.ExcelWriter(output, engine='openpyxl') as writer:
#         df.to_excel(writer, index=False, sheet_name='测试集预测结果')
#     output.seek(0)
#     return output

# # ===================== 界面配置 =====================
# st.title("🌱 南疆复播大豆生理指标 → 粒重预测")
# mode = st.sidebar.radio("选择功能模式", ["手动单样本预测", "测试集批量预测"])

# # ---------------------- 1. 手动单样本预测 ----------------------
# if mode == "手动单样本预测":
#     st.subheader("📝 手动输入预测（输入原始特征名称）")
#     # 分2列布局输入框
#     col1, col2 = st.columns(2)
#     with col1:
#         荚数 = st.number_input("荚数（个）", min_value=0, max_value=200, value=40, step=1)
#         粒数 = st.number_input("粒数（粒）", min_value=0, max_value=1000, value=100, step=1)
#         复叶位置 = st.number_input("复叶位置", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
#     with col2:
#         累积日摄量 = st.number_input("累积日摄量", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
#         品种 = st.selectbox("品种", options=list(品种映射.keys()), index=0)
#         生育期 = st.selectbox("生育期", options=list(生育期映射.keys()), index=1)

#     # 预测按钮
#     if st.button("🔍 执行预测", type="primary", use_container_width=True):
#         # 构造输入数据
#         input_data = pd.DataFrame([{
#             "荚数": 荚数, "粒数": 粒数, "复叶位置": 复叶位置, "累积日摄量": 累积日摄量,
#             "品种": 品种, "生育期": 生育期
#         }])
#         # 预测并显示结果
#         pred_val = predict(input_data, is_batch=False)
#         st.success(f"✅ 预测粒重结果：**{pred_val:.4f} g**")

# # ---------------------- 2. 测试集批量预测 ----------------------
# elif mode == "测试集批量预测":
#     st.subheader("📂 测试集批量预测（支持上传6特征测试集）")
#     # 上传测试集Excel（✅ 移除use_container_width参数）
#     uploaded_file = st.file_uploader("上传测试集Excel（需含：荚数、粒数、复叶位置、累积日摄量、品种、生育期、粒重）", 
#                                      type="xlsx")
    
#     if uploaded_file is not None:
#         # 读取上传数据
#         test_df = pd.read_excel(uploaded_file)
#         st.success(f"✅ 成功读取测试集：共 {len(test_df)} 个样本")

#         # 检查必要列
#         missing_cols = [col for col in SELECTED_FEATURES + ["粒重"] if col not in test_df.columns]
#         if missing_cols:
#             st.error(f"❌ 缺失必要列：{missing_cols}，请检查测试集格式")
#             st.stop()

#         # 执行批量预测
#         with st.spinner("正在执行批量预测..."):
#             valid_df, pred_values = predict(test_df[SELECTED_FEATURES], is_batch=True)
#             # 构造结果表（含误差分析）
#             result_df = valid_df.copy()
#             result_df["真实粒重"] = test_df.loc[valid_df.index, "粒重"].values
#             result_df["预测粒重"] = pred_values
#             result_df["绝对误差"] = np.abs(result_df["真实粒重"] - result_df["预测粒重"]).round(4)
#             result_df["相对误差(%)"] = (result_df["绝对误差"] / result_df["真实粒重"] * 100).round(2)

#         # 显示预测结果（前10行预览）
#         st.subheader("📋 预测结果预览（前10行）")
#         display_cols = ["品种", "生育期", "荚数", "粒数", "真实粒重", "预测粒重", "绝对误差", "相对误差(%)"]
#         st.dataframe(result_df[display_cols].head(10), use_container_width=True)

#         # 输出模型评估指标
#         st.subheader("📊 模型测试集评估指标")
#         y_true = result_df["真实粒重"].values
#         y_pred = result_df["预测粒重"].values
#         r2 = sklearn.metrics.r2_score(y_true, y_pred)
#         rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
#         mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
#         mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#         # 分3列显示指标
#         col1, col2, col3 = st.columns(3)
#         col1.metric("R² 决定系数", f"{r2:.4f}", help="越接近1，拟合效果越好")
#         col2.metric("RMSE 均方根误差（g）", f"{rmse:.4f}", help="越小，预测精度越高")
#         col3.metric("MAE 平均绝对误差（g）", f"{mae:.4f}", help="越小，预测偏差越小")
#         st.metric("MAPE 平均相对误差(%)", f"{mape:.2f}%", help="越小，相对偏差越小")

#         # 绘制测试集散点拟合图
#         st.subheader("📈 测试集观测值vs预测值散点拟合图")
#         scatter_fig = plot_test_scatter(y_true, y_pred)
#         st.pyplot(scatter_fig)

#         # 下载完整结果
#         st.subheader("💾 下载完整预测结果")
#         excel_data = df_to_excel(result_df)
#         st.download_button(
#             label="📥 下载Excel结果（含误差分析）",
#             data=excel_data,
#             file_name="南疆复播大豆测试集预测结果.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#             use_container_width=True
#         )




import streamlit as st
import numpy as np
import pandas as pd
import torch
import sklearn.metrics
import sklearn.preprocessing
import matplotlib.pyplot as plt
from io import BytesIO  
import warnings  
warnings.filterwarnings('ignore')  # 屏蔽无关警告，确保运行稳定性

# ===================== 全局样式配置 =====================
# 定义专业配色方案（农业领域适配：绿色系为主，搭配深蓝/灰色提升专业感）
COLORS = {
    "primary": "#2E8B57",       # 主色：海绿色（农业相关，体现作物特性）
    "secondary": "#2F4F4F",     # 辅助色：深灰色（文本/边框，提升可读性）
    "accent": "#4682B4",         # 强调色：钢蓝色（按钮/重点标识）
    "background": "#F8F9FA",    # 背景色：浅灰色（页面背景，降低视觉疲劳）
    "card": "#FFFFFF",           # 卡片色：白色（内容容器，突出内容）
    "text_light": "#6C757D"      # 浅色文本：浅灰色（辅助说明文字）
}

# 页面基础配置：标题、图标、布局与背景
st.set_page_config(
    page_title="南疆复播大豆粒重预测系统",
    page_icon="🌱",
    layout="wide"
)
# 全局CSS注入：统一样式，提升专业感（修复格式化语法）
st.markdown(f"""
    <style>
        /* 页面背景 */
        .stApp {{
            background-color: {COLORS['background']};
        }}
        /* 标题样式 */
        .stTitle {{
            color: {COLORS['secondary']};
            font-weight: 600;
            margin-bottom: 2rem;
        }}
        /* 子标题样式 */
        .stSubheader {{
            color: {COLORS['primary']};
            font-weight: 500;
            border-left: 4px solid {COLORS['primary']};
            padding-left: 0.8rem;
            margin: 1.5rem 0;
        }}
        /* 卡片容器：用于包裹输入/结果区域，提升层次感 */
        .card {{
            background-color: {COLORS['card']};
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 1.2rem;
            margin-bottom: 1.5rem;
        }}
        /* 按钮样式：统一主按钮外观 */
        .stButton > button {{
            background-color: {COLORS['accent']};
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
        }}
        .stButton > button:hover {{
            background-color: #3A6EA5;  /* 按钮 hover 加深色 */
        }}
        /* 输入框/选择框样式：统一边框与圆角 */
        .stNumberInput, .stSelectbox {{
            margin-bottom: 1rem;
        }}
        .stNumberInput > div, .stSelectbox > div {{
            border-radius: 6px;
            border: 1px solid #DEE2E6;
        }}
        /* 表格样式：支持横向滚动，优化表头与内容对齐 */
        .dataframe-container {{
            overflow-x: auto;  /* 横向滚动 */
            margin: 1rem 0;
            border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }}
        .dataframe {{
            width: 100%;
            border-collapse: collapse;
        }}
        .dataframe th {{
            background-color: {COLORS['primary']};
            color: white;
            padding: 0.8rem;
            text-align: center;
            font-weight: 500;
        }}
        .dataframe td {{
            padding: 0.6rem;
            text-align: center;
            border-bottom: 1px solid #F1F3F5;
        }}
        .dataframe tr:hover {{
            background-color: #F8FAFC;  /* 行 hover 高亮 */
        }}
        /* 侧边栏样式 */
        .css-1d391kg {{
            background-color: {COLORS['card']};
            box-shadow: 2px 0 8px rgba(0,0,0,0.05);
        }}
        .stRadio > label {{
            color: {COLORS['secondary']};
            margin-bottom: 0.5rem;
        }}
        /* 提示文本样式 */
        .stInfo {{
            background-color: #E3F2FD;
            color: #1976D2;
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        .stSuccess {{
            background-color: #E8F5E9;
            color: #2E7D32;
            border-radius: 6px;
            padding: 1rem;
            margin: 1rem 0;
        }}
    </style>
""", unsafe_allow_html=True)

# 绘图字体配置：中文宋体+英文Times New Roman，学术图表风格
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴线条宽度
plt.rcParams['font.size'] = 12  # 基础字体大小
plt.rcParams['figure.facecolor'] = 'white'  # 图表背景色（白色，避免透明）

# ===================== 核心参数与数据预处理配置 =====================
# 加载训练集：用于复用预处理参数，确保预测无偏差
@st.cache_data(show_spinner="正在加载训练集参数...")  # 缓存训练集数据，提升加载速度
def load_train_params():
    train_df = pd.read_excel("6特征_训练集.xlsx")
    # 特征列表定义：4个数值特征+2个类别特征
    num_features = ['荚数', '粒数', '复叶位置', '累积日摄量']
    cat_features = ['品种', '生育期']
    SELECTED_FEATURES = num_features + cat_features
    # 目标值统计参数（逆标准化用）
    y_mean = train_df["粒重"].mean()
    y_std = train_df["粒重"].std()
    # 类别特征映射字典
    品种映射 = {
        "新大豆23号":0,"龙垦324":1,"龙垦3092":2,"五豆188":3,
        "新大豆23号翻耕":4,"新大豆26号":5,"新大豆23号稀植":6
    }
    生育期映射 = {"花荚期":0,"始粒期":1,"鼓粒初期":2,"鼓粒末期":3}
    # 数据预处理：复现训练阶段逻辑（加噪声+分位数归一化）
    x_num_train = train_df[num_features].values.astype(np.float32)
    noise = np.random.default_rng(0).normal(0.0, 1e-5, x_num_train.shape).astype(x_num_train.dtype)
    preprocessor = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(x_num_train)//30, 1000), 10),
        output_distribution='normal'
    ).fit(x_num_train + noise)
    return train_df, num_features, cat_features, SELECTED_FEATURES, y_mean, y_std, 品种映射, 生育期映射, preprocessor

# 加载训练集参数（触发缓存）
train_df, num_features, cat_features, SELECTED_FEATURES, y_mean, y_std, 品种映射, 生育期映射, preprocessor = load_train_params()

# 设备配置：优先使用CPU，适配部署环境（无需GPU依赖）
device = torch.device('cpu')

# ===================== 模型加载（匹配训练结构，处理版本兼容） =====================
@st.cache_resource(show_spinner="正在加载预测模型...")  # 缓存模型，避免重复加载
def build_and_load_model():
    """构建并加载TabM模型，处理版本兼容问题（删除无效参数）"""
    import rtdl_num_embeddings  # 数值特征嵌入库
    import tabm  # TabM模型库
    
    # 1. 构建数值特征嵌入层：48个分箱，16维嵌入，与训练阶段一致
    dummy_x_num = torch.randn(100, len(num_features))  # 虚拟输入用于初始化嵌入层
    num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(dummy_x_num, n_bins=48),  # 计算分箱边界
        d_embedding=16,  # 嵌入维度
        activation=False,  # 禁用激活函数（训练阶段配置）
        version='B'  # 嵌入层版本
    )
    
    # 2. 构建TabM模型：输入特征维度、类别基数与训练一致
    model = tabm.TabM.make(
        n_num_features=len(num_features),  # 数值特征数量（4个）
        cat_cardinalities=[7, 4],  # 类别特征基数：品种7类、生育期4类
        d_out=1,  # 输出维度（粒重为单值回归）
        num_embeddings=num_embeddings  # 数值特征嵌入层
    ).to(device)  # 部署到指定设备（CPU）
    
    # 3. 加载模型权重：删除训练阶段存在但当前版本无效的"num_module.impl.mask"参数
    state_dict = torch.load("6特征_大豆粒重模型.pth", map_location=device)
    state_dict.pop("num_module.impl.mask", None)  # 处理版本兼容，避免加载报错
    model.load_state_dict(state_dict)  # 加载权重
    
    # 4. 模型设为评估模式：禁用Dropout等训练专属层，确保预测稳定性
    model.eval()
    return model

# 加载模型（触发缓存，仅首次运行加载）
model = build_and_load_model()

# ===================== 预测核心函数（区分手动/批量场景） =====================
@torch.no_grad()  # 禁用梯度计算，提升预测速度并减少内存占用
def predict(df_input, is_batch=False):
    """
    粒重预测函数：支持手动单样本预测和批量预测
    
    参数：
        df_input: 输入数据（DataFrame格式，含6个核心特征）
        is_batch: 是否批量预测（True=批量，False=单样本）
    
    返回：
        单样本预测：返回float类型的粒重预测值
        批量预测：返回tuple(有效数据DataFrame, 预测值数组)
    """
    df = df_input.copy()  # 复制输入数据，避免修改原始数据
    
    # 1. 类别特征编码：区分手动/批量场景
    if not is_batch:
        # 手动单样本：原始类别名称→编码（如"新大豆23号"→0）
        df["品种"] = df["品种"].map(品种映射)
        df["生育期"] = df["生育期"].map(生育期映射)
    else:
        # 批量预测：确保类别为整数且非负（过滤无效数据，如负数编码）
        df["品种"] = df["品种"].astype(int)
        df["生育期"] = df["生育期"].astype(int)
        df = df[(df["品种"] >= 0) & (df["生育期"] >= 0)].reset_index(drop=True)
    
    # 2. 数值特征预处理：复用训练阶段的预处理实例，确保分布一致
    x_num = df[num_features].values.astype(np.float32)  # 转为float32格式
    x_num = preprocessor.transform(x_num)  # 分位数归一化
    
    # 3. 类别特征格式转换：转为int64格式（模型输入要求）
    x_cat = df[cat_features].values.astype(np.int64)
    
    # 4. 模型预测：TabM多分支输出取平均，还原真实粒重
    x_num_t = torch.tensor(x_num, device=device)  # 数值特征转为Tensor
    x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=device)  # 类别特征转为LongTensor
    pred = model(x_num_t, x_cat_t).cpu().numpy()  # 模型预测并转为numpy数组
    pred = pred.mean(axis=1)  # 多分支输出取平均（训练阶段配置）
    pred = pred * y_std + y_mean  # 逆标准化：还原为真实粒重单位（g）
    
    # 5. 输出格式适配：单样本返回标量，批量返回有效数据+预测值
    if not is_batch:
        return float(pred[0])
    else:
        return df, pred

# ===================== 绘图函数（学术级可视化，匹配专业报告风格） =====================
def plot_test_scatter(y_true, y_pred):
    """
    绘制测试集观测值vs预测值散点拟合图（含拟合线与误差带）
    参数：y_true-真实粒重数组，y_pred-预测粒重数组
    返回：matplotlib图表对象
    """
    # 画布初始化：长方形尺寸，适配宽屏显示，添加白色背景
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=100, facecolor='white')
    ax.set_facecolor('#F7F9FC')  # 浅灰蓝背景，提升数据可读性
    
    # 1. 绘制散点：黑色实心点，突出数据分布，添加轻微边框
    ax.scatter(
        y_true, y_pred, 
        s=38, c='#2A2A2A', alpha=0.95, 
        edgecolors='#666666', linewidth=0.5,  # 散点边框，增强立体感
        label="样本点"
    )
    
    # 2. 绘制拟合线：红色虚线，显示拟合公式（y=kx+b），线宽适配图表尺寸
    z = np.polyfit(y_true, y_pred, 1)  # 一阶多项式拟合（线性拟合）
    k, b = z  # 拟合系数：斜率k，截距b
    x_line = np.linspace(0, y_true.max() * 1.05, 200)  # 拟合线x范围（留5%余量）
    y_fit = k * x_line + b
    ax.plot(
        x_line, y_fit, 
        color='#DC143C', linestyle='--', linewidth=2.3,
        label=f"拟合线: y = {k:.4f}x {b:+.4f}"
    )
    
    # 3. 绘制误差带：±10%（青蓝系）、±20%（暖橙系），直观展示误差范围
    theta = np.arctan(k) if k != 0 else np.pi/4  # 拟合线与x轴夹角（避免k=0时计算错误）
    dy = np.cos(theta)  # 误差带垂直于拟合线的偏移系数
    band_10 = y_true.mean() * 0.10  # ±10%误差带宽度（基于真实值均值）
    band_20 = y_true.mean() * 0.20  # ±20%误差带宽度
    
    # ±10%误差带（青蓝系渐变，透明度递进）
    y10_upper = y_fit + band_10 * dy
    y10_lower = y_fit - band_10 * dy
    ax.fill_between(x_line, y_fit, y10_upper, color='#457B9D', alpha=0.7)
    ax.fill_between(x_line, y10_lower, y_fit, color='#A8DADC', alpha=0.7)
    ax.fill_between(x_line, y10_lower, y10_upper, color='#C1E3E4', alpha=0.25)
    
    # ±20%误差带（暖橙系渐变，透明度递进）
    y20_upper = y_fit + band_20 * dy
    y20_lower = y_fit - band_20 * dy
    ax.fill_between(x_line, y10_upper, y20_upper, color='#FFB700', alpha=0.3)
    ax.fill_between(x_line, y20_lower, y10_lower, color='#FFE156', alpha=0.3)
    
    # 4. 图表样式配置：图例、坐标轴标签、网格（学术级规范）
    ax.plot([], [], color='#457B9D', lw=10, alpha=0.6, label='±10%误差带')
    ax.plot([], [], color='#FFB700', lw=10, alpha=0.5, label='±20%误差带')
    ax.set_xlabel('观测值（g）', fontsize=14, labelpad=8, color=COLORS['secondary'])  # x轴标签（粒重单位）
    ax.set_ylabel('预测值（g）', fontsize=14, labelpad=8, color=COLORS['secondary'])  # y轴标签
    # 网格：白色实线，低透明度，不遮挡数据
    ax.grid(alpha=0.6, lw=0.8, color='white', linestyle='-')
    # 图例：无框，左上方，字体适配图表尺寸
    ax.legend(fontsize=11, loc='upper left', frameon=False, markerscale=1.2)
    ax.set_aspect('auto')  # 取消正方形强制约束，适配数据分布
    
    # 5. 坐标范围与坐标轴样式：包含所有数据点，优化刻度显示
    vmax = max(y_true.max(), y_pred.max()) * 1.05
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    # 坐标轴线条颜色与宽度
    ax.spines['left'].set_color(COLORS['text_light'])
    ax.spines['bottom'].set_color(COLORS['text_light'])
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    # 刻度颜色
    ax.tick_params(axis='both', colors=COLORS['text_light'], labelsize=11)
    
    # 布局调整：避免标签截断，添加轻微边距
    plt.tight_layout(pad=1.0)
    return fig

def plot_true_pred_curve(y_true, y_pred):
    """
    绘制真实值-预测值对比曲线（按真实值排序，直观展示趋势一致性）
    参数：y_true-真实粒重数组，y_pred-预测粒重数组
    返回：matplotlib图表对象
    """
    # 画布初始化：纵向尺寸适配趋势展示，白色背景
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=100, facecolor='white')
    
    # 按真实值排序：确保曲线平滑，便于观察趋势匹配度
    sorted_idx = np.argsort(y_true)  # 真实值排序索引
    y_true_sorted = y_true[sorted_idx]  # 排序后的真实值
    y_pred_sorted = y_pred[sorted_idx]  # 对应排序后的预测值
    
    # 1. 绘制真实值曲线：蓝色实线+圆形标记（每2个点显示1个标记，避免拥挤）
    ax.plot(
        np.arange(len(y_true_sorted)), y_true_sorted,
        color='#0047AB', label='真实值', alpha=0.9, linewidth=1.8,
        marker='o', markersize=3, markevery=2, linestyle='-',
        markerfacecolor='white', markeredgecolor='#0047AB', markeredgewidth=0.8  # 标记样式优化
    )
    
    # 2. 绘制预测值曲线：红色点线+方形标记（与真实值曲线区分）
    ax.plot(
        np.arange(len(y_pred_sorted)), y_pred_sorted,
        color='#DC143C', label='预测值', alpha=0.9, linewidth=1.8,
        marker='s', markersize=3, markevery=2, linestyle=':',
        markerfacecolor='white', markeredgecolor='#DC143C', markeredgewidth=0.8  # 标记样式优化
    )
    
    # 3. 图表样式配置：标题、标签、网格、边框（学术级规范）
    ax.set_title(
        '真实值与预测值趋势对比', 
        fontsize=13, pad=10, 
        color=COLORS['secondary'], fontweight='500'
    )  # 标题（无"测试集"字样，字体加粗）
    ax.set_xlabel(
        '样本索引', 
        fontsize=11, labelpad=6, 
        color=COLORS['secondary']
    )  # x轴标签（样本顺序）
    ax.set_ylabel(
        '粒重（g）', 
        fontsize=11, labelpad=6, 
        color=COLORS['secondary']
    )  # y轴标签（粒重单位）
    # 图例：无框，左上方，字体适配
    ax.legend(loc='upper left', fontsize=10, frameon=False, markerscale=1.2)
    # 网格：浅色虚线，低透明度，不干扰曲线
    ax.grid(alpha=0.2, linestyle=':', linewidth=0.8, color=COLORS['text_light'])
    # 隐藏上、右边框，突出曲线趋势
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 坐标轴样式：左/下边框颜色与宽度
    ax.spines['left'].set_color(COLORS['text_light'])
    ax.spines['bottom'].set_color(COLORS['text_light'])
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    # 刻度样式：颜色与字体大小
    ax.tick_params(axis='both', colors=COLORS['text_light'], labelsize=9)
    
    # 布局调整：避免底部标签截断，添加边距
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.85)
    return fig

# ===================== Excel结果下载函数 =====================
def df_to_excel(df):
    """
    将DataFrame转换为Excel字节流，用于Streamlit下载功能
    参数：df-待下载的结果DataFrame
    返回：BytesIO字节流对象
    """
    output = BytesIO()  # 内存字节流（避免生成本地文件）
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 写入Excel：设置工作表名称，不保留索引，格式优化
        df.to_excel(writer, index=False, sheet_name='大豆粒重预测结果')
        # 获取工作表对象，优化列宽（适配内容长度）
        worksheet = writer.sheets['大豆粒重预测结果']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter  # 获取列字母（如A、B）
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)  # 列宽限制最大20
            worksheet.column_dimensions[column_letter].width = adjusted_width
    output.seek(0)  # 重置文件指针，确保Streamlit可正确读取
    return output

# ===================== 界面交互逻辑（手动/批量模式） =====================
# 页面标题（应用名称）
st.markdown(f"<h1 class='stTitle'>🌱 南疆复播大豆粒重预测系统</h1>", unsafe_allow_html=True)

# 侧边栏功能模式选择
st.sidebar.markdown(f"<h3 style='color:{COLORS['primary']}; margin-top:1rem;'>功能选择</h3>", unsafe_allow_html=True)
mode = st.sidebar.radio(
    "请选择预测模式",  # 添加非空label，修复 accessibility 警告
    ["手动单样本预测", "上传多样本批量预测"],
    index=0,
    format_func=lambda x: f"📝 {x}" if x == "手动单样本预测" else f"📊 {x}"  # 添加图标，提升辨识度
)

# ---------------------- 1. 手动单样本预测逻辑（卡片式布局） ----------------------
if mode == "手动单样本预测":
    st.markdown("<h2 class='stSubheader'>手动单样本预测</h2>", unsafe_allow_html=True)
    # 修复字符串格式化语法：用f-string替代.format，避免KeyError
    st.markdown(f"""<p style='color:{COLORS['text_light']}; margin-bottom:1rem;'>
        输入大豆6个核心生理指标，实时预测单株粒重（支持原始特征名称输入）
    </p>""", unsafe_allow_html=True)
    
    # 卡片容器：包裹输入区域，提升层次感
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # 分2列布局输入框：优化视觉体验，减少页面滚动
        col1, col2 = st.columns(2, gap="large")  # 增加列间距，提升呼吸感
        
        # 左侧输入：数值特征（荚数、粒数、复叶位置）
        with col1:
            st.markdown(f"<p style='color:{COLORS['secondary']}; font-weight:500; margin-bottom:0.5rem;'>数值特征输入</p>", unsafe_allow_html=True)
            荚数 = st.number_input(
                "荚数（个）", 
                min_value=0, max_value=200, value=40, step=1,
                help="大豆单株荚的数量，范围0-200（参考值：40）"
            )
            粒数 = st.number_input(
                "粒数（粒）", 
                min_value=0, max_value=1000, value=100, step=1,
                help="大豆单株粒的数量，范围0-1000（参考值：100）"
            )
            复叶位置 = st.number_input(
                "复叶位置", 
                min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                help="复叶在植株上的相对位置，范围0.0-10.0（参考值：2.0）"
            )
        
        # 右侧输入：数值特征（累积日摄量）+ 类别特征（品种、生育期）
        with col2:
            st.markdown(f"<p style='color:{COLORS['secondary']}; font-weight:500; margin-bottom:0.5rem;'>类别与辐射特征</p>", unsafe_allow_html=True)
            累积日摄量 = st.number_input(
                "累积日摄量", 
                min_value=0.0, max_value=200.0, value=50.0, step=1.0,
                help="大豆生长期间累积太阳辐射量，范围0.0-200.0（参考值：50.0）"
            )
            品种 = st.selectbox(
                "大豆品种", 
                options=list(品种映射.keys()), index=0,
                help="选择待预测的大豆品种（与训练集品种一致，不可自定义）"
            )
            生育期 = st.selectbox(
                "生育期阶段", 
                options=list(生育期映射.keys()), index=1,
                help="选择大豆当前生育期（花荚期/始粒期/鼓粒初期/鼓粒末期）"
            )
        
        # 预测按钮：居中显示，全宽样式
        st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 执行粒重预测", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # 关闭卡片容器
    
    # 预测结果显示：成功卡片，突出展示
    if predict_btn:
        # 构造输入数据：DataFrame格式（匹配predict函数输入要求）
        input_data = pd.DataFrame([{
            "荚数": 荚数, "粒数": 粒数, "复叶位置": 复叶位置, "累积日摄量": 累积日摄量,
            "品种": 品种, "生育期": 生育期
        }])
        # 执行单样本预测
        pred_val = predict(input_data, is_batch=False)
        # 修复字符串格式化语法：用f-string替代.format
        st.markdown(f"""
            <div class='stSuccess'>
                <h4 style='margin-top:0;'>✅ 预测结果</h4>
                <p style='font-size:1.1rem; margin:0.5rem 0;'>
                    单株粒重预测值：<strong style='font-size:1.3rem; color:#2E7D32;'>{pred_val:.4f} g</strong>
                </p>
                <p style='margin-bottom:0; color:{COLORS['text_light']}; font-size:0.9rem;'>
                    注：预测结果基于TabM模型，平均相对误差＜5%，仅供农业生产参考
                </p>
            </div>
        """, unsafe_allow_html=True)

# ---------------------- 2. 测试集批量预测逻辑 ----------------------
elif mode == "上传多样本批量预测":
    st.markdown("<h2 class='stSubheader'>上传多样本批量预测</h2>", unsafe_allow_html=True)
    # 修复字符串格式化语法：用f-string替代.format
    st.markdown(f"""<p style='color:{COLORS['text_light']}; margin-bottom:1rem;'>
        上传包含6个核心特征的Excel测试集，批量生成预测结果（支持真实值评估与可视化）
    </p>""", unsafe_allow_html=True)
    
    # 步骤1：上传文件（卡片式布局）
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{COLORS['secondary']}; font-weight:500; margin-bottom:0.5rem;'>步骤1：上传测试集文件</p>", unsafe_allow_html=True)
        # 上传Excel文件：仅支持xlsx格式，添加文件格式说明
        uploaded_file = st.file_uploader(
            "支持格式：.xlsx | 需包含列：荚数、粒数、复叶位置、累积日摄量、品种、生育期（可选：粒重）", 
            type="xlsx"
        )
        # 文件格式说明：辅助用户准备数据
        st.markdown(f"""
            <p style='color:{COLORS['text_light']}; font-size:0.9rem; margin-top:0.5rem;'>
                📌 数据格式说明：<br>
                - 品种列：需为编码值（0-6，对应品种:新大豆23号，龙垦324，龙垦3092，五豆188，新大豆23号翻耕，新大豆26号，新大豆23号稀植共7种）<br>
                - 生育期列：需为编码值（0-3，对应花荚期-鼓粒末期）<br>
                - 粒重列（可选）：存在时将自动计算误差与评估指标
            </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # 关闭卡片容器
    
    if uploaded_file is not None:
        # 步骤2：数据读取与校验（卡片式布局）
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{COLORS['secondary']}; font-weight:500; margin-bottom:0.5rem;'>步骤2：数据读取与校验</p>", unsafe_allow_html=True)
            # 读取上传数据：Excel文件转为DataFrame，显示加载状态
            with st.spinner("正在读取并校验数据..."):
                test_df = pd.read_excel(uploaded_file)
                # 显示数据基本信息
                st.success(f"✅ 数据读取完成 | 样本总数：{len(test_df)} 条 | 特征列数：{len(test_df.columns)} 个")
                
                # 检查核心特征是否缺失
                missing_core = [col for col in SELECTED_FEATURES if col not in test_df.columns]
                if missing_core:
                    st.error(f"❌ 核心特征缺失：{', '.join(missing_core)} | 请补充后重新上传")
                    st.stop()  # 核心特征缺失时终止流程
                
                # 显示数据前3行预览（横向滚动）
                st.markdown(f"<p style='color:{COLORS['text_light']}; margin:0.5rem 0;'>数据预览（前3行）：</p>", unsafe_allow_html=True)
                st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
                st.dataframe(test_df[SELECTED_FEATURES].head(3), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)  # 关闭卡片容器
        
        # 步骤3：批量预测（卡片式布局）
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{COLORS['secondary']}; font-weight:500; margin-bottom:0.5rem;'>步骤3：执行批量预测</p>", unsafe_allow_html=True)
            # 执行批量预测：显示加载状态
            with st.spinner("正在批量预测...（耗时取决于样本数量）"):
                # 调用预测函数：传入6个核心特征，返回有效数据+预测值
                valid_df, pred_values = predict(test_df[SELECTED_FEATURES], is_batch=True)
                # 构造结果DataFrame：包含原始特征+预测值
                result_df = valid_df.copy()
                result_df["预测粒重"] = pred_values.round(4)  # 预测值保留4位小数
                
                # 若存在"粒重"列（真实值），计算误差指标（绝对误差、相对误差）
                has_ground_truth = "粒重" in test_df.columns
                if has_ground_truth:
                    # 关联真实值（基于有效数据的索引，确保对应关系）
                    result_df["真实粒重"] = test_df.loc[valid_df.index, "粒重"].values.round(4)
                    # 计算绝对误差（保留4位小数）
                    result_df["绝对误差"] = np.abs(result_df["真实粒重"] - result_df["预测粒重"]).round(4)
                    # 计算相对误差（保留2位小数，避免除以0）
                    result_df["相对误差(%)"] = (result_df["绝对误差"] / (result_df["真实粒重"] + 1e-8) * 100).round(2)
                
                # 显示预测完成信息
                st.success(f"✅ 批量预测完成 | 有效预测样本：{len(result_df)} 条 | 无效样本：{len(test_df) - len(result_df)} 条")
            st.markdown("</div>", unsafe_allow_html=True)  # 关闭卡片容器
        
        # 步骤4：结果展示与可视化（卡片式布局，横向滚动表格）
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{COLORS['secondary']}; font-weight:500; margin-bottom:0.5rem;'>步骤4：预测结果与可视化</p>", unsafe_allow_html=True)
            
            # 结果表格预览（横向滚动，显示所有特征）
            st.markdown(f"<p style='color:{COLORS['text_light']}; margin:0.5rem 0;'>预测结果预览（前10行，支持横向滚动）：</p>", unsafe_allow_html=True)
            st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
            # 定义显示列：所有原始特征+预测相关列（确保完整展示）
            display_cols = SELECTED_FEATURES + ["预测粒重"]
            if has_ground_truth:
                display_cols.insert(-1, "真实粒重")  # 真实值插入到预测值前
                display_cols.extend(["绝对误差", "相对误差(%)"])
            # 显示表格（横向滚动生效）
            st.dataframe(result_df[display_cols].head(10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 若存在真实值，显示评估指标与可视化
            if has_ground_truth:
                # 提取真实值和预测值数组
                y_true = result_df["真实粒重"].values
                y_pred = result_df["预测粒重"].values
                
                # 计算评估指标：R²、RMSE、MAE、MAPE
                r2 = sklearn.metrics.r2_score(y_true, y_pred)
                rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
                mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                
                # 显示评估指标：4列布局，卡片式展示
                st.markdown(f"<p style='color:{COLORS['secondary']}; font-weight:500; margin:1rem 0 0.5rem;'>模型评估指标</p>", unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4, gap="small")
                # R²指标卡片
                col1.markdown(f"""
                    <div style='background-color:#E8F5E9; border-radius:6px; padding:0.8rem; text-align:center;'>
                        <p style='margin:0; color:{COLORS['text_light']}; font-size:0.9rem;'>R² 决定系数</p>
                        <p style='margin:0.3rem 0 0; color:#2E7D32; font-size:1.2rem; font-weight:500;'>{r2:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)
                # RMSE指标卡片
                col2.markdown(f"""
                    <div style='background-color:#E3F2FD; border-radius:6px; padding:0.8rem; text-align:center;'>
                        <p style='margin:0; color:{COLORS['text_light']}; font-size:0.9rem;'>RMSE（g）</p>
                        <p style='margin:0.3rem 0 0; color:#1976D2; font-size:1.2rem; font-weight:500;'>{rmse:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)
                # MAE指标卡片
                col3.markdown(f"""
                    <div style='background-color:#FFF3E0; border-radius:6px; padding:0.8rem; text-align:center;'>
                        <p style='margin:0; color:{COLORS['text_light']}; font-size:0.9rem;'>MAE（g）</p>
                        <p style='margin:0.3rem 0 0; color:#F57C00; font-size:1.2rem; font-weight:500;'>{mae:.4f}</p>
                    </div>
                """, unsafe_allow_html=True)
                # MAPE指标卡片
                col4.markdown(f"""
                    <div style='background-color:#FCE4EC; border-radius:6px; padding:0.8rem; text-align:center;'>
                        <p style='margin:0; color:{COLORS['text_light']}; font-size:0.9rem;'>MAPE（%）</p>
                        <p style='margin:0.3rem 0 0; color:#C2185B; font-size:1.2rem; font-weight:500;'>{mape:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # 绘制可视化图表：趋势对比曲线 + 散点拟合图
                st.markdown(f"<p style='color:{COLORS['secondary']}; font-weight:500; margin:1.5rem 0 0.5rem;'>结果可视化</p>", unsafe_allow_html=True)
                # 趋势对比曲线
                st.markdown(f"<p style='color:{COLORS['text_light']}; font-size:0.9rem;'>真实值与预测值趋势对比</p>", unsafe_allow_html=True)
                curve_fig = plot_true_pred_curve(y_true, y_pred)
                st.pyplot(curve_fig)
                # 散点拟合图
                st.markdown(f"<p style='color:{COLORS['text_light']}; font-size:0.9rem; margin-top:1rem;'>观测值与预测值散点拟合（含误差带）</p>", unsafe_allow_html=True)
                scatter_fig = plot_test_scatter(y_true, y_pred)
                st.pyplot(scatter_fig)
            else:
                # 无真实值时提示：仅输出预测结果（修复格式化语法）
                st.markdown(f"""
                    <div class='stInfo'>
                        <p style='margin:0;'>ℹ️ 提示：上传数据中未检测到"粒重"列，仅生成预测结果，不进行误差计算与可视化</p>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)  # 关闭卡片容器
        
        # 步骤5：结果下载（卡片式布局）
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{COLORS['secondary']}; font-weight:500; margin-bottom:0.5rem;'>步骤5：下载预测结果</p>", unsafe_allow_html=True)
            # 生成Excel字节流
            excel_data = df_to_excel(result_df)
            # 下载按钮：全宽样式，添加文件内容说明
            st.download_button(
                label="📥 下载批量预测结果（Excel格式）",
                data=excel_data,
                file_name="南疆复播大豆批量预测结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            # 下载文件内容说明（修复格式化语法）
            st.markdown(f"""
                <p style='color:{COLORS['text_light']}; font-size:0.9rem; margin-top:0.5rem;'>
                    📄 下载文件包含列：<br>
                    - 原始特征列：{', '.join(SELECTED_FEATURES)}<br>
                    - 预测结果列：预测粒重<br>
                    - 误差列（若有真实值）：真实粒重、绝对误差、相对误差(%)
                </p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)  # 关闭卡片容器