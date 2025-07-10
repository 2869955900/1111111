import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sksurv.ensemble import RandomSurvivalForest

# 加载训练好的模型
model = joblib.load('rsf.pkl')

# 默认值
default_values = {
    "SDMA-ADMA_pos-140": 0.001121,
    "Thymine_pos-150": 0.000069,
    "Phosphocreatine_neg-067": 0.000018,
    "Proline_pos-132": 0.111990,
    "Glycerophosphorylcholine_pos-080": 0.006630,
    "Guanidineacetic_acid_pos-087": 0.000432  # 使用下划线替代空格
}

# 创建函数，用于预测风险评分以及绘制累计风险函数和生存函数
def predict_risk(SDMA_ADMA_pos_140, Thymine_pos_150, Phosphocreatine_neg_067, 
                 Proline_pos_132, Glycerophosphorylcholine_pos_080, Guanidineacetic_acid_pos_087):
    
    # 使用准确的列名，确保与训练时一致
    correct_columns = ["SDMA-ADMA_pos-140", "Thymine_pos-150", "Phosphocreatine_neg-067",
                       "Proline_pos-132", "Glycerophosphorylcholine_pos-080", "Guanidineacetic_acid_pos-087"]  # 保证一致性

    # 创建一个 DataFrame，确保列名与训练时一致
    input_data = pd.DataFrame([[SDMA_ADMA_pos_140, Thymine_pos_150, Phosphocreatine_neg_067,
                                Proline_pos_132, Glycerophosphorylcholine_pos_080, Guanidineacetic_acid_pos_087]], 
                              columns=correct_columns)

    # 使用模型进行预测，获取风险评分
    risk_score = model.predict(input_data)[0]
    
    # 预测累计风险
    cumulative_hazard = model.predict_cumulative_hazard(input_data)

    # 绘制累计风险函数
    fig, ax = plt.subplots()
    for i in range(cumulative_hazard.shape[0]):
        ax.step(cumulative_hazard.columns, cumulative_hazard.iloc[i], where="post", label=f"Sample {i+1}")
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Hazard')
    ax.set_title('Cumulative Hazard Function')
    ax.legend()
    cumulative_hazard_path = "cumulative_hazard.png"
    plt.savefig(cumulative_hazard_path)
    plt.close()

    # 绘制生存函数
    survival_function = model.predict_survival_function(input_data)
    fig2, ax2 = plt.subplots()
    for i in range(survival_function.shape[0]):
        ax2.step(survival_function.columns, survival_function.iloc[i], where="post", label=f"Sample {i+1}")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Survival Function')
    ax2.set_title('Survival Function')
    ax2.legend()
    survival_function_path = "survival_function.png"
    plt.savefig(survival_function_path)
    plt.close()

    return risk_score, cumulative_hazard_path, survival_function_path


# 设置 Streamlit 页面
st.title("Risk Prediction for Survival Analysis")
st.write("Enter the feature values to predict the risk score and view the survival analysis plots.")

# 创建输入框，用户可以输入特征值
SDMA_ADMA_pos_140 = st.number_input("SDMA-ADMA_pos-140", value=default_values["SDMA-ADMA_pos-140"])
Thymine_pos_150 = st.number_input("Thymine_pos-150", value=default_values["Thymine_pos-150"])
Phosphocreatine_neg_067 = st.number_input("Phosphocreatine_neg-067", value=default_values["Phosphocreatine_neg-067"])
Proline_pos_132 = st.number_input("Proline_pos-132", value=default_values["Proline_pos-132"])
Glycerophosphorylcholine_pos_080 = st.number_input("Glycerophosphorylcholine_pos-080", value=default_values["Glycerophosphorylcholine_pos-080"])
Guanidineacetic_acid_pos_087 = st.number_input("Guanidineacetic_acid_pos-087", value=default_values["Guanidineacetic_acid_pos-087"])

# 当用户点击 "Submit" 按钮时，进行预测并显示结果
if st.button("Submit"):
    risk_score, cumulative_hazard_path, survival_function_path = predict_risk(
        SDMA_ADMA_pos_140, Thymine_pos_150, Phosphocreatine_neg_067,
        Proline_pos_132, Glycerophosph
