import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle

#  وظيفة لتنسيق التقرير النهائي
def generate_report(input_df, prob, high_risk, low_risk):
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #007bff;'>📄 التقرير الطبي </h2>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**تاريخ الفحص:** {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        st.write(f"**حالة المريض:** {'⚠️ احتمالية إصابة مرتفعة' if prob > 0.5 else '✅ حالة مستقرة'}")
    with col_b:
        st.write(f"**درجة اليقين (Confidence):** {prob:.1%}")

    st.markdown("### 📋 ملخص البيانات السريرية")
    st.table(input_df.T.rename(columns={0: 'القيمة'})) # عرض البيانات بشكل رأسي أنيق

    st.markdown("### التوصيف الفني للقرار (AI Insights)")
    st.write("بناءً على نموذج **XGBoost** وتحليل **SHAP**، تم تحديد المؤشرات التالية كأكثر العوامل تأثيراً:")
    
    rep_c1, rep_c2 = st.columns(2)
    with rep_c1:
        st.info("**عوامل الخطر الرئيسية:**")
        for _, row in high_risk.iterrows():
            st.write(f"- {translate.get(row['Feature'])} (+{row['Impact']:.2f})")
    with rep_c2:
        st.success("**عوامل الحماية المكتشفة:**")
        for _, row in low_risk.iterrows():
            st.write(f"- {translate.get(row['Feature'])} ({row['Impact']:.2f})")

    st.warning("**إخلاء مسؤولية:** هذا التقرير هو نتاج خوارزميات ذكاء اصطناعي لدعم القرار السريري، ويجب مراجعته واعتماده من قبل طبيب القلب المختص قبل اتخاذ أي إجراء علاجي.")

# ......



st.set_page_config(
    page_title="Explainable Heart Disease | AI Diagnosis",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .risk-box { padding: 20px; border-radius: 10px; background-color: #fff5f5; border-right: 5px solid #ff4b4b; margin-bottom: 10px; }
    .safe-box { padding: 20px; border-radius: 10px; background-color: #f0fff4; border-right: 5px solid #28a745; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ لم يتم العثور على ملف 'model.pkl'. تأكد من وجوده في نفس المجلد.")
        return None

model = load_model()

st.sidebar.image("images/aya.jpg", width=150) 
st.sidebar.title("إدخال بيانات المريض")
st.sidebar.markdown("---")

def get_user_inputs():
    age = st.sidebar.number_input("العمر (age)", 1, 100, 50)
    sex = st.sidebar.radio("الجنس (sex)", [0, 1], format_func=lambda x: "ذكر" if x == 1 else "أنثى")
    
    st.sidebar.markdown("**المؤشرات الحيوية**")
    cp = st.sidebar.selectbox("نوع ألم الصدر (cp)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("ضغط الدم (trestbps)", 80, 200, 120)
    chol = st.sidebar.slider("الكوليسترول (chol)", 100, 600, 200)
    fbs = st.sidebar.radio("السكر الصائم > 120؟ (fbs)", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    
    st.sidebar.markdown("**الفحوصات المتقدمة**")
    restecg = st.sidebar.selectbox("نتائج تخطيط القلب (restecg)", [0, 1, 2])
    thalach = st.sidebar.slider("أقصى ضربات قلب (thalach)", 60, 220, 150)
    exang = st.sidebar.radio("ذبحة عند التمرين؟ (exang)", [0, 1], format_func=lambda x: "نعم" if x == 1 else "لا")
    oldpeak = st.sidebar.number_input("انخفاض ST (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("ميل ST (slope)", [0, 1, 2])
    ca = st.sidebar.selectbox("الأوعية الملونة (ca)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("فحص الثلاسيميا (Thal)", [0, 1, 2, 3])

    features = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame([features])

input_df = get_user_inputs()

st.title("🩺 Explainable Heart Disease Prediction")
st.markdown("---")

with st.expander("🔍 عرض بيانات المريض المدخلة"):
    st.write(input_df)

if model is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 نتيجة التحليل")
        prob = model.predict_proba(input_df)[0][1]
        
        if prob > 0.5:
            st.error(f"احتمالية إصابة عالية")
            st.metric(label="مستوى الخطر", value=f"{prob:.1%}", delta="مرتفع", delta_color="inverse")
        else:
            st.success(f"احتمالية إصابة منخفضة")
            st.metric(label="مستوى الخطر", value=f"{prob:.1%}", delta="آمن", delta_color="normal")
        
        st.info(" .ملاحظة: هذا التوقع داعم للقرار الطبي ولا يغني عن تشخيص الطبيب المختص")

    with col2:
        st.subheader(" شرح القرار SHAP Visualization")
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📝 التقرير الطبي الذكيSmart Analysis Report")
    
    shap_vals = shap_values.values[0]
    feature_names = input_df.columns
    analysis_df = pd.DataFrame({'Feature': feature_names, 'Impact': shap_vals}).sort_values(by='Impact', ascending=False)

    translate = {
        'age': 'العمر', 'sex': 'الجنس', 'cp': 'نوع ألم الصدر', 
        'trestbps': 'ضغط الدم المرتفع', 'chol': 'مستوى الكوليسترول', 'fbs': 'السكر الصائم',
        'restecg': 'تخطيط القلب', 'thalach': 'معدل ضربات القلب', 
        'exang': 'الذبحة الجهدية', 'oldpeak': 'انخفاض قطعة ST', 
        'slope': 'ميل قطعة ST', 'ca': 'عدد الأوعية المسدودة', 'thal': 'فحص الثلاسيميا'
    }

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="risk-box"><b style="color: red;"> :عوامل زادت من احتمالية الإصابة</b> </div>', unsafe_allow_html=True)
        high_risk = analysis_df[analysis_df['Impact'] > 0].head(3)
        if not high_risk.empty:
            for _, row in high_risk.iterrows():
                st.write(f" 🚩**{translate.get(row['Feature'], row['Feature'])}**: ساهم بشكل سلبي في رفع نسبة الخطر لإصابة القلب ")
        else:
            st.write(".لا توجد عوامل خطر بارزة")

    with c2:
        st.markdown('<div class="safe-box" > <b style="color: green;"> :عوامل ساهمت في خفض احتمالية الإصابة </b></div>', unsafe_allow_html=True)
        low_risk = analysis_df[analysis_df['Impact'] < 0].tail(3)
        if not low_risk.empty:
            for _, row in low_risk.iterrows():
                st.write(f"🛡️**{translate.get(row['Feature'], row['Feature'])}**: يعمل كعامل حماية ويقلل من احتمالية الإصابة بأمراض القلب ")
        else:
            st.write(".لا توجد عوامل حماية بارزة")

# استدعاء الوظيفة عند ضغط الزر في نهاية التطبيق
if st.button("🖨️ توليد التقرير الطبي الكامل"):
    generate_report(input_df, prob, high_risk, low_risk)


st.markdown("---")
st.caption("Developed by Aya Boubellouta | © 2026 Explainable AI Project")