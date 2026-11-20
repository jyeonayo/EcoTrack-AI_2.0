from pathlib import Path
import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="EcoTrack AI 2.0", page_icon="🌿", layout="wide")

ROOT = Path(__file__).parent
DATA = ROOT / "data"

FILES = {
    "검사 측정 데이터(무부하검사 샘플)": DATA / "inspection_unloaded_sample.csv",
    "운행차 배출가스 허용기준(샘플)": DATA / "in_use_emission_standards_sample.csv",
    "제작차 배출가스 허용기준(2026-10-01 자료)": DATA / "manufacture_emission_standards_20261001.csv",
}

def read_csv_safely(file_or_path):
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949 형식으로 저장해 주세요.")

def to_number(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(",", "")
    if s in ("", "-", "미측정", "해당없음", "NaN", "nan"):
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) if m else np.nan

def get_preloaded(name):
    p = FILES[name]
    if not p.exists():
        return None
    return read_csv_safely(p)

def date_filter(df, manufacture_date, inspection_date):
    d = df.copy()
    if "제작일자기준시작일" in d.columns:
        start = pd.to_datetime(d["제작일자기준시작일"], errors="coerce")
        end = pd.to_datetime(d["제작일자기준종료일"], errors="coerce")
        md = pd.Timestamp(manufacture_date)
        d = d[(start.isna() | (start <= md)) & (end.isna() | (end >= md))]
    if "적용일자기준시작일" in d.columns:
        start = pd.to_datetime(d["적용일자기준시작일"], errors="coerce")
        end = pd.to_datetime(d["적용일자기준종료일"], errors="coerce")
        idate = pd.Timestamp(inspection_date)
        d = d[(start.isna() | (start <= idate)) & (end.isna() | (end >= idate))]
    return d

def standards_lookup(standards, test_type, fuel, vehicle_class, manufacture_date, inspection_date):
    required = ["검사타입코드", "연료", "차종"]
    if not all(c in standards.columns for c in required):
        return pd.DataFrame()
    d = standards.copy()
    d = d[
        d["검사타입코드"].astype(str).str.strip().eq(test_type)
        & d["연료"].astype(str).str.strip().eq(fuel)
        & d["차종"].astype(str).str.strip().eq(vehicle_class)
    ]
    d = date_filter(d, manufacture_date, inspection_date)
    return d

def standard_numeric(row, col):
    return to_number(row[col]) if col in row.index else np.nan

def metric_columns(df):
    preferred = ["일산화탄소데이터", "탄화수소데이터", "이산화탄소데이터", "산소데이터", "질소산화물데이터"]
    return [c for c in preferred if c in df.columns]

st.title("🌿 EcoTrack AI 2.0")
st.caption("자동차 배출가스 기준 비교 및 AI 기반 검사결과 분석 시제품")

st.warning(
    "중요한 범위 안내: 이 앱은 교육·연구용 시제품입니다. 도로 원격측정값을 법정 검사 기준에 직접 대입하지 않습니다. "
    "제공된 운행차 기준 파일은 ASM-Idling/KD-147 부하검사 기준 샘플이며, 제공된 검사 측정 파일은 무부하검사 샘플입니다. "
    "따라서 두 파일을 서로 직접 비교해 법적 적합/부적합을 판정할 수 없습니다."
)

tabs = st.tabs(["① 기준 판별", "② 검사 데이터 탐색", "③ AI 학습", "④ 데이터·한계"])

# ---------- TAB 1: RULE-BASED STANDARDS ----------
with tabs[0]:
    st.subheader("규제 기준 판별 엔진")
    st.write("차량·검사 조건에 맞는 운행차 허용기준을 검색하고, 측정값과 비교합니다.")
    standards = get_preloaded("운행차 배출가스 허용기준(샘플)")
    if standards is None:
        st.error("운행차 기준 CSV가 data 폴더에 없습니다.")
    else:
        c1, c2, c3 = st.columns(3)
        test_types = sorted(standards["검사타입코드"].dropna().astype(str).unique())
        fuels = sorted(standards["연료"].dropna().astype(str).unique())
        vehicles = sorted(standards["차종"].dropna().astype(str).unique())
        with c1:
            test_type = st.selectbox("검사 방식", test_types)
            fuel = st.selectbox("연료", fuels)
        with c2:
            vehicle_class = st.selectbox("차종", vehicles)
            manufacture_date = st.date_input("자동차 제작일", value=pd.Timestamp("2010-01-01").date())
        with c3:
            inspection_date = st.date_input("검사 기준일", value=pd.Timestamp.today().date())
            st.caption("제작일·검사일은 기준 행의 적용기간 확인에 사용합니다.")

        matches = standards_lookup(standards, test_type, fuel, vehicle_class, manufacture_date, inspection_date)
        st.markdown("**기준 검색 결과**")
        if matches.empty:
            st.error("입력한 조건과 일치하는 기준 행을 찾지 못했습니다. 이 경우 임의의 기준을 선택하지 않고 '판정 불가'로 처리합니다.")
        else:
            if len(matches) > 1:
                st.warning(f"조건에 맞는 기준이 {len(matches)}개입니다. 적용 기준을 하나로 확정할 수 없어 행을 모두 보여줍니다. 실제 판정 전 공식 원자료와 세부 조건을 확인해야 합니다.")
                st.dataframe(matches, use_container_width=True)
                chosen_idx = st.selectbox("검토할 기준 행 선택(교육용)", list(matches.index), format_func=lambda x: f"기준 행 {x+1}")
                standard_row = matches.loc[chosen_idx]
            else:
                standard_row = matches.iloc[0]
                st.success("조건에 맞는 기준 행 1개를 찾았습니다.")
                st.dataframe(matches, use_container_width=True)

            st.markdown("**측정값 입력**")
            st.caption("측정값은 선택한 검사 방식과 동일한 검사 절차·단위로 얻은 값이어야 합니다. 무부하검사 측정값을 ASM/KD-147 기준과 비교하면 안 됩니다.")
            metric_map = {
                "CO": ("일산화탄소(CO)기준", "CO 측정값"),
                "HC": ("탄화수소(HC)기준", "HC 측정값"),
                "NOx": ("질소산화물(NOX)기준", "NOx 측정값"),
                "매연": ("매연(SMOKE)기준", "매연 측정값"),
            }
            entered = {}
            cols = st.columns(4)
            for i, (key, (std_col, label)) in enumerate(metric_map.items()):
                limit = standard_numeric(standard_row, std_col)
                with cols[i]:
                    if pd.notna(limit):
                        entered[key] = st.number_input(label, min_value=0.0, value=0.0, step=0.1, key=f"measure_{key}")
                    else:
                        st.text_input(label, value="기준값 없음", disabled=True, key=f"missing_{key}")
            if st.button("기준 비교 실행", type="primary"):
                results = []
                for key, (std_col, _) in metric_map.items():
                    limit = standard_numeric(standard_row, std_col)
                    if pd.isna(limit):
                        results.append({"항목": key, "측정값": None, "허용기준": None, "결과": "비교 불가(기준값 없음)"})
                    else:
                        val = entered.get(key)
                        results.append({"항목": key, "측정값": val, "허용기준": limit, "결과": "기준 이내" if val <= limit else "기준 초과"})
                result_df = pd.DataFrame(results)
                st.dataframe(result_df, use_container_width=True)
                if (result_df["결과"] == "기준 초과").any():
                    st.error("하나 이상의 입력 항목이 선택된 기준값을 초과했습니다. 이 결과는 입력값을 이용한 교육용 비교이며 공식 검사 판정이나 행정처분을 대신하지 않습니다.")
                elif (result_df["결과"] == "비교 불가(기준값 없음)").all():
                    st.warning("비교 가능한 기준 항목이 없습니다.")
                else:
                    st.success("입력한 항목 중 기준 초과는 확인되지 않았습니다. 다만 이것만으로 차량 전체의 법적 적합을 확정할 수는 없습니다.")

    st.divider()
    st.info("현재 제공된 운행차 허용기준 샘플에는 부하검사(ASM-Idling) 290행과 부하검사(KD-147) 10행이 있습니다. 제공된 무부하검사 샘플과 검사 방식이 다르므로 이 화면에서 무부하검사 결과를 직접 판정하지 않습니다.")

# ---------- TAB 2: EXPLORATION ----------
with tabs[1]:
    st.subheader("제공된 실제 검사 데이터 탐색")
    inspection = get_preloaded("검사 측정 데이터(무부하검사 샘플)")
    if inspection is None:
        st.error("검사 데이터 CSV가 data 폴더에 없습니다.")
    else:
        st.write(f"내장된 무부하검사 샘플: **{len(inspection):,}행 × {len(inspection.columns)}열**")
        st.caption("이 파일은 샘플 자료이며 차량별 법적 판정 라벨이나 차량 식별·연식 정보가 포함되어 있다고 가정하지 않습니다.")
        st.dataframe(inspection.head(20), use_container_width=True)
        cols = metric_columns(inspection)
        if cols:
            selected = st.selectbox("분포를 확인할 측정 항목", cols)
            numeric = pd.to_numeric(inspection[selected], errors="coerce").dropna()
            if len(numeric):
                fig = px.histogram(x=numeric, nbins=25, labels={"x": selected, "count": "빈도"}, title=f"{selected} 분포")
                st.plotly_chart(fig, use_container_width=True)
                st.metric("유효 측정값 수", f"{len(numeric):,}")
                st.metric("평균", f"{numeric.mean():.3f}")
                st.metric("중앙값", f"{numeric.median():.3f}")
        st.download_button("검사 샘플 CSV 다운로드", inspection.to_csv(index=False, encoding="utf-8-sig"), "inspection_sample.csv", "text/csv")

# ---------- TAB 3: ML TRAINING ----------
with tabs[2]:
    st.subheader("AI 지도학습: Random Forest")
    st.write("실제 검사 결과(적합/부적합)가 붙어 있는 학습용 CSV를 업로드하면 분류 모델을 학습하고 평가합니다.")
    st.warning(
        "현재 내장된 무부하검사 샘플에는 확인 가능한 적합/부적합 정답 열이 없습니다. "
        "정답을 임의로 만들어 학습시키지 않습니다. 실제 검사결과 라벨이 있는 데이터가 준비되어야 지도학습이 가능합니다."
    )
    st.markdown("**학습 데이터 형식**")
    st.code("CO,HC,CO2,O2,NOx,검사결과\n0.1,25,14.5,0.5,0,적합\n1.2,180,12.0,1.1,0,부적합", language="csv")
    st.caption("위 두 행은 형식 예시일 뿐 실제 관측값이나 학습 데이터가 아닙니다. 실제 측정값과 공식 검사 결과로 채워야 합니다.")
    uploaded_train = st.file_uploader("정답 라벨이 포함된 학습용 CSV 업로드", type=["csv"], key="train_csv")
    if uploaded_train is not None:
        try:
            train_df = read_csv_safely(uploaded_train)
            st.write(f"업로드 데이터: {len(train_df):,}행 × {len(train_df.columns)}열")
            st.dataframe(train_df.head(10), use_container_width=True)
            target = st.selectbox("정답 열 선택", list(train_df.columns), index=len(train_df.columns)-1)
            numeric_cols = [c for c in train_df.columns if c != target and pd.api.types.is_numeric_dtype(pd.to_numeric(train_df[c], errors="coerce"))]
            feature_cols = st.multiselect("학습에 사용할 수치형 측정 항목", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
            if st.button("Random Forest 학습 및 평가", type="primary"):
                if not feature_cols:
                    st.error("학습에 사용할 측정 항목을 하나 이상 선택해 주세요.")
                else:
                    work = train_df[feature_cols + [target]].copy()
                    for c in feature_cols:
                        work[c] = pd.to_numeric(work[c], errors="coerce")
                    work = work.dropna(subset=[target])
                    y = work[target].astype(str)
                    counts = y.value_counts()
                    if len(counts) < 2 or counts.min() < 2:
                        st.error("학습과 평가를 위해 정답 종류가 2개 이상이고, 각 종류에 최소 2개 이상의 사례가 필요합니다.")
                    elif len(work) < 10:
                        st.error("현재 데이터는 10행 미만입니다. 성능을 신뢰하기 어려우므로 더 많은 실제 검사 자료를 준비해 주세요.")
                    else:
                        X = work[feature_cols]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.25, random_state=42, stratify=y
                        )
                        pipe = Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
                        ])
                        pipe.fit(X_train, y_train)
                        pred = pipe.predict(X_test)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
                        c2.metric("Precision", f"{precision_score(y_test, pred, average='weighted', zero_division=0):.3f}")
                        c3.metric("Recall", f"{recall_score(y_test, pred, average='weighted', zero_division=0):.3f}")
                        c4.metric("F1-score", f"{f1_score(y_test, pred, average='weighted', zero_division=0):.3f}")
                        st.text("분류 성능 보고서")
                        st.code(classification_report(y_test, pred, zero_division=0))
                        cm = confusion_matrix(y_test, pred, labels=sorted(y.unique()))
                        cm_df = pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique()))
                        st.write("혼동행렬")
                        st.dataframe(cm_df)
                        st.session_state["trained_model"] = pipe
                        st.session_state["trained_features"] = feature_cols
                        st.session_state["trained_classes"] = sorted(y.unique())
                        st.success("학습과 평가가 완료되었습니다. 테스트 데이터 성능을 확인한 뒤 모델을 사용해야 합니다.")
            if "trained_model" in st.session_state:
                buffer = io.BytesIO()
                joblib.dump({"model": st.session_state["trained_model"], "features": st.session_state["trained_features"], "classes": st.session_state["trained_classes"]}, buffer)
                st.download_button("학습 모델 다운로드(.joblib)", buffer.getvalue(), "ecotrack_random_forest.joblib", "application/octet-stream")
        except Exception as e:
            st.error(f"CSV를 처리하지 못했습니다: {e}")

# ---------- TAB 4: DATA AND LIMITS ----------
with tabs[3]:
    st.subheader("내장 데이터 출처와 적용 범위")
    for title, path in FILES.items():
        if path.exists():
            df = get_preloaded(title)
            st.markdown(f"**{title}**")
            st.write(f"- 행·열: {df.shape[0]}행 × {df.shape[1]}열")
            st.write(f"- 파일: `{path.relative_to(ROOT)}`")
    manufacture = get_preloaded("제작차 배출가스 허용기준(2026-10-01 자료)")
    if manufacture is not None:
        st.info(
            f"제작차 허용기준 자료는 {len(manufacture)}행이며, 제작·인증 단계의 시험기준을 담고 있습니다. "
            "운행 중인 차량의 정기검사 기준과 목적·시험방법이 다르므로 운행차 적합 판정에 직접 사용하지 않습니다."
        )
    st.markdown("**중요한 기술적 한계**")
    st.markdown("""
    - 운행차 기준 샘플은 ASM-Idling 및 KD-147 부하검사 기준을 포함합니다.
    - 검사 측정 샘플은 자동차관리법상 무부하검사 자료입니다. 두 파일의 검사 방법이 다르므로 직접 비교하지 않습니다.
    - 현재 내장 검사 샘플에는 차량별 연료·제작연도·공식 적합/부적합 라벨이 충분히 연결되어 있지 않습니다.
    - 지도학습 모델은 동일하거나 비교 가능한 검사방법으로 측정된 실제 데이터와 공식 결과 라벨을 추가로 확보한 뒤 학습해야 합니다.
    - 도로 원격측정(RSD) 자료는 검사장 정지/부하/무부하 시험과 다른 측정 환경이므로, 별도 검증 없이 법정 검사기준에 직접 대입하지 않습니다.
    - 이 프로그램은 단속·행정처분을 자동 결정하지 않습니다. 기준 초과 의심은 공식 절차에 따른 추가 검사·확인의 참고 정보입니다.
    """)
    st.markdown("**공식 법령 확인**")
    st.markdown("- [국가법령정보센터: 대기환경보전법](https://www.law.go.kr/법령/대기환경보전법)")
    st.markdown("- [국가법령정보센터: 대기환경보전법 시행규칙](https://www.law.go.kr/법령/대기환경보전법시행규칙)")
    st.markdown("- [공공데이터포털: 무부하검사 배출가스검사정보](https://www.data.go.kr/data/15088042/fileData.do)")
