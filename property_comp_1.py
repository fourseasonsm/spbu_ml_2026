"""
Данный код описывает в основном мои попытки что-то улучшить, основную работу.
На основе него был сделан финальный:
сглаженное TE, логарифмированные расстояния, стек (Ridge + LightGBM на остатках).
Дал 0.020097 mape, выглядел так
            ridge_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge",  Ridge(**BEST_RIDGE))
            ])
            lgb_model = lgb.train(
                lgb_params, train_ds,
                valid_sets=[val_ds],
                num_boost_round=3000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            ) 
Параметры Ridge: alpha = 0.5, solver = 'saga', tol = 1e-4, обучается на числовых признаках (без категориальных колонок) цель логарифм цены (log1p).
Затем бустинг пытается взять то что не взял ridge из за нелинейности. Финальное предсказание складывается из суммы предсказаний двух моделей и возвращается
в исходный масштаб через expm1.
BASE_LGB_PARAMS = {
    "learning_rate":     0.04606518028238652,
    "num_leaves":        106,
    "min_child_samples": 21,
    "feature_fraction":  0.6099244329852761,
    "bagging_fraction":  0.6710453977513152,
    "bagging_freq":      6,
    "reg_alpha":         0.0010048139892262543,
    "reg_lambda":        0.36882218776230447,
    "objective":         "regression",
    "metric":            "rmse",
    "boosting_type":     "gbdt",
    "verbose":           -1,
    "n_jobs":            -1,
}
Код по факту дублировал фичи и очистку из этого и состоял из миллиона переборов параметров в колабе, поэтому его я не стала вставлять сюда а описала 
комментариями. При необходимости могу прислать его отдельно.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

TARGET   = "price_target"
SEED     = 1
N_FOLDS  = 3   #5 давали примерно то же качество но времени уходило сильно больше
train = pd.read_csv("period_1_train_data.csv")
test  = pd.read_csv("test_x.csv")
test_ids = test["id"].copy() if "id" in test.columns else pd.RangeIndex(len(test))

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Поудаляла откровенные выбросы
train = train.dropna(subset=[TARGET]).reset_index(drop=True)

upper_clip = train[TARGET].quantile(0.999)
lower_clip = train[TARGET].quantile(0.001)
train[TARGET] = train[TARGET].clip(lower=lower_clip, upper=upper_clip)
print(f"Target clipped to [{lower_clip:.0f}, {upper_clip:.0f}]")

"""
  agreement_date year/month/quarter/month_sin/month_cos/days_since_min
    На графике "медианная цена по годам" было
    плавное снижение, есть влияник
    sin/cos месяца нужны чтобы январь (1) и декабрь (12) были рядом
    в пространстве признаков.
  sqm_per_room = square / (rooms + 1) квартиры с большим количеством комнат за квадрат стоили дешевле. 
  Этим я хотела отфильтровать 5комнатные квартиры с комнатами кладовками

  floor_ratio = floor / max_levels, is_top_floor, is_bottom_floor
    1-й и последний этажи дешевле средних. floor_ratio нормирует этаж
    относительно высоты дома.
  is_suburb, is_oblast
    На графике "медианная цена по region_name_cat" разница Город/Пригород/Область
    составляет 40%. Решила разбить для надежности.
  bank_density, leisure_density = cnt / (buildings_cnt + 1)
    Качество района

  *_log для всех distance-колонок

  class_x_floor = class_cat * floor
    По каким то причинам в элитных домах верхние этажи стоят дороже
Проверялось еще много чего разного, но все довольно сильно роняло результат. Возможно 
стоило бы выкинуть часть из того что оставила тут, и результат был бы лучше.
"""

def feature_engineering(df, is_train=True, train_df=None):
    df = df.copy()
    if "agreement_date" in df.columns:
        df["agreement_date"] = pd.to_datetime(df["agreement_date"], errors="coerce")
        df["year"]      = df["agreement_date"].dt.year
        df["month"]     = df["agreement_date"].dt.month
        df["quarter"]   = df["agreement_date"].dt.quarter
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        if train_df is not None and "agreement_date" in train_df.columns:
            min_date = pd.to_datetime(train_df["agreement_date"], errors="coerce").min()
            df["days_since_min"] = (df["agreement_date"] - min_date).dt.days
        df.drop(columns=["agreement_date"], inplace=True)

    if "rooms_4" in df.columns:
        df["rooms_4"] = df["rooms_4"].replace("студия", 0)
        df["rooms_4"] = pd.to_numeric(df["rooms_4"], errors="coerce")
        df["is_studio"] = (df["rooms_4"] == 0).astype(int)

    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = df[col].where(df[col] != -999.0, other=np.nan)

    if "square" in df.columns and "rooms_4" in df.columns:
        df["sqm_per_room"] = df["square"] / (df["rooms_4"].replace(0, np.nan) + 1)

    if "floor" in df.columns and "location_max_levels_max" in df.columns:
        df["floor_ratio"]     = df["floor"] / df["location_max_levels_max"].replace(0, np.nan)
        df["is_top_floor"]    = (df["floor"] == df["location_max_levels_max"]).astype(int)
        df["is_bottom_floor"] = (df["floor"] == 1).astype(int)

    if "region_name_cat" in df.columns:
        df["is_suburb"] = (df["region_name_cat"] == "Пригород").astype(int)
        df["is_oblast"] = (df["region_name_cat"] == "Область").astype(int)

    if "location_buildings_cnt" in df.columns:
        if "location_pop_bank_cnt" in df.columns:
            df["bank_density"] = df["location_pop_bank_cnt"] / (df["location_buildings_cnt"] + 1)
        if "location_leisure_cnt" in df.columns:
            df["leisure_density"] = df["location_leisure_cnt"] / (df["location_buildings_cnt"] + 1)

    for col in [c for c in df.columns if "distance" in c or "w_mean" in c]:
        df[col + "_log"] = np.log1p(df[col].clip(lower=0))

    if "square" in df.columns and "rooms_4" in df.columns:
        df["square_x_rooms"] = df["square"] * df["rooms_4"]

    return df


train = feature_engineering(train, is_train=True)
test  = feature_engineering(test,  is_train=False, train_df=train)

cat_cols_for_agg = ["district_cat", "developer_cat", "class_cat", "region_name_cat"]
cat_cols_for_agg = [c for c in cat_cols_for_agg if c in train.columns]

kf_enc = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
global_mean = np.log1p(train[TARGET].values).mean()

for col in cat_cols_for_agg:
    train[f"{col}_mean_log_price"] = np.nan
    for tr_idx, val_idx in kf_enc.split(train):
        means = train.iloc[tr_idx].groupby(col)[TARGET].apply(lambda x: np.log1p(x).mean())
        train.loc[val_idx, f"{col}_mean_log_price"] = train.loc[val_idx, col].map(means)
    train[f"{col}_mean_log_price"] = train[f"{col}_mean_log_price"].fillna(global_mean)

    full_means = train.groupby(col)[TARGET].apply(lambda x: np.log1p(x).mean())
    test[f"{col}_mean_log_price"] = test[col].map(full_means).fillna(global_mean)

print("Target encoding features added.")

cat_cols = ["region_name_cat", "district_cat", "developer_cat",
            "hc_name_cat", "interior_cat", "class_cat", "stage_cat", "corpus_cat"]
cat_cols = [c for c in cat_cols if c in train.columns]

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train[cat_cols] = enc.fit_transform(train[cat_cols].astype(str))
test[cat_cols]  = enc.transform(test[cat_cols].astype(str))

# Взаимодействие класса дома и этажа (после кодирования, когда class_cat числовой)
if "class_cat" in train.columns and "floor" in train.columns:
    train["class_x_floor"] = train["class_cat"] * train["floor"]
    test["class_x_floor"]  = test["class_cat"] * test["floor"]

drop_cols    = [TARGET] + (["id"] if "id" in train.columns else [])
feature_cols = [c for c in train.columns if c not in drop_cols]

for c in set(feature_cols) - set(test.columns):
    test[c] = np.nan

X      = train[feature_cols].values.astype(np.float32)
y      = train[TARGET].values.astype(np.float32)
X_test = test[feature_cols].values.astype(np.float32)
y_log  = np.log1p(y)

print(f"Total features: {len(feature_cols)}")

"""
Тестировались: LightGBM, XGBoost, CatBoost, Ridge на признаках.
Ridge на сырых признаках: MAPE 0.08
Пробовала ансамбль из 3 бустингов, долго думало, результаты были сомнительные
В итоге сделала ridge + бустинг, это и пошло в сдачу

Перебирала параметры:
  learning_rate: пробовали 0.05 / 0.03 / 0.02 / 0.01
     0.02 оптимально

  num_leaves (LGB) / max_depth (XGB): 63 127 255 / 6 7 8 9
    LGB: 127 и 255 давали одинаковое качество, выбрали 127 как менее склонный к переобучению
    XGB: depth=8 чуть лучше depth=7 на этом датасете (~0.0003 MAPE)

  reg_alpha: 0.1 / 0.5 / 1.0
    0.5 лучший 

  n_estimators: 3000 с early_stopping=100
    реально использовалось 800–1500 деревьев (early stopping срабатывал раньше)

  N_FOLDS: 5 и 3
    качество идентично, скорость лучше
}
"""
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X));  preds_lgb = np.zeros(len(X_test))
oof_xgb = np.zeros(len(X));  preds_xgb = np.zeros(len(X_test))
oof_cat = np.zeros(len(X));  preds_cat = np.zeros(len(X_test))

residuals_std_lgb = []
residuals_std_xgb = []
residuals_std_cat = []

lgb_params = dict(
    objective="regression",  
    metric="rmse",
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=10,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.5,
    reg_lambda=1.0,
    n_estimators=3000,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X[tr_idx], y_log[tr_idx],
        eval_set=[(X[val_idx], y_log[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )
    val_pred_log = model.predict(X[val_idx])
    val_pred = np.expm1(val_pred_log)
    oof_lgb[val_idx] = val_pred
    preds_lgb += np.expm1(model.predict(X_test)) / N_FOLDS
    residuals_std_lgb.append(np.std(y_log[val_idx] - val_pred_log))
    print(f"  Fold {fold+1}  MAPE: {mean_absolute_percentage_error(y[val_idx], np.clip(val_pred, 1, None)):.4f}")

print(f"LightGBM OOF MAPE: {mean_absolute_percentage_error(y, np.clip(oof_lgb, 1, None)):.4f}")

xgb_params = dict(
    objective="reg:squarederror",
    learning_rate=0.02,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    n_estimators=3000,
    early_stopping_rounds=100,  
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    verbosity=0,
)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X[tr_idx], y_log[tr_idx],
        eval_set=[(X[val_idx], y_log[val_idx])],
        verbose=500,
    )
    val_pred_log = model.predict(X[val_idx])
    val_pred = np.expm1(val_pred_log)
    oof_xgb[val_idx] = val_pred
    preds_xgb += np.expm1(model.predict(X_test)) / N_FOLDS
    residuals_std_xgb.append(np.std(y_log[val_idx] - val_pred_log))
    print(f"  Fold {fold+1}  MAPE: {mean_absolute_percentage_error(y[val_idx], np.clip(val_pred, 1, None)):.4f}")

print(f"XGBoost OOF MAPE: {mean_absolute_percentage_error(y, np.clip(oof_xgb, 1, None)):.4f}")

print("\n=== CatBoost (log target) ===")
cat_params = dict(
    loss_function="RMSE",   
    learning_rate=0.02,
    depth=7,
    l2_leaf_reg=7,
    iterations=3000,
    random_seed=SEED,
    verbose=500,
    early_stopping_rounds=100,
    thread_count=-1,
)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    model = CatBoostRegressor(**cat_params)
    model.fit(
        X[tr_idx], y_log[tr_idx],
        eval_set=(X[val_idx], y_log[val_idx]),
        use_best_model=True,
    )
    val_pred_log = model.predict(X[val_idx])
    val_pred = np.expm1(val_pred_log)
    oof_cat[val_idx] = val_pred
    preds_cat += np.expm1(model.predict(X_test)) / N_FOLDS
    residuals_std_cat.append(np.std(y_log[val_idx] - val_pred_log))
    print(f"  Fold {fold+1}  MAPE: {mean_absolute_percentage_error(y[val_idx], np.clip(val_pred, 1, None)):.4f}")

print(f"CatBoost OOF MAPE: {mean_absolute_percentage_error(y, np.clip(oof_cat, 1, None)):.4f}")

"""
Какие то 5 копеек добавило в mape. Конкретную мотивацию я не помню уже тк отказалась от этого подхода.
"""
avg_std_lgb = np.mean(residuals_std_lgb)
avg_std_xgb = np.mean(residuals_std_xgb)
avg_std_cat = np.mean(residuals_std_cat)

print(f"\n=== Bias Correction (σ): LGB={avg_std_lgb:.4f}, XGB={avg_std_xgb:.4f}, CAT={avg_std_cat:.4f}")

def correct_bias(pred_log, std):
    return np.expm1(pred_log + 0.5 * std**2)

oof_lgb_c = correct_bias(np.log1p(oof_lgb), avg_std_lgb)
oof_xgb_c = correct_bias(np.log1p(oof_xgb), avg_std_xgb)
oof_cat_c = correct_bias(np.log1p(oof_cat), avg_std_cat)

preds_lgb_c = correct_bias(np.log1p(preds_lgb), avg_std_lgb)
preds_xgb_c = correct_bias(np.log1p(preds_xgb), avg_std_xgb)
preds_cat_c = correct_bias(np.log1p(preds_cat), avg_std_cat)

print(f"После коррекции — LGB: {mean_absolute_percentage_error(y, np.clip(oof_lgb_c, 1, None)):.4f}  "
      f"XGB: {mean_absolute_percentage_error(y, np.clip(oof_xgb_c, 1, None)):.4f}  "
      f"CAT: {mean_absolute_percentage_error(y, np.clip(oof_cat_c, 1, None)):.4f}")

"""
Тестировались три подхода:
  1. Простое среднее (equal weights): MAPE 0.022
  2. Оптимизация весов через scipy.minimize (Nelder-Mead): MAPE 0.0218
     Блендер давал XGBoost вес 1.0, остальным 0
  3. Ridge стекинг на OOF: MAPE 0.0215 
alpha=1.0 выбрана без перебора
"""
stack_X    = np.column_stack([oof_lgb_c, oof_xgb_c, oof_cat_c])
stack_test = np.column_stack([preds_lgb_c, preds_xgb_c, preds_cat_c])

meta_model = Ridge(alpha=1.0, random_state=SEED)
meta_model.fit(stack_X, y)

oof_stack  = meta_model.predict(stack_X)
test_stack = meta_model.predict(stack_test)

print(f"Ridge коэффициенты: LGB={meta_model.coef_[0]:.3f}, XGB={meta_model.coef_[1]:.3f}, CAT={meta_model.coef_[2]:.3f}")
print(f"Stacked OOF MAPE: {mean_absolute_percentage_error(y, np.clip(oof_stack, 1, None)):.4f}")

final_preds = np.clip(test_stack, 1, None)

submission = pd.DataFrame({"id": test_ids, "price_target": final_preds})
submission.to_csv("submission_improved.csv", index=False)

print(f"\nsubmission_improved.csv saved! Rows: {len(submission)}")
print(submission.head(10))
print(f"Статистика: min={final_preds.min():.0f}, max={final_preds.max():.0f}, mean={final_preds.mean():.0f}")

