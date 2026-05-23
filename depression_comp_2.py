"""
РЕЗУЛЬТАТЫ:
  Logistic Regression OOF F1 : 0.8297  (threshold=0.56)
  CatBoost OOF F1            : 0.8164  (threshold=0.31)
  Ensemble OOF F1            : 0.8340  (threshold=0.44)
  Ensemble weights           : LR=0.80, CatBoost=0.20
Для сдачи использовала вариант с ансамблем, давал лучше на public. В итоге на private оказался лучше вариант с чистой Logistic Regression
но разница была не особо большой (продвинулась бы на одно место)
Само обучение и перебор делала в colab
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report
from scipy.sparse import hstack, csr_matrix
from catboost import CatBoostClassifier

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
y     = train['label'].astype(int).values

# Дисбаланс классов: 8730 негативных / 2070 позитивных
# Учитывается через class_weight='balanced' в LR
# и auto_class_weights='Balanced' в CatBoost
scale_pos = (y == 0).sum() / (y == 1).sum()

# Минимальная очистка: lowercase, удаление ссылок и не-ascii символов.
# Знаки !?. оставлены
def clean(s):
    s = s.str.lower()
    s = s.str.replace(r'http\S+', ' ', regex=True)
    s = s.str.replace(r'[^a-z\s!?.]', ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

for df in [train, test]:
    df['title_c'] = clean(df['title'].fillna(''))
    df['body_c']  = clean(df['body'].fillna(''))
    df['full_c']  = df['title_c'] + ' ' + df['body_c']


# депрессивные посты в среднем длиннее (mean 1010 vs 286 символов),
# содержат больше ключевых слов, выше доля местоимения "I", ниже лексическое разнообразие.
# Всего 17 признаков добавлены к TF-IDF для обеих моделей.
#
# Ключевые слова подобраны из общих идей что может стать маркером депрессии и плохого состояния (сидела и читала датасет, потом составила). Сильное увеличение
# количества и разнообразия ключевых слов ничего не дало, пробовала еще их ранжировать по степени депрессивности (suicid как совсем жесть
# exhausted как просто может быть). Результата не было, даже ухудшало, выкинула.
depr_kw = (r'depress|suicid|hopeless|worthless|empty|numb|anxious|anxiety|'
           r'selfharm|cutting|tired of|no point|want to die|kill myself|'
           r'lonely|loneliness|cry|crying|isolat|cant sleep|insomnia|'
           r'hate myself|burden|disappear|overdose|nothing matters|'
           r'meaningless|pointless|exhausted|breakdown|trauma|vomit|'
           r'falling apart|give up|dysmorphia|hate my body|therapist|therapy')

def hand_feats(df):
    t, b, f = df['title_c'], df['body_c'], df['full_c']
    wlen = f.str.split().str.len().clip(1)
    return np.column_stack([
        f.str.len(),                                                              # длина полного текста
        b.str.len(),                                                              # длина body
        t.str.len(),                                                              # длина title
        wlen,                                                                     # количество слов
        df['body'].isna().astype(int),                                            # body отсутствует (2074 пропуска в трейне)
        f.apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0),  # средняя длина слова
        f.str.count('!'),                                                         # восклицательные знаки
        f.str.count(r'\?'),                                                       # вопросительные знаки
        f.str.count(r'\.\.\.'),                                                   # многоточия (мб растерянность, есть в этом какая то окраска)
        f.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1)),         # доля заглавных букв
        f.str.count(depr_kw),                                                     # число совпадений с депр. лексиконом
        f.str.count(r"\bno\b|\bnot\b|\bnever\b|\bnothing\b|\bnobody\b"),          # плотность отрицаний
        f.str.count(r'\bi\b') / wlen,                                             # доля "I" 
        f.apply(lambda x: len(set(x.split())) / max(len(x.split()), 1)),          # лексическое разнообразие
        b.str.count(depr_kw),                                                     # ключевые слова только в body
        t.str.count(depr_kw),                                                     # ключевые слова только в title
        (f.str.count(depr_kw) > 0).astype(int),                                  # бинарный флаг: есть хоть одно ключевое слово
    ]).astype(np.float32)

Xh_raw    = hand_feats(train)
Xh_te_raw = hand_feats(test)

# LR чувствителен к масштабу признаков (text_len 10000, i_ratio 0.03),
# поэтому ручные фичи масштабируются через MaxAbsScaler перед подачей в LR.
# CatBoost масштаб не важен 
scaler       = MaxAbsScaler()
Xh_scaled    = csr_matrix(scaler.fit_transform(Xh_raw))
Xh_te_scaled = csr_matrix(scaler.transform(Xh_te_raw))
Xh_raw       = csr_matrix(Xh_raw)
Xh_te_raw    = csr_matrix(Xh_te_raw)

#   - word (1,2)-граммы на полном тексте: 80k фич 
#   - char_wb (3,5)-граммы на полном тексте: 40k фич 
#   - word (1,2)-граммы только на title: 20k фич 
# sublinear_tf=True (log-масштабирование TF) стабильно улучшало F1.
# max_df=0.9 убирает слова, встречающиеся почти везде.
# min_df=2-5 убирает редкие шумовые токены.
vw = TfidfVectorizer(ngram_range=(1, 2), max_features=80_000,
                     sublinear_tf=True, min_df=3, max_df=0.9)
vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                     max_features=40_000, sublinear_tf=True, min_df=5)
vt = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000,
                     sublinear_tf=True, min_df=2)

Xw_tr = vw.fit_transform(train['full_c'])
Xc_tr = vc.fit_transform(train['full_c'])
Xt_tr = vt.fit_transform(train['title_c'])
Xw_te = vw.transform(test['full_c'])
Xc_te = vc.transform(test['full_c'])
Xt_te = vt.transform(test['title_c'])

# LR — scaled ручные фичи CatBoost просто как есть
X_tr_lr  = hstack([Xw_tr, Xc_tr, Xt_tr, Xh_scaled])
X_te_lr  = hstack([Xw_te, Xc_te, Xt_te, Xh_te_scaled])
X_tr_cat = hstack([Xw_tr, Xc_tr, Xt_tr, Xh_raw])
X_te_cat = hstack([Xw_te, Xc_te, Xt_te, Xh_te_raw])

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Перебирались: C ∈ {0.1, 0.3, 1.0, 3.0, 10.0}, solver ∈ {saga, lbfgs, liblinear},
# class_weight ∈ {'balanced', {0:1, 1:4.2}}.
#
# Лучший результат: C=10.0, solver='saga', class_weight='balanced' OOF F1=0.8297
# При C=3.0 F1=0.8285 почти одинаково
# 'balanced' и явный вес {0:1, 1:4.2} давали схожий результат..
lr = LogisticRegression(C=10.0, max_iter=2000, class_weight='balanced',
                        solver='saga', n_jobs=-1)
lr_oof = cross_val_predict(lr, X_tr_lr, y, cv=cv5, method='predict_proba')[:, 1]

best_f1_lr, best_thr_lr = 0, 0.5
for thr in np.arange(0.2, 0.8, 0.01):
    f1 = f1_score(y, (lr_oof >= thr).astype(int))
    if f1 > best_f1_lr:
        best_f1_lr, best_thr_lr = f1, thr

print(f"LR  OOF F1: {best_f1_lr:.4f}  (thr={best_thr_lr:.2f})")


# Параметры зафиксированы по аналогии с LGBM, который давал OOF F1=0.816
# на тех же данных при n_estimators=800.
# CatBoost выбран как альтернатива LGBM 
cat_params = {
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 6,             
    'l2_leaf_reg': 0.1,
    'min_child_samples': 20,
    'colsample_bylevel': 0.7,
    'subsample': 0.8,
    'random_seed': 42,
    'verbose': 0,
    'thread_count': -1,
    'early_stopping_rounds': 50,
    'eval_metric': 'F1',
    'auto_class_weights': 'Balanced',
}

cat     = CatBoostClassifier(**cat_params)
cat_oof = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(cv5.split(X_tr_cat, y)):
    cat_fold = CatBoostClassifier(**cat_params)
    cat_fold.fit(X_tr_cat[train_idx], y[train_idx],
                 eval_set=(X_tr_cat[val_idx], y[val_idx]),
                 verbose=0)
    cat_oof[val_idx] = cat_fold.predict_proba(X_tr_cat[val_idx])[:, 1]

best_f1_cat, best_thr_cat = 0, 0.5
for thr in np.arange(0.2, 0.8, 0.01):
    f1 = f1_score(y, (cat_oof >= thr).astype(int))
    if f1 > best_f1_cat:
        best_f1_cat, best_thr_cat = f1, thr

print(f"CAT OOF F1: {best_f1_cat:.4f}  (thr={best_thr_cat:.2f})")


# Простое взвешенное усреднение вероятностей.
# Веса и порог перебираются совместно по сетке на OOF-предсказаниях
# Результат: оптимальный вес CatBoost=0.20, LR=0.80 
# но небольшая доля CatBoost стабильно добавляет +0.004 к F1. Но как выяснилось позже этот прикол был только на public датасете
best_f1, best_w, best_thr = 0, 0.5, 0.5
for w in np.arange(0.1, 1.0, 0.05):        
    blend = w * cat_oof + (1 - w) * lr_oof
    for thr in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(y, (blend >= thr).astype(int))
        if f1 > best_f1:
            best_f1, best_w, best_thr = f1, w, thr

blend_oof = best_w * cat_oof + (1 - best_w) * lr_oof
print(f"\nEnsemble OOF F1 : {best_f1:.4f}")
print(f"CatBoost weight : {best_w:.2f}  |  LR weight: {1-best_w:.2f}  |  threshold: {best_thr:.2f}")
print("\n=== Classification report ===")
print(classification_report(y, (blend_oof >= best_thr).astype(int),
      target_names=['no_depression', 'depression']))

lr.fit(X_tr_lr, y)
cat.fit(X_tr_cat, y)

blend_te = (best_w       * cat.predict_proba(X_te_cat)[:, 1] +
            (1 - best_w) * lr.predict_proba(X_te_lr)[:, 1])
pred_te  = (blend_te >= best_thr).astype(int)

sub = pd.DataFrame({'id': test['id'], 'label': pred_te})
sub.to_csv('submission_final.csv', index=False)
print(f"\nDone: submission_final.csv")
print(sub['label'].value_counts())
