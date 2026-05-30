import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier
import json

import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont('NanumGothic.ttf')
    plt.rcParams['font.family'] = 'NanumGothic'
except Exception as e:
    print('font warn', e)
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10
RNG = 42

df = pd.read_csv('titanic.csv')
print("Shape:", df.shape)

# ============ STEP 2: EDA ============
results = {}
results['shape'] = list(df.shape)
results['dtypes'] = {c: str(t) for c, t in df.dtypes.items()}

# Missing ratio
miss = (df.isnull().mean() * 100).round(2)
results['missing'] = miss[miss > 0].to_dict()

# Target balance
results['target_balance'] = df['Survived'].value_counts().to_dict()

# --- Figure 1: Histograms of numeric ---
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col in zip(axes.ravel(), num_cols):
    df[col].dropna().hist(bins=30, ax=ax, color='#4C72B0', edgecolor='white')
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col); ax.set_ylabel('Count')
plt.suptitle('Histograms of Numeric Variables', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('fig_hist.png', dpi=120, bbox_inches='tight'); plt.close()

# --- Figure 2: Boxplots (outlier detection) ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, col in zip(axes, ['Age', 'Fare', 'SibSp']):
    sns.boxplot(y=df[col], ax=ax, color='#DD8452')
    ax.set_title(f'Boxplot: {col}')
plt.suptitle('Boxplots for Outlier Detection', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('fig_box.png', dpi=120, bbox_inches='tight'); plt.close()

# --- Figure 3: Correlation heatmap ---
plt.figure(figsize=(8, 6))
corr = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=.5)
plt.title('Correlation Heatmap', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('fig_heat.png', dpi=120, bbox_inches='tight'); plt.close()

# --- Figure 4: Countplots (categorical vs target) ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, ['Sex', 'Pclass', 'Embarked']):
    sns.countplot(data=df, x=col, hue='Survived', ax=ax, palette='Set2')
    ax.set_title(f'{col} vs Survived')
plt.suptitle('Categorical Variables vs Survival', fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig('fig_count.png', dpi=120, bbox_inches='tight'); plt.close()

# --- Figure 5: target distribution ---
plt.figure(figsize=(5, 4))
df['Survived'].value_counts().plot(kind='bar', color=['#C44E52', '#55A868'])
plt.title('Target Variable Distribution (Survived)', weight='bold')
plt.xlabel('Survived (0=No, 1=Yes)'); plt.ylabel('Count')
plt.xticks([0, 1], ['Died', 'Survived'], rotation=0)
plt.tight_layout()
plt.savefig('fig_target.png', dpi=120, bbox_inches='tight'); plt.close()

# Outlier counts via IQR
def iqr_outliers(s):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum())
results['outliers'] = {c: iqr_outliers(df[c].dropna()) for c in ['Age', 'Fare', 'SibSp', 'Parch']}
print("EDA figures saved.")

# ============ STEP 3 & 5: Feature Engineering + Model experiments ============
def build_features(data, derived=True):
    d = data.copy()
    # Title extraction from Name
    d['Title'] = d['Name'].str.extract(r',\s*([^\.]+)\.')
    d['Title'] = d['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    d['Title'] = d['Title'].replace(['Mlle','Ms'],'Miss').replace('Mme','Mrs')
    if derived:
        d['FamilySize'] = d['SibSp'] + d['Parch'] + 1            # derived 1
        d['IsAlone'] = (d['FamilySize'] == 1).astype(int)        # derived 2
        d['FarePerPerson'] = d['Fare'] / d['FamilySize']         # derived 3
    return d

df_fe = build_features(df)

target = 'Survived'
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', target]
X_all = df_fe.drop(columns=drop_cols)
y = df_fe[target]

num_feats = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'FarePerPerson']
cat_feats = ['Sex', 'Embarked', 'Pclass', 'Title']

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2,
                                          random_state=RNG, stratify=y)

def make_preproc(imputer_strategy, encoding, scaler):
    num_imp = SimpleImputer(strategy=imputer_strategy if imputer_strategy in
                            ['mean','median','most_frequent'] else 'median')
    scalers = {'standard': StandardScaler(), 'minmax': MinMaxScaler(),
               'robust': RobustScaler(), 'none': 'passthrough'}
    num_pipe = Pipeline([('imp', num_imp),
                         ('sc', scalers[scaler] if scaler!='none' else 'passthrough')])
    if encoding == 'onehot':
        cat_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                             ('enc', cat_enc)])
    else:  # label-style ordinal
        from sklearn.preprocessing import OrdinalEncoder
        cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                             ('enc', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                    unknown_value=-1))])
    return ColumnTransformer([('num', num_pipe, num_feats),
                              ('cat', cat_pipe, cat_feats)])

def evaluate(model, Xtr, Xte):
    model.fit(Xtr, y_tr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1] if hasattr(model,'predict_proba') else pred
    return dict(
        Accuracy=round(accuracy_score(y_te, pred), 4),
        Precision=round(precision_score(y_te, pred), 4),
        Recall=round(recall_score(y_te, pred), 4),
        F1=round(f1_score(y_te, pred), 4),
        ROC_AUC=round(roc_auc_score(y_te, proba), 4))

models = {
    'LogisticRegression': lambda: LogisticRegression(max_iter=1000, random_state=RNG),
    'RandomForest': lambda: RandomForestClassifier(n_estimators=200, random_state=RNG),
    'XGBoost': lambda: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                     eval_metric='logloss', random_state=RNG, verbosity=0),
}

# Experiment definitions (from assignment table)
experiments = {
    'Base':  dict(imp='median', enc='label',  scaler='none',     fs=False, derived=False),
    'Exp-1': dict(imp='mean',   enc='onehot',  scaler='standard', fs=False, derived=True),
    'Exp-2': dict(imp='median', enc='label',   scaler='minmax',   fs=True,  derived=True),
    'Exp-3': dict(imp='most_frequent', enc='onehot', scaler='robust', fs=True, derived=True),
}

exp_results = {}
for name, cfg in experiments.items():
    # rebuild features depending on derived flag
    dfe = build_features(df, derived=cfg['derived'])
    Xa = dfe.drop(columns=drop_cols)
    nf = num_feats if cfg['derived'] else ['Age','Fare','SibSp','Parch']
    cf = cat_feats
    Xtr_, Xte_, _, _ = train_test_split(Xa, y, test_size=0.2, random_state=RNG, stratify=y)

    def mk(imp, enc, sc, nfeats, cfeats):
        num_imp = SimpleImputer(strategy=imp)
        scalers = {'standard':StandardScaler(),'minmax':MinMaxScaler(),
                   'robust':RobustScaler(),'none':'passthrough'}
        npipe = Pipeline([('imp',num_imp),
                          ('sc', scalers[sc] if sc!='none' else 'passthrough')])
        if enc=='onehot':
            cpipe = Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                              ('enc',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
        else:
            from sklearn.preprocessing import OrdinalEncoder
            cpipe = Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                              ('enc',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))])
        return ColumnTransformer([('num',npipe,nfeats),('cat',cpipe,cfeats)])

    steps = [('pre', mk(cfg['imp'], cfg['enc'], cfg['scaler'], nf, cf))]
    if cfg['fs']:
        steps.append(('fs', SelectKBest(f_classif, k=8)))
    exp_results[name] = {}
    for mname, mfac in models.items():
        pipe = Pipeline(steps + [('clf', mfac())])
        exp_results[name][mname] = evaluate(pipe, Xtr_, Xte_)

results['experiments'] = exp_results

# ============ STEP 4: Feature Selection - before/after + importance ============
pre_full = mk = None
preproc = make_preproc('median', 'onehot', 'standard')
Xtr_t = preproc.fit_transform(X_tr)
Xte_t = preproc.transform(X_te)
feat_names = preproc.get_feature_names_out()

rf = RandomForestClassifier(n_estimators=300, random_state=RNG)
rf.fit(Xtr_t, y_tr)
imp_series = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)

# Figure 6: feature importance
plt.figure(figsize=(8, 6))
imp_series.head(12).iloc[::-1].plot(kind='barh', color='#4C72B0')
plt.title('Top 12 Feature Importances (Random Forest)', weight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('fig_importance.png', dpi=120, bbox_inches='tight'); plt.close()
results['top_features'] = imp_series.head(10).round(4).to_dict()

# before vs after FS
def acc_with(Xtr_in, Xte_in):
    m = RandomForestClassifier(n_estimators=200, random_state=RNG)
    m.fit(Xtr_in, y_tr)
    p = m.predict(Xte_in)
    return round(accuracy_score(y_te, p),4), round(f1_score(y_te,p),4)

acc_before = acc_with(Xtr_t, Xte_t)
skb = SelectKBest(f_classif, k=8).fit(Xtr_t, y_tr)
acc_after = acc_with(skb.transform(Xtr_t), skb.transform(Xte_t))
results['fs_compare'] = {'before': {'acc': acc_before[0], 'f1': acc_before[1]},
                         'after_k8': {'acc': acc_after[0], 'f1': acc_after[1]}}

# ============ Bonus: GridSearchCV ============
gs_pipe = Pipeline([('pre', make_preproc('median','onehot','standard')),
                    ('clf', RandomForestClassifier(random_state=RNG))])
param_grid = {'clf__n_estimators':[100,200,300], 'clf__max_depth':[4,6,None]}
gs = GridSearchCV(gs_pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_tr, y_tr)
gs_pred = gs.predict(X_te)
results['gridsearch'] = {'best_params': {k: str(v) for k,v in gs.best_params_.items()},
                         'cv_best': round(gs.best_score_,4),
                         'test_acc': round(accuracy_score(y_te, gs_pred),4)}

with open('results.json','w') as f:
    json.dump(results, f, indent=2)
print("DONE. All results saved.")
print(json.dumps(results['experiments'], indent=2))
