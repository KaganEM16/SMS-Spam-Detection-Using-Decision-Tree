"""
SMS Spam Tespiti - Decision Tree Sınıflandırması

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------
# 1. VERİ YÜKLEME VE KEŞİFSEL ANALİZ (EDA)
# ---------------------------------------------------------------

def load_data(filepath="SMSSpamCollection"):
    """Veri setini yükler ve temel bilgileri gösterir."""
    df = pd.read_csv(filepath, sep="\t", header=None, names=["label", "message"])
    df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})
    df["msg_length"] = df["message"].apply(len)
    df["word_count"] = df["message"].apply(lambda x: len(x.split()))
    print("=" * 55)
    print("  VERİ SETİ GENEL BİLGİLERİ")
    print("=" * 55)
    print(f"  Toplam mesaj sayısı : {len(df)}")
    print(f"  Ham  (meşru) mesaj  : {(df.label == 'ham').sum()}  "
          f"({(df.label == 'ham').mean()*100:.1f}%)")
    print(f"  Spam mesaj          : {(df.label == 'spam').sum()}  "
          f"({(df.label == 'spam').mean()*100:.1f}%)")
    print(f"  Eksik değer         : {df.isnull().sum().sum()}")
    print("=" * 55)
    return df


def eda_plots(df):
    """Keşifsel veri analizi grafikleri."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SMS Spam Collection – Keşifsel Veri Analizi", fontsize=16, fontweight="bold")

    # 1) Sınıf dağılımı (pasta)
    counts = df["label"].value_counts()
    axes[0, 0].pie(counts, labels=counts.index, autopct="%1.1f%%",
                   colors=["#4CAF50", "#F44336"], startangle=140,
                   textprops={"fontsize": 13})
    axes[0, 0].set_title("Sınıf Dağılımı (Ham / Spam)")

    # 2) Mesaj uzunluğu dağılımı
    for lbl, clr in zip(["ham", "spam"], ["#4CAF50", "#F44336"]):
        axes[0, 1].hist(df[df.label == lbl]["msg_length"], bins=50,
                        alpha=0.6, color=clr, label=lbl)
    axes[0, 1].set_xlabel("Karakter Sayısı")
    axes[0, 1].set_ylabel("Frekans")
    axes[0, 1].set_title("Mesaj Uzunluğu Dağılımı")
    axes[0, 1].legend()

    # 3) Kelime sayısı kutu grafiği
    df.boxplot(column="word_count", by="label", ax=axes[1, 0],
               boxprops=dict(color="#1565C0"),
               medianprops=dict(color="#F44336", linewidth=2))
    axes[1, 0].set_title("Kelime Sayısı (Ham vs Spam)")
    axes[1, 0].set_xlabel("Etiket")
    axes[1, 0].set_ylabel("Kelime Sayısı")
    plt.sca(axes[1, 0])
    plt.title("Kelime Sayısı (Ham vs Spam)")

    # 4) Ortalama mesaj uzunluğu çubuk grafiği
    mean_len = df.groupby("label")["msg_length"].mean()
    bars = axes[1, 1].bar(mean_len.index, mean_len.values,
                          color=["#4CAF50", "#F44336"], edgecolor="black")
    for bar, val in zip(bars, mean_len.values):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2, f"{val:.0f}",
                        ha="center", fontsize=12, fontweight="bold")
    axes[1, 1].set_title("Ortalama Mesaj Uzunluğu")
    axes[1, 1].set_ylabel("Karakter Sayısı")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✓] EDA grafikleri kaydedildi: eda_plots.png")


# ---------------------------------------------------------------
# 2. ÖZELLİK ÇIKARIMI (TF-IDF)
# ---------------------------------------------------------------

def prepare_features(df, max_features=5000):
    """TF-IDF vektörü + eğitim/test bölmesi."""
    X = df["message"]
    y = df["label_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),      # unigram + bigram
        stop_words="english",
        sublinear_tf=True
    )

    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf  = vectorizer.transform(X_test)

    print(f"\n[✓] Eğitim seti boyutu : {X_train_tf.shape}")
    print(f"[✓] Test seti boyutu   : {X_test_tf.shape}")
    return X_train_tf, X_test_tf, y_train, y_test, vectorizer


# ---------------------------------------------------------------
# 3. HİPERPARAMETRE OPTİMİZASYONU (GridSearchCV)
# ---------------------------------------------------------------

def tune_hyperparameters(X_train, y_train):
    """Karar ağacı için en iyi hiperparametreleri bulur."""
    print("\n[...] Hiperparametre araması yapılıyor (GridSearchCV)...")
    param_grid = {
        "criterion"      : ["gini", "entropy"],
        "max_depth"       : [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(dt, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    print(f"[✓] En iyi parametreler : {grid.best_params_}")
    print(f"[✓] CV F1 skoru         : {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


# ---------------------------------------------------------------
# 4. MODEL EĞİTİMİ VE DEĞERLENDİRME
# ---------------------------------------------------------------

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Modeli eğitir ve kapsamlı metriklerle değerlendirir."""
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)       # sensitivity
    f1   = f1_score(y_test, y_pred)
    spec = recall_score(y_test, y_pred, pos_label=0)   # specificity

    print("\n" + "=" * 55)
    print("  MODEL DEĞERLENDİRME SONUÇLARI")
    print("=" * 55)
    print(f"  Accuracy    (Doğruluk)  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision   (Kesinlik)  : {prec:.4f}")
    print(f"  Sensitivity (Duyarlılık): {rec:.4f}")
    print(f"  Specificity (Özgüllük)  : {spec:.4f}")
    print(f"  F1-Score                : {f1:.4f}")
    print("=" * 55)
    print("\n  Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # 5-fold Cross Validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return y_pred, y_prob, {
        "Accuracy": acc, "Precision": prec,
        "Sensitivity": rec, "Specificity": spec, "F1": f1
    }


# ---------------------------------------------------------------
# 5. GÖRSELLEŞTİRME
# ---------------------------------------------------------------

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"],
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Tahmin Edilen", fontsize=13)
    ax.set_ylabel("Gerçek", fontsize=13)
    ax.set_title("Karmaşıklık Matrisi (Confusion Matrix)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✓] confusion_matrix.png kaydedildi.")


def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#1565C0", lw=2,
            label=f"ROC Eğrisi (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate (1 - Özgüllük)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Duyarlılık)", fontsize=12)
    ax.set_title("ROC Eğrisi", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[✓] roc_curve.png kaydedildi.  AUC = {roc_auc:.4f}")


def plot_feature_importance(model, vectorizer, top_n=20):
    feat_names = np.array(vectorizer.get_feature_names_out())
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
    ax.barh(range(top_n), importances[top_idx][::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feat_names[top_idx][::-1], fontsize=11)
    ax.set_xlabel("Önem Skoru (Gini)", fontsize=12)
    ax.set_title(f"En Önemli {top_n} Özellik (TF-IDF Token)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✓] feature_importance.png kaydedildi.")


def plot_metrics_bar(metrics: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = list(metrics.keys())
    values = list(metrics.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    bars = ax.bar(names, values, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f"{val:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Skor", fontsize=12)
    ax.set_title("Değerlendirme Metrikleri", fontsize=14, fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("metrics_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✓] metrics_bar.png kaydedildi.")


def plot_decision_tree_sample(model, vectorizer, max_depth=3):
    """Karar ağacının ilk 3 seviyesini görselleştirir."""
    fig, ax = plt.subplots(figsize=(20, 8))
    feat_names = vectorizer.get_feature_names_out()
    plot_tree(model, ax=ax,
              feature_names=feat_names,
              class_names=["Ham", "Spam"],
              filled=True, rounded=True,
              max_depth=max_depth,
              fontsize=9,
              impurity=True)
    ax.set_title(f"Karar Ağacı (ilk {max_depth} seviye)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("decision_tree_visual.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[✓] decision_tree_visual.png kaydedildi.")


# ---------------------------------------------------------------
# 6. LİTERATÜR KARŞILAŞTIRMASI
# ---------------------------------------------------------------

def plot_literature_comparison(our_metrics):
    """
    Almeida vd. (2011) çalışmasından alınan Decision Tree sonuçlarıyla
    karşılaştırma.  Kaynak:
      Almeida T.A., Gómez Hidalgo J.M., Yamakami A. (2011).
      "Contributions to the Study of SMS Spam Filtering",
      ACM DocEng'11.  (J48 / C4.5 Decision Tree sonuçları)
    """
    methods    = ["Literatür\n(Almeida 2011\nJ48/C4.5)", "Bu Çalışma\n(sklearn\nDecision Tree)"]
    acc_vals   = [0.9778, our_metrics["Accuracy"]]
    f1_vals    = [0.9208, our_metrics["F1"]]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, acc_vals, width, label="Accuracy", color="#2196F3", edgecolor="black")
    b2 = ax.bar(x + width/2, f1_vals,  width, label="F1-Score",  color="#FF9800", edgecolor="black")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.4f}",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_ylim(0.85, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Skor", fontsize=12)
    ax.set_title("Literatür Karşılaştırması", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("literature_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[✓] literature_comparison.png kaydedildi.")


# ---------------------------------------------------------------
# 7. CANLI TAHMİN DEMO
# ---------------------------------------------------------------

def predict_demo(model, vectorizer):
    test_msgs = [
        "Congratulations! You've won a FREE iPhone. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been suspended. Call 0800-FREE now to verify.",
        "Can you pick up some milk on the way home please?",
        "Win £1000 cash prize! Text WIN to 85023. 150p/msg.",
    ]
    print("\n" + "=" * 60)
    print("  CANLI TAHMİN DEMOSİ")
    print("=" * 60)
    vecs = vectorizer.transform(test_msgs)
    preds = model.predict(vecs)
    probs = model.predict_proba(vecs)[:, 1]
    for msg, pred, prob in zip(test_msgs, preds, probs):
        label = "🚫 SPAM" if pred == 1 else "✅ HAM "
        print(f"\n  {label}  (spam olasılığı: {prob:.2f})")
        print(f"  \"{msg[:70]}\"")
    print("=" * 60)


# ---------------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------------

if __name__ == "__main__":
    # 1. Veri yükle
    df = load_data("SMSSpamCollection")   # ← dosya yolu gerekirse değiştirin

    # 2. EDA
    eda_plots(df)

    # 3. Özellik çıkarımı
    X_train, X_test, y_train, y_test, vectorizer = prepare_features(df, max_features=5000)

    # 4. Hiperparametre optimizasyonu
    best_model, best_params = tune_hyperparameters(X_train, y_train)

    # 5. Değerlendirme
    y_pred, y_prob, metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)

    # 6. Grafikler
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_feature_importance(best_model, vectorizer, top_n=20)
    plot_metrics_bar(metrics)
    plot_decision_tree_sample(best_model, vectorizer, max_depth=3)

    # 7. Literatür karşılaştırması
    plot_literature_comparison(metrics)

    # 8. Demo tahmin
    predict_demo(best_model, vectorizer)

    print("\n[✓] Tüm işlemler tamamlandı.")
