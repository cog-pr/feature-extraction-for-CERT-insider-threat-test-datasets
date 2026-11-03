# このコードの問い：メール専用の特徴量を除くと、週単位RFの検出性能はどう変わるか。
# 同じデータ分割で「全特徴量」と「メール専用特徴量なし」を比較する。
# 読む順番：末尾の main() → 呼び出される各関数、の順に追うと流れを理解しやすい。
# 1. CSVの列名を確認 → 2. データを分割 → 3. 2条件を学習・評価 → 4. 集計・保存。
# --run がなければ1だけ。--run --nrun 20 なら2条件×20組＝合計40回学習する。
"""Compare weekly RF with all features and with email-specific features removed.

No arguments: inspect the CSV header only.  --run: train both conditions.
The original split/evaluation helpers are reused; original results are untouched.
"""

# argparse：コマンドの引数を受け取る。gzip：圧縮CSVを読む。
# pickle：Pythonの配列・辞書等を保存。json/csv：設定や結果を読みやすい形式で保存。
import argparse
import csv
from datetime import datetime
import gc
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import platform


# 内部で使う名前と、結果の表に表示する日本語名。
# all / no_email は、比較する2条件の識別名。
METRICS = ("recall", "fpr", "precision", "f1")
LABELS = {"all": "全特徴量", "no_email": "メール専用特徴量なし"}
METRIC_LABELS = {"recall": "検出率", "fpr": "誤検知率", "precision": "適合率", "f1": "F1"}


def is_email_column(name):
    # 列名がメール専用特徴量ならTrue、それ以外ならFalseを返す。
    # feature_extraction.py の f_calc() の命名規則に対応する4区分：
    # 全時間帯 / 勤務時間内 / 時間外 / 週末。
    groups = ("email", "workhouremail", "afterhouremail", "weekendemail")
    # n_email等は件数、email_*等は宛先数・添付ファイル等の集計値。
    # any()：4区分のどれかに一致すればTrue。startswith()：指定文字列で始まるか。
    # 全行動数などの共通集計は除外しないので、メール行動の影響が一切なくなるわけではない。
    return any(name == "n_" + group or name.startswith(group + "_") for group in groups)


def binary_counts(matrix):
    # 混同行列（行＝正解、列＝予測）から、TN・FP・FN・TPの順に件数を取り出す。
    # TN：正常を正常と予測、FP：正常を不正と予測。
    # FN：不正を正常と予測、TP：不正を不正と予測。
    # 著者のユーザー評価では正解にシナリオ番号1〜4が残るため、2行2列とは限らない。
    # このデータでは0行・0列が正常、残りをまとめて不正として集計する。
    # [1:]は「先頭以外すべて」、sum()はその範囲の合計。
    return (int(matrix[0, 0]), int(matrix[0, 1:].sum()),
            int(matrix[1:, 0].sum()), int(matrix[1:, 1:].sum()))


def metrics_from_counts(counts):
    # 件数から4指標を計算する。ここで返す割合は0〜1（0.8なら80％）。
    tn, fp, fn, tp = counts
    def divide(a, b):
        # 不正と予測した件数が0等の場合、0除算を避けるため指標を0とする。
        return a / b if b else 0.0
    # 検出率＝TP/(TP+FN)：実際に不正な対象のうち、正しく不正と予測した割合。
    # 誤検知率＝FP/(FP+TN)：実際に正常な対象のうち、誤って不正と予測した割合。
    # 適合率＝TP/(TP+FP)：不正と予測した対象のうち、実際に不正だった割合。
    # F1＝2TP/(2TP+FP+FN)：検出率と適合率の調和平均。
    # zip()で指標名と値を順に組み合わせ、dict()で辞書にする。
    return dict(zip(METRICS, (divide(tp, tp + fn), divide(fp, fp + tn),
                              divide(tp, tp + fp), divide(2 * tp, 2 * tp + fp + fn))))


def summarize(rows, out):
    # rows：完了した各回の結果を並べたリスト。1組につき2条件×2評価単位＝4行。
    # out：保存先フォルダ。CSVには各回の値、mdには平均・標準偏差・条件間の差を出す。
    import numpy as np

    # utf-8-sigはBOM付きUTF-8。WindowsのExcel等でも日本語を読みやすくする。
    with (out / "metrics_per_run.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # 以下でMarkdownの表を文字列として組み立てる。
    lines = ["# メール専用特徴量の有無による比較", "",
             "各回で同じ学習ユーザー・評価データ・RF設定を使用。数値は各回で計算後に平均。",
             "±は回ごとの標準偏差（標本標準偏差）であり、信頼区間ではない。",
             "割合は％、差はパーセントポイント。差＝メール専用特徴量なし－全特徴量。", ""]
    for unit, title in (("week", "週データ単位"), ("user", "ユーザー単位")):
        # 同じ週データを、ユーザー×週の1行単位と、ユーザー単位の両方で評価する。
        lines += ["## " + title, "", "| 指標 | 全特徴量：平均±標準偏差 | メール専用特徴量なし：平均±標準偏差 | 同じ回同士の差：平均±標準偏差 |",
                  "|---|---:|---:|---:|"]
        for metric in METRICS:
            values = {}
            for condition in LABELS:
                # 対象の条件・評価単位だけを取り出し、乱数番号runの順に並べる。
                # 両条件の順を揃えることで、同じ回同士の差を計算できる。
                selected = sorted((r for r in rows if r["unit"] == unit and r["condition"] == condition),
                                  key=lambda r: r["run"])
                # 0〜1の割合を％に変換する。各回で計算済みの指標を平均する方式。
                # 全回の混同行列を合算してから指標を計算する方式とは異なる。
                values[condition] = np.array([r[metric] * 100 for r in selected])
            def describe(v):
                # ddof=1は標本標準偏差。回ごとのばらつきであり、95％信頼区間ではない。
                # 1回だけの動作確認では、標本標準偏差は算出できない。
                return "{:.2f} ± {}".format(v.mean(), "{:.2f}".format(v.std(ddof=1)) if len(v) > 1 else "算出不可")
            # 配列同士の引き算は、同じ位置（同じrun）の値同士を引く。
            # 差は「メール専用特徴量なし－全特徴量」。誤検知率は低い方がよい点に注意。
            lines.append("| {} | {} | {} | {} |".format(
                METRIC_LABELS[metric], describe(values["all"]), describe(values["no_email"]),
                describe(values["no_email"] - values["all"])))
        lines.append("")
    lines += ["## 解釈上の注意", "",
              "- 全行動数など、メール行動も含む共通集計は両条件に残る。メールログ全廃の実験ではない。",
              "- ユーザー単位は著者コードの定義を使用。不正ユーザーは不正な週を1件以上検出した場合に検出と数える。",
              "- 適合率などの分母が0の場合は0として記録する。混同行列の件数もCSVに保存する。",
              "- RFパラメータと標準判定は固定。各条件での再調整は行わない。",
              "- 性能差だけからメール情報一般の必要性や、統計的有意差を断定しない。", ""]
    (out / "比較結果.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    # 実行の中心。__file__はこのスクリプト自身のパス。
    # スクリプトのあるTNSM2020を基準にするので、repo直下からでも実行できる。
    base = Path(__file__).resolve().parent
    # ① コマンドの引数を受け取る。
    # --run：学習を実行する指定。--nrun：比較を繰り返す組数。--jobs：並列処理数。
    # --data / --params / --output を省略すると、下に書いた標準の場所を使う。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Train; without this flag only inspect the header")
    parser.add_argument("--nrun", type=int, default=20, help="Paired repetitions, seeds 0..nrun-1")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--data", type=Path, default=base / "data" / "weekr5.2.csv.gz")
    parser.add_argument("--params", type=Path, default=base / "params.pkl")
    parser.add_argument("--output", type=Path, help="New directory only; existing directories are refused")
    args = parser.parse_args()
    if args.nrun < 1 or args.jobs < 1:
        parser.error("--nrun and --jobs must be positive")
    # ② 圧縮CSVの先頭1行（列名）だけを読む。この時点では全データを読み込まない。
    with gzip.open(str(args.data), "rt", encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle))
    mail_columns = [c for c in columns if is_email_column(c)]
    if not mail_columns or any(c not in columns for c in ("user", "week", "insider")):
        parser.error("Expected user/week/insider and email-specific columns")
    print("CSV columns: {}; email-specific columns: {}".format(len(columns), len(mail_columns)), flush=True)
    print("Examples: " + ", ".join(mail_columns[:8]), flush=True)
    if not args.run:
        # --runなしならここで終了。列の確認だけで、学習や保存は行わない。
        print("Inspection only. No training or files written. Add --run to train.")
        return

    # 学習を行う場合だけ、数値処理・データ処理・RF・著者の補助コードを読み込む。
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import clf_helpers

    # ③ 以前の実験と同じ週単位RFの設定と、全データを読み込む。
    # ["RF"]["week"]で、保存された複数設定の中から対象のものを取り出す。
    with args.params.open("rb") as handle:
        rf_params = pickle.load(handle)["RF"]["week"]
    data = pd.read_csv(args.data)
    # ④ 結果は日時付きの新規フォルダへ保存する。以前の結果を上書きしない。
    # exist_ok=False：同名フォルダが既にある場合はエラーにして止める。
    output = args.output or base / "res" / ("email_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    # 後から条件を確認できるよう、設定・乱数・環境の版を記録する。
    # SHA256はparams.pklの内容を識別する値。同じ設定ファイルか照合するために保存する。
    settings = {"nrun": args.nrun, "seeds": list(range(args.nrun)), "jobs": args.jobs,
                "data": str(args.data.resolve()), "params": rf_params,
                "params_sha256": hashlib.sha256(args.params.read_bytes()).hexdigest(),
                "data_rows": len(data), "csv_columns": columns, "email_columns": mail_columns,
                "split": "by_user_time", "train_week_fraction": 0.5, "train_users": 400,
                "normalization": "StandardScaler", "decision": "RF.predict (positive probability > 0.5)",
                "python": platform.python_version(),
                "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")}}
    (output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "email_columns.txt").write_text("\n".join(mail_columns) + "\n", encoding="utf-8")
    # 著者のユーザー単位評価で使う並列処理数も、指定した値に合わせる。
    clf_helpers.num_cores = args.jobs
    rows = []
    print("Output: " + str(output.resolve()), flush=True)
    for run in range(args.nrun):
        # ⑤ 1組につき1回だけ分割する。run=0,1,...を乱数の種に使う。
        # 著者コードがNumPyとPythonの乱数を設定し、正常な学習ユーザーを選び直す。
        # 今のデータでは前半37週間の不正ユーザー34人＋無作為抽出した366人で学習する。
        # 評価は後半期間。学習に含めた既知の不正ユーザー34人は評価から除外する。
        # StandardScalerは学習データで平均・標準偏差を求め、その値で評価データも変換する。
        split = clf_helpers.split_data(data, random_state=run, dname="r5.2", normalization="StandardScaler",
                                       by_user_time=True, by_user_time_trainper=0.5, limit_ntrain_user=400)
        # x_colsはメタ情報・正解ラベル・定数列等を除いた、実際の入力特徴量名。
        # keepは列ごとの採否：メール専用列がFalse、それ以外がTrueの配列。
        keep = np.array([not is_email_column(c) for c in split["x_cols"]])
        if keep.all() or not keep.any():
            raise ValueError("Feature selection did not produce two meaningful conditions")
        # 同じ分割で比較したことを後から確認できるよう、ユーザーと元の行番号も保存する。
        # feature_columnsとresultsは空の辞書で用意し、下のループで2条件分を追加する。
        pair = {"run": run, "train_users": split["train_users"], "test_users": split["test_users"],
                "train_row_index": split["train_info"].index.to_numpy(),
                "test_row_index": split["test_info"].index.to_numpy(),
                "feature_columns": {}, "results": {}}
        for condition in LABELS:
            # ⑥ まず全特徴量、次にメール専用特徴量なし、の順で実行する。
            # 全特徴量のmaskは全部True。除外条件のmaskは先ほど作ったkeep。
            mask = np.ones(len(keep), dtype=bool) if condition == "all" else keep
            # [:, mask]は「行はすべて、列はTrueの位置だけ」を取り出すNumPyの書き方。
            # 学習・評価の両方に同じmaskを使う。行の選び方や正解ラベルは変えない。
            # 標準化は列ごとに独立なので、標準化後に列を除いても残る列の値は変わらない。
            x_train = split["x_train"] if condition == "all" else split["x_train"][:, mask]
            x_test = split["x_test"] if condition == "all" else split["x_test"][:, mask]
            # 条件ごとに新しいRFを作る。**rf_paramsは辞書の各項目を引数として渡す記法。
            # 同じrunでは、木の本数等の設定と乱数を揃え、入力特徴量だけを変える。
            # 入力が違うため、学習後の木や予測が同じになるという意味ではない。
            clf = RandomForestClassifier(**rf_params)
            clf.set_params(random_state=run, n_jobs=args.jobs)
            # 著者コードで学習・予測・混同行列・ユーザー単位の評価を行う。
            # 戻り値は「学習済みモデル、学習側の結果、評価側の結果」の3つ。
            # この比較では予測・評価結果を保存するので、モデルの戻り値は _ で受ける。
            # y_*_binは正常0/不正1、y_orgには元のシナリオ番号を渡す。
            # predict()の標準判定を使うため、閾値を独自に調整する処理はない。
            _, train, test = clf_helpers.do_classification(
                clf, x_train, split["y_train_bin"], x_test, split["y_test_bin"],
                y_org={"train": split["y_train"], "test": split["y_test"]}, by_user=True, split_output=split)
            pair["feature_columns"][condition] = [c for c, selected in zip(split["x_cols"], mask) if selected]
            pair["results"][condition] = {"train": train, "test_in": test}
            for unit, key in (("week", "bin"), ("user", "test_users")):
                # ⑦ 2種類の混同行列から4指標を計算する。
                # week：各ユーザーの1週間（CSVの1行）を正しく分類できたか。
                # user：不正ユーザーの不正な週を1件以上検出したか（著者の評価定義）。
                # 正常ユーザーは、1件でも不正と予測されるとユーザー単位の誤検知になる。
                counts = binary_counts(test["cms"][key])
                rows.append(dict(run=run, condition=condition, unit=unit, n_features=int(mask.sum()),
                                 tn=counts[0], fp=counts[1], fn=counts[2], tp=counts[3],
                                 **metrics_from_counts(counts)))
            print("Run {}/{}: {} done ({} features)".format(run + 1, args.nrun, condition, mask.sum()), flush=True)
            # 大きな配列等への参照を減らし、不要なオブジェクトの回収を促す。
            del clf, x_train, x_test
            gc.collect()
        # ⑧ 2条件が完了した組ごとに、予測・評価結果と使用列名を保存する。
        # :02dは0を00、1を01とする表示指定。xbは既存ファイルがあれば作成を拒否する。
        with (output / "run_{:02d}.pickle".format(run)).open("xb") as handle:
            pickle.dump(pair, handle, protocol=4)
        # 完了した組までの集計を更新する。途中で止まった場合も、完了分を確認できる。
        summarize(rows, output)
        del split, pair, train, test
        gc.collect()
    # 全組の完了後にだけ作る印。このファイルがなければ本実験完了とは扱わない。
    (output / "COMPLETED.txt").write_text("Completed {} paired runs.\n".format(args.nrun), encoding="utf-8")
    print("Complete: " + str((output / "比較結果.md").resolve()), flush=True)


if __name__ == "__main__":
    # このファイルをコマンドで直接実行したときにmain()を呼ぶ。
    # 他のコードからimportしただけなら、実験は始まらない。
    main()
