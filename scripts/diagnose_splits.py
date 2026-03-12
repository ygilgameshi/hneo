"""
数据划分质量诊断脚本（支持 Mode 1 / Mode 2）

检查项：
① 正样本 peptide / (peptide,hla) / (peptide,hla,tissue) 重叠
② 负样本重叠率（跨 split）
③ 假阴性率（负样本中混入正样本）
④ 正负样本长度分布对比
⑤ Val/Test 集大小 & PPV@2% 分母可靠性

支持两种负样本来源：
  A. *_with_negatives.tsv（--with_negatives）
  B. NegativeSampleCache .pkl（--pkl_dir，自动识别 train/val/test）

用法：
  # Mode 1 ablation（TSV）
  python scripts/diagnose_splits.py \
      --splits_dir output/mode1_ablation/data_splits \
      --with_negatives

  # Mode 2（PKL 缓存）
  python scripts/diagnose_splits.py \
      --splits_dir output/0227_mode2_proteome/data_splits \
      --pkl_dir    data/negative_samples \
      --pkl_stem   mode2_data_mode2_9b696ed7
"""

import argparse
import pickle
import pandas as pd
from pathlib import Path


# ────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────

def _rename(df):
    rename = {}
    if 'MHC_Restriction_Name' in df.columns: rename['MHC_Restriction_Name'] = 'hla'
    if 'Peptide'              in df.columns: rename['Peptide']               = 'peptide'
    if 'Label'                in df.columns: rename['Label']                 = 'label'
    return df.rename(columns=rename)


def load_pos_splits(splits_dir):
    """加载纯正样本 TSV（train/val/test.tsv）"""
    pos = {}
    for name in ['train', 'val', 'test']:
        p = Path(splits_dir) / f'{name}.tsv'
        if p.exists():
            pos[name] = _rename(pd.read_csv(p, sep='\t'))
    return pos


def load_neg_from_tsv(splits_dir):
    """从 *_with_negatives.tsv 加载带负样本数据"""
    neg = {}
    for name in ['train', 'val', 'test']:
        p = Path(splits_dir) / f'{name}_with_negatives.tsv'
        if p.exists():
            neg[name] = _rename(pd.read_csv(p, sep='\t'))
        else:
            print(f"  ⚠️  {p.name} 不存在，跳过")
    return neg


def load_neg_from_pkl(pkl_dir, pkl_stem):
    """
    从 NegativeSampleCache .pkl 加载带负样本数据。
    文件名格式：{pkl_stem}_train.pkl / _val.pkl / _test.pkl
    task_datasets: {task_id: DataFrame}，合并后返回。
    """
    neg = {}
    for name in ['train', 'val', 'test']:
        p = Path(pkl_dir) / f'{pkl_stem}_{name}.pkl'
        if not p.exists():
            print(f"  ⚠️  {p.name} 不存在，跳过")
            continue
        print(f"  📂 Loading {p.name} ...")
        with open(p, 'rb') as f:
            task_datasets = pickle.load(f)

        # 打印 task_id 样例（帮助判断 Mode 1/2 格式）
        sample_ids = list(task_datasets.keys())[:3]
        print(f"     task_id 样例: {sample_ids}")

        dfs = []
        for task_id, df in task_datasets.items():
            d = _rename(df.copy())
            d['task_id'] = task_id
            dfs.append(d)
        neg[name] = pd.concat(dfs, ignore_index=True)
        n_pos = (neg[name]['label'] == 1).sum()
        n_neg = (neg[name]['label'] == 0).sum()
        print(f"     pos={n_pos:,}  neg={n_neg:,}  total={len(neg[name]):,}")

    return neg


# ────────────────────────────────────────────────────────────
# ① 正样本重叠
# ────────────────────────────────────────────────────────────

def check_positive_overlap(pos):
    print("\n" + "=" * 70)
    print("① 正样本重叠检查")
    print("=" * 70)

    has_tissue = any('tissue' in df.columns or 'Host' in df.columns
                     for df in pos.values())
    tissue_col = None
    for df in pos.values():
        for c in ['tissue', 'Host']:
            if c in df.columns:
                tissue_col = c
                break

    splits = list(pos.keys())
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            pep_a   = set(pos[a]['peptide'])
            pep_b   = set(pos[b]['peptide'])
            pairs_a = set(zip(pos[a]['peptide'], pos[a]['hla']))
            pairs_b = set(zip(pos[b]['peptide'], pos[b]['hla']))

            ov_pep   = pep_a   & pep_b
            ov_pairs = pairs_a & pairs_b

            print(f"\n  {a} ∩ {b}:")
            print(f"    peptide overlap:       {len(ov_pep):>7,} "
                  f"({100*len(ov_pep)/max(len(pep_b),1):.2f}% of {b})")
            print(f"    (pep,hla) overlap:     {len(ov_pairs):>7,} "
                  f"({100*len(ov_pairs)/max(len(pairs_b),1):.2f}% of {b})")

            if tissue_col:
                t3_a = set(zip(pos[a]['peptide'], pos[a]['hla'],
                               pos[a].get(tissue_col, pd.Series())))
                t3_b = set(zip(pos[b]['peptide'], pos[b]['hla'],
                               pos[b].get(tissue_col, pd.Series())))
                ov_t3 = t3_a & t3_b
                print(f"    (pep,hla,tissue) overlap:{len(ov_t3):>6,} "
                      f"({100*len(ov_t3)/max(len(t3_b),1):.2f}% of {b})")

            if len(ov_pairs) > 0:
                print(f"    ⚠️  (peptide,hla) 完全重复 → 存在数据泄露风险！")
            elif len(ov_pep) > 0:
                print(f"    ℹ️  相同 peptide 但 HLA 不同 → 跨 HLA 共享，属正常")
            else:
                print(f"    ✓  无重叠")


# ────────────────────────────────────────────────────────────
# ② 负样本重叠
# ────────────────────────────────────────────────────────────

def check_negative_overlap(neg):
    print("\n" + "=" * 70)
    print("② 负样本重叠检查（跨 split）")
    print("=" * 70)

    if not neg:
        print("  未提供负样本数据，跳过")
        return

    neg_only = {n: df[df['label'] == 0] for n, df in neg.items()}

    splits = list(neg_only.keys())
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            pep_a = set(neg_only[a]['peptide'])
            pep_b = set(neg_only[b]['peptide'])
            ov    = pep_a & pep_b
            rate  = len(ov) / max(len(pep_b), 1)
            print(f"\n  负样本 {a} ∩ {b}:")
            print(f"    peptide overlap: {len(ov):,} / {len(pep_b):,} = {100*rate:.1f}%")
            if rate > 0.5:
                print(f"    ⚠️  重叠率 > 50%！模型在 val/test 上见过大量 train 负样本")
            elif rate > 0.2:
                print(f"    ⚠️  重叠率 > 20%，可能导致指标虚高")
            else:
                print(f"    ✓  重叠率低，负样本划分合理")


# ────────────────────────────────────────────────────────────
# ③ 假阴性检查
# ────────────────────────────────────────────────────────────

def check_false_negatives(neg, pos):
    print("\n" + "=" * 70)
    print("③ 假阴性检查（负样本中混入了正样本？）")
    print("=" * 70)

    if not neg:
        print("  未提供负样本数据，跳过")
        return

    # 全量正样本集合
    all_pos_pairs = set()
    all_pos_pep   = set()
    for df in pos.values():
        all_pos_pairs |= set(zip(df['peptide'], df['hla']))
        all_pos_pep   |= set(df['peptide'])

    for name, df in neg.items():
        neg_df    = df[df['label'] == 0]
        neg_pairs = set(zip(neg_df['peptide'], neg_df['hla']))
        neg_pep   = set(neg_df['peptide'])

        fn_pairs  = neg_pairs & all_pos_pairs
        fn_pep    = neg_pep   & all_pos_pep
        rate_p    = len(fn_pairs) / max(len(neg_pairs), 1)
        rate_pep  = len(fn_pep)   / max(len(neg_pep), 1)

        print(f"\n  {name} 负样本 ({len(neg_df):,} 条):")
        print(f"    完全假阴性 (pep,hla):  {len(fn_pairs):>7,} = {100*rate_p:.3f}%")
        print(f"    Peptide 假阴性 (任意HLA):{len(fn_pep):>6,} = {100*rate_pep:.2f}%")

        if rate_p > 0.01:
            print(f"    ⚠️  (peptide,hla) 假阴性 > 1%，负样本标签严重污染！")
        elif rate_p > 0:
            print(f"    ℹ️  少量 (peptide,hla) 假阴性 (<1%)，影响较小")
        else:
            print(f"    ✓  无 (peptide,hla) 假阴性")


# ────────────────────────────────────────────────────────────
# ④ 长度分布
# ────────────────────────────────────────────────────────────

def check_length_distribution(neg):
    print("\n" + "=" * 70)
    print("④ 正负样本长度分布对比")
    print("=" * 70)

    if not neg:
        print("  未提供负样本数据，跳过")
        return

    for name, df in neg.items():
        df = df.copy()
        df['length'] = df['peptide'].str.len()
        pos_len = df[df['label'] == 1]['length'].value_counts().sort_index()
        neg_len = df[df['label'] == 0]['length'].value_counts().sort_index()
        all_len = sorted(set(pos_len.index) | set(neg_len.index))

        print(f"\n  {name}:")
        print(f"    {'Len':>4}  {'Pos':>8}  {'Neg':>10}  {'Pos%':>7}  {'Neg%':>7}")
        for l in all_len:
            p  = pos_len.get(l, 0)
            n  = neg_len.get(l, 0)
            pp = 100 * p / max(pos_len.sum(), 1)
            np_= 100 * n / max(neg_len.sum(), 1)
            flag = "  ⚠️ " if abs(pp - np_) > 10 else ""
            print(f"    {l:>4}  {p:>8,}  {n:>10,}  {pp:>6.1f}%  {np_:>6.1f}%{flag}")


# ────────────────────────────────────────────────────────────
# ⑤ Val/Test 大小 & PPV@2% 可靠性
# ────────────────────────────────────────────────────────────

def check_val_size(pos, neg):
    print("\n" + "=" * 70)
    print("⑤ Val/Test 大小 & PPV@2% 分母可靠性")
    print("=" * 70)

    for name in ['val', 'test']:
        if name in neg:
            df    = neg[name]
            n_pos = int((df['label'] == 1).sum())
            n_neg = int((df['label'] == 0).sum())
            total = n_pos + n_neg
            n_hla = df['hla'].nunique() if 'hla' in df.columns else '?'
            k     = max(1, int(total * 0.02))
            print(f"\n  {name}:")
            print(f"    Positives: {n_pos:,}  Negatives: {n_neg:,}  HLAs: {n_hla}")
            print(f"    Ratio:     1:{n_neg//max(n_pos,1)}")
            print(f"    PPV@2% k = {k}  "
                  f"(只需 top-{k} 里有 {k} 个正样本即 PPV=1.0)")
            if k < 50:
                print(f"    ⚠️  k 极小，PPV@2% 不可靠（随机性高）！")
            elif k < 200:
                print(f"    ℹ️  k 偏小，PPV@2% 参考价值有限")
            else:
                print(f"    ✓  k={k} 足够，PPV@2% 可信")
        elif name in pos:
            n_pos = len(pos[name])
            print(f"\n  {name}: {n_pos:,} positives（无负样本文件）")


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='数据划分质量诊断')
    parser.add_argument('--splits_dir', type=str, required=True,
                        help='data_splits 目录（含 train/val/test.tsv）')
    parser.add_argument('--with_negatives', action='store_true',
                        help='加载 *_with_negatives.tsv（Mode 1 ablation）')
    parser.add_argument('--pkl_dir', type=str, default=None,
                        help='NegativeSampleCache 目录（Mode 2 pkl）')
    parser.add_argument('--pkl_stem', type=str, default=None,
                        help='pkl 文件名前缀，如 mode2_data_mode2_9b696ed7')
    args = parser.parse_args()

    print("=" * 70)
    print("数据划分质量诊断")
    print(f"splits_dir: {args.splits_dir}")
    print("=" * 70)

    # 加载正样本
    pos = load_pos_splits(args.splits_dir)
    print(f"\n正样本 splits: {list(pos.keys())}")
    for name, df in pos.items():
        n = (df['label'] == 1).sum() if 'label' in df.columns else len(df)
        hla_col = 'hla' if 'hla' in df.columns else '?'
        n_hla = df[hla_col].nunique() if hla_col != '?' else '?'
        print(f"  {name}: {n:,} pos, {n_hla} HLAs")

    # 加载负样本
    neg = {}
    if args.with_negatives:
        print("\n加载 *_with_negatives.tsv ...")
        neg = load_neg_from_tsv(args.splits_dir)
    elif args.pkl_dir and args.pkl_stem:
        print(f"\n加载 pkl 缓存: {args.pkl_dir}/{args.pkl_stem}_*.pkl ...")
        neg = load_neg_from_pkl(args.pkl_dir, args.pkl_stem)
    else:
        print("\nℹ️  未指定负样本来源，只做正样本检查")
        print("   加 --with_negatives 或 --pkl_dir + --pkl_stem 以检查负样本")

    # 五项检查
    check_positive_overlap(pos)
    check_negative_overlap(neg)
    check_false_negatives(neg, pos)
    check_length_distribution(neg)
    check_val_size(pos, neg)

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
