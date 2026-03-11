import pandas as pd
import re
import glob
from pathlib import Path
import json
import numpy as np


def load_headers_from_first_file(first_file):
    """
    从第一个文件正确提取两级表头，处理重复列名
    """
    print(f"从 {Path(first_file).name} 读取表头...")

    # 读取前两行
    with open(first_file, 'r', encoding='utf-8') as f:
        header_line1 = f.readline().strip().split(',')
        header_line2 = f.readline().strip().split(',')

    print(f"  一级表头: {len(header_line1)} 列")
    print(f"  二级表头: {len(header_line2)} 列")

    # 构建唯一的列名（处理重复）
    unique_columns = []
    column_info = []
    name_counter = {}

    for i, (level1, level2) in enumerate(zip(header_line1, header_line2)):
        level1 = level1.strip()
        level2 = level2.strip()

        # 创建唯一的列名
        # 如果level2重复（如多个"Name"），添加level1前缀
        if level2 in name_counter:
            name_counter[level2] += 1
            unique_name = f"{level1}_{level2}_{name_counter[level2]}"
        else:
            name_counter[level2] = 0
            unique_name = f"{level1}_{level2}" if level2 else f"{level1}_col{i}"

        unique_columns.append(unique_name)

        column_info.append({
            'index': i,
            'level1': level1,
            'level2': level2,
            'unique_name': unique_name,
            'combined': f"{level1}|{level2}"
        })

    return unique_columns, column_info


def find_key_column_indices(column_info):
    """
    精确定位关键列的索引（改进版：支持模糊匹配）
    """
    indices = {}

    print("\n查找关键列...")

    # === 辅助函数：模糊匹配 ===
    def fuzzy_match(text, target):
        """大小写不敏感的包含匹配"""
        if not text or not target:
            return False
        return target.lower() in text.lower()

    # === 1. 找Species列 ===
    for col in column_info:
        if fuzzy_match(col['level1'], 'Epitope') and col['level2'] == 'Species':
            indices['Species'] = col['index']
            print(f"  ✓ Species: 列 {col['index']}")
            break

    # === 2. 找MHC Restriction Name ===
    for col in column_info:
        if fuzzy_match(col['level1'], 'MHC Restriction') and col['level2'] == 'Name':
            indices['MHC_Restriction_Name'] = col['index']
            print(f"  ✓ MHC Restriction Name: 列 {col['index']}")
            break

    # === 3. 找Epitope Name ===
    for col in column_info:
        if fuzzy_match(col['level1'], 'Epitope') and col['level2'] == 'Name':
            indices['Epitope_Name'] = col['index']
            print(f"  ✓ Epitope Name: 列 {col['index']}")
            break

    # === 4. 找其他关键列（改进版：逐个精确查找） ===

    # Source Molecule (在Epitope下)
    for col in column_info:
        if 'Epitope' in col['level1'] and col['level2'] == '"Source Molecule"':
            indices['Source_Molecule'] = col['index']
            print(f"  ✓ Source_Molecule: 列 {col['index']}")
            break

    # Source Molecule IRI
    for col in column_info:
        if 'Epitope' in col['level1'] and col['level2'] == '"Source Molecule IRI"':
            indices['Source_Molecule_IRI'] = col['index']
            print(f"  ✓ Source_Molecule_IRI: 列 {col['index']}")
            break

    # Source Organism
    for col in column_info:
        if 'Epitope' in col['level1'] and col['level2'] == '"Source Organism"':
            indices['Source_Organism'] = col['index']
            print(f"  ✓ Source_Organism: 列 {col['index']}")
            break

    # Starting Position
    for col in column_info:
        if 'Epitope' in col['level1'] and col['level2'] == '"Starting Position"':
            indices['Epitope_Start'] = col['index']
            print(f"  ✓ Epitope_Start: 列 {col['index']}")
            break

    # Ending Position
    for col in column_info:
        if 'Epitope' in col['level1'] and col['level2'] == '"Ending Position"':
            indices['Epitope_End'] = col['index']
            print(f"  ✓ Epitope_End: 列 {col['index']}")
            break

    # Epitope IRI
    for col in column_info:
        if 'Epitope' in col['level1'] and col['level2'] == '"Epitope IRI"':
            indices['Epitope_IRI'] = col['index']
            print(f"  ✓ Epitope_IRI: 列 {col['index']}")
            break

    # PMID (在Reference下)
    for col in column_info:
        if 'Reference' in col['level1'] and col['level2'] == 'PMID':
            indices['PMID'] = col['index']
            print(f"  ✓ PMID: 列 {col['index']}")
            break

    # Method (在Assay下)
    for col in column_info:
        if 'Assay' in col['level1'] and col['level2'] == 'Method':
            indices['Assay_Method'] = col['index']
            print(f"  ✓ Assay_Method: 列 {col['index']}")
            break

    # Response measured (在Assay下)
    for col in column_info:
        if 'Assay' in col['level1'] and col['level2'] == '"Response measured"':
            indices['Response_Measured'] = col['index']
            print(f"  ✓ Response_Measured: 列 {col['index']}")
            break

    # Qualitative Measurement (在Assay下)
    for col in column_info:
        if 'Assay' in col['level1'] and col['level2'] == '"Qualitative Measurement"':
            indices['Qualitative_Measurement'] = col['index']
            print(f"  ✓ Qualitative_Measurement: 列 {col['index']}")
            break

    # Source Tissue (在Antigen Presenting Cell下)
    for col in column_info:
        if 'Antigen Presenting Cell' in col['level1'] and col['level2'] == '"Source Tissue"':
            indices['Source_Tissue'] = col['index']
            print(f"  ✓ Source_Tissue: 列 {col['index']}")
            break

    # Disease (在in vivo Process下)
    for col in column_info:
        if 'in vivo Process' in col['level1'] and col['level2'] == 'Disease':
            indices['Disease'] = col['index']
            print(f"  ✓ Disease: 列 {col['index']}")
            break

    # MHC Class (在MHC Restriction下)
    for col in column_info:
        if 'MHC Restriction' in col['level1'] and col['level2'] == 'Class':
            indices['MHC_Class'] = col['index']
            print(f"  ✓ MHC_Class: 列 {col['index']}")
            break

    # Molecule Parent IRI (在Epitope下) → UniProt ID 来源
    # IEDB原始表头: Epitope | Molecule Parent IRI
    # 对应clean data新增列: UniProt_ID，用于负样本配对采样
    for col in column_info:
        if 'Epitope' in col['level1'] and 'Molecule Parent IRI' in col['level2']:
            indices['Molecule_Parent_IRI'] = col['index']
            print(f"  ✓ Molecule_Parent_IRI: 列 {col['index']}")
            break
    if 'Molecule_Parent_IRI' not in indices:
        # 宽松备用匹配：有些版本表头带引号
        for col in column_info:
            if 'Epitope' in col['level1'] and 'parent' in col['level2'].lower():
                indices['Molecule_Parent_IRI'] = col['index']
                print(f"  ✓ Molecule_Parent_IRI (备用匹配): 列 {col['index']}")
                break
    if 'Molecule_Parent_IRI' not in indices:
        print(f"  ⚠ Molecule_Parent_IRI: 未找到（负样本配对采样将降级为全蛋白质组随机切割）")

    print(f"\n总共找到 {len(indices)} 个关键列")

    # 检查必需列
    required = ['Species', 'MHC_Restriction_Name', 'Epitope_Name']
    missing = [k for k in required if k not in indices]

    if missing:
        print(f"\n⚠ 警告：缺少必需列: {missing}")
        return None

    return indices


def debug_show_all_columns(column_info):
    """
    显示所有列的分组和名称
    """
    print("\n" + "=" * 80)
    print("所有列的完整信息:")
    print("=" * 80)

    current_group = None
    group_columns = []

    for col in column_info:
        # 检测是否进入新分组
        if col['level1'] and col['level1'] != current_group:
            # 打印前一个分组
            if current_group and group_columns:
                print(f"\n【{current_group}】分组下的列:")
                for idx, name in group_columns:
                    print(f"  列 {idx:3d}: {name}")

            current_group = col['level1']
            group_columns = []

        if col['level2']:
            group_columns.append((col['index'], col['level2']))

    # 打印最后一个分组
    if current_group and group_columns:
        print(f"\n【{current_group}】分组下的列:")
        for idx, name in group_columns:
            print(f"  列 {idx:3d}: {name}")

    print("\n" + "=" * 80)



def process_file_with_unique_names(file_path, unique_columns, column_indices, is_first_file=True):
    """
    使用唯一列名处理文件
    """
    print(f"\n{'=' * 80}")
    print(f"处理: {Path(file_path).name}")
    print(f"{'=' * 80}")

    if is_first_file:
        # 第一个文件：跳过前两行表头，使用唯一列名
        df = pd.read_csv(file_path, skiprows=2, names=unique_columns, low_memory=False)
    else:
        # 后续文件：直接读取，使用唯一列名
        df = pd.read_csv(file_path, names=unique_columns, low_memory=False)

    print(f"原始数据: {len(df):,} 行 × {len(df.columns)} 列")

    # 使用列索引提取数据
    data_dict = {}

    for key, idx in column_indices.items():
        if idx < len(df.columns):
            col_name = unique_columns[idx]
            data_dict[key] = df[col_name].values

    df_extracted = pd.DataFrame(data_dict)

    print(f"提取关键列: {list(df_extracted.columns)}")

    # === 1. 过滤Species ===
    if 'Species' not in df_extracted.columns:
        print("错误：未找到Species列！")
        return None

    before_filter = len(df_extracted)
    df_filtered = df_extracted[
        df_extracted['Species'].astype(str).str.contains('Mus musculus', case=False, na=False)
    ].copy()

    print(f"过滤Mus musculus: {len(df_filtered):,} / {before_filter:,} ({len(df_filtered) / before_filter * 100:.1f}%)")

    if len(df_filtered) == 0:
        print("警告：过滤后无数据")
        return None

    # === 2. 过滤HLA格式 ===
    if 'MHC_Restriction_Name' not in df_filtered.columns:
        print("错误：未找到MHC_Restriction_Name列！")
        return None

    # 显示MHC列的样例数据（前20个独特值）
    print(f"\nMHC Restriction Name样例（前20个独特值）:")
    mhc_samples = df_filtered['MHC_Restriction_Name'].dropna().astype(str).unique()[:20]
    for i, mhc in enumerate(mhc_samples, 1):
        print(f"  {i:2d}. '{mhc}'")

    # 小鼠MHC-I标准格式：H2-Kb, H2-Db, H2-Ld, H2-Kd, H2-Kk 等
    hla_pattern = r'^H-?2-[A-Za-z]\w*$'

    before_hla_filter = len(df_filtered)

    # 清理MHC列（去除前后空格）
    df_filtered['MHC_Restriction_Name'] = df_filtered['MHC_Restriction_Name'].astype(str).str.strip()

    df_filtered['_hla_valid'] = df_filtered['MHC_Restriction_Name'].apply(
        lambda x: bool(re.match(hla_pattern, x)) if pd.notna(x) and x != 'nan' else False
    )

    n_valid = df_filtered['_hla_valid'].sum()
    n_invalid = (~df_filtered['_hla_valid']).sum()

    print(f"\nHLA格式验证:")
    print(f"  ✓ 符合标准格式: {n_valid:,} ({n_valid / before_hla_filter * 100:.1f}%)")
    print(f"  ✗ 不符合格式: {n_invalid:,} ({n_invalid / before_hla_filter * 100:.1f}%)")

    # 显示不符合格式的示例
    if n_invalid > 0:
        invalid_examples = df_filtered[~df_filtered['_hla_valid']]['MHC_Restriction_Name'].unique()[:10]
        print(f"\n不符合格式的示例:")
        for ex in invalid_examples:
            print(f"  ✗ '{ex}'")

    df_filtered = df_filtered[df_filtered['_hla_valid']].copy()
    df_filtered.drop('_hla_valid', axis=1, inplace=True)

    print(f"\n过滤HLA格式: {len(df_filtered):,} / {before_hla_filter:,}")

    if len(df_filtered) == 0:
        print("警告：HLA过滤后无数据")
        return None

    # === 3. 处理肽段 ===
    df_filtered.rename(columns={'Epitope_Name': 'Peptide'}, inplace=True)

    if 'Peptide' in df_filtered.columns:
        # 清理肽段序列
        df_filtered['Peptide'] = df_filtered['Peptide'].astype(str).str.strip()

        # 过滤掉非标准氨基酸序列
        df_filtered = df_filtered[
            df_filtered['Peptide'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', na=False)
        ].copy()

        # 计算长度
        df_filtered['Peptide_Length'] = df_filtered['Peptide'].str.len()

        # 只保留8-15mer
        before_length = len(df_filtered)
        df_filtered = df_filtered[
            (df_filtered['Peptide_Length'] >= 8) &
            (df_filtered['Peptide_Length'] <= 15)
            ].copy()

        print(f"过滤肽段长度(8-15mer): {len(df_filtered):,} / {before_length:,}")

    if len(df_filtered) == 0:
        print("警告：肽段过滤后无数据")
        return None

    # === 4. 推断组织 ===
    df_filtered = infer_tissue_from_disease(df_filtered)

    # === 5. 提取 UniProt ID ===
    # 从 Molecule_Parent_IRI（如 http://www.uniprot.org/uniprot/P29996）提取纯 ID
    if 'Molecule_Parent_IRI' in df_filtered.columns:
        df_filtered['UniProt_ID'] = (
            df_filtered['Molecule_Parent_IRI']
            .astype(str)
            .str.extract(r'uniprot\.org/uniprot/([A-Z0-9]+)', expand=False)
        )
        n_with_uniprot = df_filtered['UniProt_ID'].notna().sum()
        pct = n_with_uniprot / len(df_filtered) * 100
        print(f"UniProt ID 覆盖率: {n_with_uniprot:,} / {len(df_filtered):,} ({pct:.1f}%)")
    else:
        df_filtered['UniProt_ID'] = None
        print("⚠ UniProt_ID: 列不存在，已填 None（后续负样本生成将降级为全蛋白质组随机切割）")

    # === 6. 标记为正样本 ===
    df_filtered['Label'] = 1

    print(f"\n✓ 最终保留: {len(df_filtered):,} 行")

    # === 7. 统计信息 ===
    if len(df_filtered) > 0:
        print(f"\n数据统计:")
        print(f"  独特HLA: {df_filtered['MHC_Restriction_Name'].nunique()}")
        print(f"  独特肽段: {df_filtered['Peptide'].nunique():,}")
        print(f"  独特组织: {df_filtered['Inferred_Tissue'].nunique()}")

        # UniProt 和位置信息覆盖率（用于负样本配对采样）
        if 'UniProt_ID' in df_filtered.columns:
            n_uniprot = df_filtered['UniProt_ID'].notna().sum()
            n_with_pos = (
                df_filtered['Epitope_Start'].notna() &
                df_filtered['Epitope_End'].notna()
            ).sum() if ('Epitope_Start' in df_filtered.columns and
                        'Epitope_End' in df_filtered.columns) else 0
            print(f"  UniProt ID 覆盖: {n_uniprot:,} ({n_uniprot/len(df_filtered)*100:.1f}%)"
                  f"  ← 可做配对负样本采样")
            print(f"  含坐标信息: {n_with_pos:,} ({n_with_pos/len(df_filtered)*100:.1f}%)"
                  f"  ← 可做重叠过滤")

        # HLA分布（Top 10）
        print(f"\nTop 10 HLA分布:")
        for i, (hla, count) in enumerate(df_filtered['MHC_Restriction_Name'].value_counts().head(10).items(), 1):
            pct = count / len(df_filtered) * 100
            print(f"  {i:2d}. {hla}: {count:,} ({pct:.1f}%)")

        # 组织分布（Top 10）
        print(f"\nTop 10 组织分布:")
        for i, (tissue, count) in enumerate(df_filtered['Inferred_Tissue'].value_counts().head(10).items(), 1):
            pct = count / len(df_filtered) * 100
            print(f"  {i:2d}. {tissue}: {count:,} ({pct:.1f}%)")

        # 肽段长度分布
        print(f"\n肽段长度分布:")
        for length in sorted(df_filtered['Peptide_Length'].unique()):
            count = (df_filtered['Peptide_Length'] == length).sum()
            pct = count / len(df_filtered) * 100
            print(f"  {int(length)}-mer: {count:,} ({pct:.1f}%)")

    return df_filtered


def infer_tissue_from_disease(df):
    """
    从Disease列推断Tissue（组织）
    """
    tissue_mapping = {
        # 肿瘤
        'melanoma': 'Skin',
        'lung cancer': 'Lung',
        'nsclc': 'Lung',
        'sclc': 'Lung',
        'hepatocellular': 'Liver',
        'hepatitis': 'Liver',
        'breast cancer': 'Breast',
        'breast carcinoma': 'Breast',
        'colorectal': 'Colon',
        'colon cancer': 'Colon',
        'pancreatic': 'Pancreas',
        'ovarian': 'Ovary',
        'prostate': 'Prostate',
        'glioblastoma': 'Brain',
        'glioma': 'Brain',
        'leukemia': 'Blood',
        'lymphoma': 'Lymph',
        'multiple myeloma': 'Blood',
        'cervical': 'Cervix',
        'gastric': 'Stomach',
        'esophageal': 'Esophagus',
        'bladder': 'Bladder',
        'kidney': 'Kidney',
        'renal': 'Kidney',

        # 病毒
        'hiv': 'Blood',
        'influenza': 'Lung',
        'ebv': 'Lymph',
        'cmv': 'Multiple',
        'sars': 'Lung',
        'covid': 'Lung',
        'hepatitis b': 'Liver',
        'hepatitis c': 'Liver',
        'hepatitis delta': 'Liver',
    }

    def map_to_tissue(row):
        # 优先使用Source_Tissue
        if 'Source_Tissue' in row:
            tissue = str(row['Source_Tissue']).strip()
            if tissue and tissue.lower() not in ['nan', 'none', '']:
                return tissue

        # 从Disease推断
        if 'Disease' in row:
            disease = str(row['Disease']).lower().strip()
            if disease and disease != 'nan':
                for keyword, tissue in tissue_mapping.items():
                    if keyword in disease:
                        return tissue

        return 'Unknown'

    df['Inferred_Tissue'] = df.apply(map_to_tissue, axis=1)

    return df


def save_in_chunks(df, file_basename, output_dir='cleaned_data', chunk_size=300000):
    """
    分块保存数据
    """
    if df is None or len(df) == 0:
        return []

    Path(output_dir).mkdir(exist_ok=True)

    n_chunks = (len(df) - 1) // chunk_size + 1
    chunk_files = []

    print(f"\n保存数据（{n_chunks} 块）:")

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(df))

        df_chunk = df.iloc[start:end]

        chunk_file = Path(output_dir) / f"{file_basename}_chunk_{i:03d}.csv"
        df_chunk.to_csv(chunk_file, index=False)

        chunk_files.append(str(chunk_file))
        print(f"  Chunk {i}: {len(df_chunk):,} 行 → {chunk_file.name}")

    return chunk_files


def process_all_files_final(
        file_pattern='mhc_ligand_full_*.csv',
        output_dir='cleaned_data'
):
    """
    最终修复版：处理所有文件
    """
    files = sorted(glob.glob(file_pattern))

    if not files:
        print("错误：未找到文件！")
        return None

    print(f"找到 {len(files)} 个文件\n")

    # === 1. 从第一个文件获取表头 ===
    first_file = files[0]
    unique_columns, column_info = load_headers_from_first_file(first_file)

    # === 调试：显示所有列 ===
    # debug_show_all_columns(column_info)

    # === 2. 定位关键列 ===
    column_indices = find_key_column_indices(column_info)

    if column_indices is None:
        print("错误：无法定位关键列！")
        return None

    # === 3. 处理所有文件 ===
    all_stats = {}

    for i, file_path in enumerate(files):
        try:
            is_first = (i == 0)

            df = process_file_with_unique_names(
                file_path,
                unique_columns,
                column_indices,
                is_first_file=is_first
            )

            if df is not None and len(df) > 0:
                # 保存
                file_basename = f'cleaned_{i:02d}'
                chunk_files = save_in_chunks(df, file_basename, output_dir)

                # 统计
                all_stats[file_basename] = {
                    'source_file': Path(file_path).name,
                    'n_samples': int(len(df)),
                    'n_hla': int(df['MHC_Restriction_Name'].nunique()),
                    'n_peptides': int(df['Peptide'].nunique()),
                    'n_tissues': int(df['Inferred_Tissue'].nunique()),
                    'n_chunks': len(chunk_files),
                }
            else:
                print(f"  ⚠ {Path(file_path).name} 处理后无有效数据")

        except Exception as e:
            print(f"\n❌ 处理 {file_path} 出错:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            continue

    # === 4. 保存总体统计 ===
    print(f"\n{'=' * 80}")
    print("✓ 所有文件处理完成！")
    print(f"{'=' * 80}")

    if all_stats:
        total_samples = sum(s['n_samples'] for s in all_stats.values())
        total_hlas = len(set().union(*[
            set() for s in all_stats.values()
        ]))

        print(f"\n总体统计:")
        print(f"  总样本数: {total_samples:,}")
        print(f"  成功处理: {len(all_stats)}/{len(files)} 个文件")

        # 保存JSON
        stats_file = Path(output_dir) / 'processing_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)

        print(f"\n统计文件: {stats_file}")
    else:
        print("\n⚠ 警告：没有成功处理任何文件！")

    return all_stats


# === 主程序 ===
if __name__ == "__main__":
    stats = process_all_files_final(
        file_pattern='mhc_ligand_full_*.csv',
        output_dir='cleaned_data_mouse'
    )