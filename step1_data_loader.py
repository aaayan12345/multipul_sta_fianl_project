"""
Step 1: 数据加载与清洗
合并绩效1+绩效2的策略交易记录 & 账户A/B/C交易记录
统一列名: datetime, stock_code, stock_name, action, volume, price, amount
"""
import pandas as pd
import numpy as np
import re

# ============================================================
# 1. 加载策略交易记录
# ============================================================

def load_strategy_from_file1():
    """加载量化策略绩效-1.xlsx中的12个策略"""
    xls = pd.ExcelFile('量化策略绩效-1.xlsx')
    summary = pd.read_excel('量化策略绩效-1.xlsx', sheet_name='总表')

    # sheet名 → 策略名映射
    sheet_to_strategy = {
        '交易记录煤炭': '煤炭周期优选动态轮动策略',
        '交易记录300': '沪深300增强策略',
        '交易记录半导体': '半导体优选策略',
        '交易记录红利': '成长红利量化选股',
        '交易记录双创': '双创50增强',
        '交易记录计算机': '计算机ETF优选策略',
        '交易记录科创': '科创参数加强板',
        '交易记录机器人': '机器人ETF优选策略',
        '交易记录化工': '化工ETF优选策略',
        '交易记录中证2000': '中证2000增强',
        '交易记录形态': '形态识别',
        '交易记录800': '中证800增强',
    }

    all_records = []
    for sheet_name, strategy_name in sheet_to_strategy.items():
        df = pd.read_excel('量化策略绩效-1.xlsx', sheet_name=sheet_name)
        df['strategy_name'] = strategy_name
        df['source'] = '绩效1'
        all_records.append(df)

    return pd.concat(all_records, ignore_index=True)


def load_strategy_from_file2():
    """加载量化策略绩效-2.xlsx中有交易记录的22个策略"""
    xls = pd.ExcelFile('量化策略绩效-2.xlsx')

    # sheet名 → 策略名映射 (22个有交易记录的)
    sheet_to_strategy = {
        '交易记录军工': '军工etf增强',
        '交易记录1000': '中证1000增强',
        '交易记录旅游': '旅游etf增强',
        '交易记录国企': '国企etf增强',
        '交易记录游戏': '游戏etf增强',
        '交易记录酒': '酒etf增强',
        '交易记录动量趋势': '动量趋势策略',
        '交易记录锤子策略': '锤子策略',
        '交易记录综合全': '综合全',
        '医疗etf增强': '医疗etf增强',
        '交易记录通信': '通信etf增强',
        '交易记录房地产': '房地产etf增强',
        '交易记录食品': '食品etf增强',
        '交易记录创业板': '创业板增强',
        '交易记录etf动量改': 'etf动量改',
        '交易记录行业': '行业etf增强',
        '交易记录养殖': '养殖etf增强',
        '交易记录综合拆分1': '综合拆分1',
        '交易记录综合拆分2': '综合拆分2',
        '交易记录申万动量': '申万动量',
        '交易记录均衡持仓': '均衡持仓',
        '交易记录杠铃': '杠铃',
    }

    all_records = []
    for sheet_name, strategy_name in sheet_to_strategy.items():
        df = pd.read_excel('量化策略绩效-2.xlsx', sheet_name=sheet_name)
        df['strategy_name'] = strategy_name
        df['source'] = '绩效2'
        all_records.append(df)

    return pd.concat(all_records, ignore_index=True)


# ============================================================
# 1.3 加载新增CSV格式策略
# ============================================================

def load_strategy_from_csv(filepath, strategy_name, source_label):
    """加载CSV格式的策略交易记录（带ETF策略1/2、朝花夕拾等）
    列映射: product_id→strategy_name, symbol→stock_code, side→action,
            qty→volume, trade_date+trade_time→datetime
    """
    df = pd.read_csv(filepath, encoding='utf-8-sig')

    df['datetime'] = pd.to_datetime(df['trade_time'])  # trade_time已是完整datetime字符串
    df['stock_code'] = (df['symbol'].astype(str)
                        .str.replace('.XSHE', '', regex=False)
                        .str.replace('.XSHG', '', regex=False)
                        .str.replace('.SZSE', '', regex=False)
                        .str.replace('.SHSE', '', regex=False)
                        .str.strip())
    df['action'] = df['side'].str.upper().str.strip()
    df['volume'] = pd.to_numeric(df['qty'], errors='coerce').abs()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').abs()
    df['strategy_name'] = strategy_name
    df['source'] = source_label

    # 只保留实际买卖
    df = df[df['action'].isin(['BUY', 'SELL'])].copy()

    return df[['datetime', 'stock_code', 'action', 'volume', 'price', 'amount',
               'strategy_name', 'source']]


# ============================================================
# 2. 加载模拟账户
# ============================================================

def load_account(account_id):
    """加载模拟账户A/B/C"""
    file_map = {
        'A': '模拟账户A的记录.xlsx',
        'B': '模拟账户B的记录.xlsx',
        'C': '模拟账户C的记录.xlsx',
    }
    df = pd.read_excel(file_map[account_id])

    # 实际列名: 交收日期, 业务标示, 证券代码, 证券名称, 成交价格, 成交数量, 成交金额(A无), 初始金额
    col_mapping = {
        '交收日期': 'datetime_raw',
        '业务标示': 'btype',
        '证券代码': 'stock_code',
        '证券名称': 'stock_name',
        '成交价格': 'price',
        '成交数量': 'volume',
    }
    df = df.rename(columns=col_mapping)
    df['account_id'] = account_id
    df['source'] = f'模拟账户{account_id}'

    # Action: 证券买入→BUY, 证券卖出→SELL, 其他→OTHER
    df['action'] = df['btype'].apply(
        lambda x: 'BUY' if '买入' in str(x) else ('SELL' if '卖出' in str(x) else 'OTHER')
    )

    # amount: 账户A没有成交金额列, 用 price*volume 计算
    if '成交金额' in df.columns:
        df['amount'] = pd.to_numeric(df['成交金额'], errors='coerce')
    else:
        df['amount'] = pd.to_numeric(df['price'], errors='coerce') * pd.to_numeric(df['volume'], errors='coerce').abs()

    return df


# ============================================================
# 3. 统一格式化
# ============================================================

def standardize_strategy_df(df):
    """统一策略数据的列名和格式"""
    df = df.copy()

    # 去除初始资金记录: symbol为空或trade_time为NaT
    df = df[df['symbol'].notna()].copy()
    df = df[df['trade_time'].notna()].copy()

    # 统一列名
    df['datetime'] = pd.to_datetime(df['trade_time'])
    df['stock_code'] = df['symbol'].astype(str).str.replace('SHSE.', '').str.replace('SZSE.', '')
    df['price'] = pd.to_numeric(df['vwap'], errors='coerce')

    # amount转数值，失败置NaN
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').astype(float)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').astype(float)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype(float)
    # 有NaN的用 price*volume 填充
    mask = df['amount'].isna()
    df.loc[mask, 'amount'] = (df.loc[mask, 'price'] * df.loc[mask, 'volume'].abs()).astype(float)

    # 方向: 用volume符号判断, volume>0为买入, volume<0为卖出
    df['action'] = df['volume'].apply(lambda x: 'BUY' if x > 0 else ('SELL' if x < 0 else 'OTHER'))

    # volume和amount取绝对值
    df['volume'] = df['volume'].abs()
    df['amount'] = df['amount'].abs()

    return df[
        ['datetime', 'stock_code', 'action', 'volume', 'price', 'amount',
         'strategy_name', 'source']
    ]


def standardize_account_df(df):
    """统一账户数据的列名和格式"""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime_raw'], format='%Y%m%d')

    # 只保留实际买卖交易 (去除配售、中签、新股入账等)
    df = df[df['action'].isin(['BUY', 'SELL'])].copy()

    # stock_code 统一为6位字符串
    df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)

    # price和volume转数值
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    # NaN amount用 price*volume 补
    mask = df['amount'].isna()
    df.loc[mask, 'amount'] = (df.loc[mask, 'price'] * df.loc[mask, 'volume']).abs()

    return df[
        ['datetime', 'stock_code', 'stock_name', 'action', 'volume', 'price', 'amount',
         'account_id', 'source']
    ]


# ============================================================
# 4. 主流程
# ============================================================

def main():
    # 加载策略
    print("加载策略数据...")
    s1 = load_strategy_from_file1()
    s2 = load_strategy_from_file2()
    strategies_raw = pd.concat([s1, s2], ignore_index=True)
    strategies = standardize_strategy_df(strategies_raw)

    # 加载新增CSV格式策略 (3个)
    print("加载新增CSV策略...")
    s3 = load_strategy_from_csv('带ETF的策略1.csv', '国证2000ETF增强', '新增ETF')
    s4 = load_strategy_from_csv('带ETF策略2.csv', '创业板300ETF增强', '新增ETF')
    s5 = load_strategy_from_csv('朝花夕拾策略.csv', '朝花夕拾策略', '新增择时')
    strategies = pd.concat([strategies, s3, s4, s5], ignore_index=True)

    # 加载账户
    print("加载账户数据...")
    accounts = []
    for aid in ['A', 'B', 'C']:
        acc = load_account(aid)
        accounts.append(standardize_account_df(acc))
    accounts = pd.concat(accounts, ignore_index=True)

    # 统计
    print(f"\n===== 数据加载完成 =====")
    print(f"策略总数: {strategies['strategy_name'].nunique()}")
    print(f"策略交易记录: {len(strategies)} 条")
    print(f"账户总数: {accounts['account_id'].nunique()}")
    print(f"账户交易记录: {len(accounts)} 条")

    print(f"\n各策略记录数:")
    for name, cnt in strategies.groupby('strategy_name').size().sort_values(ascending=False).items():
        print(f"  {name}: {cnt} 条")

    print(f"\n各账户记录数:")
    for name, cnt in accounts.groupby('account_id').size().items():
        print(f"  账户{name}: {cnt} 条")

    # 时间范围
    print(f"\n策略时间范围: {strategies['datetime'].min()} ~ {strategies['datetime'].max()}")
    print(f"账户时间范围: {accounts['datetime'].min()} ~ {accounts['datetime'].max()}")

    # 股票覆盖
    strat_stocks = set(strategies['stock_code'].unique())
    acct_stocks = set(accounts['stock_code'].unique())
    print(f"\n策略涉及股票数: {len(strat_stocks)}")
    print(f"账户涉及股票数: {len(acct_stocks)}")
    print(f"重叠股票数: {len(strat_stocks & acct_stocks)}")

    # 保存
    strategies.to_csv('clean_strategies.csv', index=False)
    accounts.to_csv('clean_accounts.csv', index=False)
    print(f"\n已保存: clean_strategies.csv, clean_accounts.csv")

    return strategies, accounts


if __name__ == '__main__':
    strategies, accounts = main()
