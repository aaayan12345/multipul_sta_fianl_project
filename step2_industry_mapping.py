"""
Step 2: 股票代码 → 申万一级行业映射
策略: 规则快速匹配 + DeepSeek API 补充未匹配
"""
import pandas as pd
import re
import json
import time
from collections import defaultdict
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
import os
from dotenv import load_dotenv
load_dotenv()
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 申万一级行业 31 类
SW_LIST = [
    "煤炭", "电力设备", "电子", "房地产", "纺织服饰", "非银金融",
    "钢铁", "公用事业", "国防军工", "环保", "机械设备", "基础化工",
    "计算机", "家用电器", "建筑材料", "建筑装饰", "交通运输",
    "美容护理", "农林牧渔", "汽车", "轻工制造", "商贸零售",
    "社会服务", "石油石化", "食品饮料", "通信", "传媒",
    "医药生物", "银行", "有色金属", "综合"
]

# ============================================================
# 1. 数据准备
# ============================================================

def normalize_code(code):
    code = str(code).strip()
    code = re.sub(r'^(SHSE|SZSE|SH|SZ)\.?', '', code)
    if code.isdigit():
        code = code.zfill(6)
    return code

def load_stock_list():
    """加载所有去重股票, 尽可能获取名称"""
    # 从akshare获取A股列表
    stock_info = {}
    try:
        import akshare as ak
        print("从akshare获取A股基本信息...")
        df = ak.stock_info_a_code_name()
        for _, row in df.iterrows():
            code = normalize_code(str(row['code']))
            stock_info[code] = str(row['name'])
        print(f"  获取到 {len(stock_info)} 只股票名称")
    except Exception as e:
        print(f"  akshare 获取失败: {e}")

    # 补充账户数据中的股票名
    acct = pd.read_csv('clean_accounts.csv')
    for _, row in acct.iterrows():
        code = normalize_code(row['stock_code'])
        name = str(row.get('stock_name', ''))
        if name and name != 'nan' and code not in stock_info:
            stock_info[code] = name

    # 加载我们需要的所有股票
    strat = pd.read_csv('clean_strategies.csv')
    our_codes = set()
    for c in pd.concat([strat['stock_code'], acct['stock_code']]):
        our_codes.add(normalize_code(c))

    # 过滤掉可转债、债券等非股票
    filtered = set()
    for c in our_codes:
        # 过滤可转债 (12xxxx), 债券 (1xxxxx but not stock), 短代码
        if len(c) < 5:
            continue
        if c.startswith(('12', '11', '10', '13', '14')):  # 可转债/债券
            continue
        if c.startswith('9'):  # B股等
            continue
        filtered.add(c)

    print(f"过滤后股票: {len(filtered)} 只 (排除 {len(our_codes) - len(filtered)} 只非股票)")

    return filtered, stock_info

# ============================================================
# 2. 关键词规则快速匹配
# ============================================================

def classify_by_rules(code, name=''):
    """返回 (industry, confidence)"""
    name = str(name)

    # 精确关键词
    rules = [
        (r'煤$|煤矿|煤业|焦煤|焦炭', '煤炭'),
        (r'电力$|发电$|水电|火电|核电|风电|热电|绿电|新能源$', '公用事业'),
        (r'钢铁$|特钢|钢管', '钢铁'),
        (r'矿业$|黄金|白银|铜业|铝业|铅锌|镍业|钴业|锂业|稀土|钨业|钼业|钛业|锡业|有色', '有色金属'),
        (r'石油|石化|油气|炼化', '石油石化'),
        (r'化工$|化学|化肥|农药|塑料|橡胶|树脂|涂料|染料|助剂|聚氨酯|碳纤维|氟化工|硅化工|氯碱|纯碱|烧碱|甲醇|钛白粉|炭黑', '基础化工'),
        (r'建材|水泥|玻璃|陶瓷|石膏|防水|管材', '建筑材料'),
        (r'建筑|装饰|装修|幕墙|园林|钢结构|基建|工程|施工', '建筑装饰'),
        (r'汽车|客车|轿车|汽配|轮胎|车桥|车轴', '汽车'),
        (r'家电|空调|冰箱|洗衣机|彩电|厨电|小家电|照明', '家用电器'),
        (r'纺织|服装|服饰|鞋业|家纺|毛纺|棉纺|化纤|染整|丝绸|皮革|箱包|内衣|袜子', '纺织服饰'),
        (r'轻工|造纸|包装|家具|家居|卫浴|玩具|文具|五金|锁具', '轻工制造'),
        (r'制药|药业|医药|生物$|基因|细胞|免疫|疫苗|诊断|检测试剂|医疗器械|医疗设备|中药|血液制品|胰岛素', '医药生物'),
        (r'食品|饮料|白酒|啤酒|红酒|乳业|调味品|酱油|醋|食用油|休闲食品|烘焙|糖果|预制菜|肉制品', '食品饮料'),
        (r'农业$|牧业|渔业|养殖|饲料|种子|种业|粮食|农[场垦林田]', '农林牧渔'),
        (r'房地产|地产|房产|置业|物业|园区', '房地产'),
        (r'银行$', '银行'),
        (r'证券|券商|保险|信托|期货', '非银金融'),
        (r'交通|运输|物流|快递|航运|港口|高速[公路]|铁路|机场|航空|海运|远洋|仓储|配送|供应链', '交通运输'),
        (r'电子$|半导体|芯片|集成电路|晶圆|封装|PCB|LED|OLED|光学|光电子|激光|传感器|电容|电阻|电感|连接器', '电子'),
        (r'通信|电信|5G|6G|基站|天线|射频|滤波器|光通信|光纤|光缆|光模块|交换机|路由器', '通信'),
        (r'计算机$|软件|云计算|大数据|人工智能|AI|区块链|信创|信息安全|网络安全|数字货币|智慧|智能', '计算机'),
        (r'传媒|游戏|影视|动漫|出版|广电|数字媒体|元宇宙|广告|营销|直播|短视频', '传媒'),
        (r'军工|国防|航天|航空|舰船|兵器|导弹|卫星|火箭|雷达|战斗机|军品', '国防军工'),
        (r'机械|设备|机床|泵|阀|轴承|齿轮|减速器|电机|风机|压缩机|真空|液压|气动|密封|弹簧|紧固件|链条|模具|刀具', '机械设备'),
        (r'电力设备|电气|变压器|开关|断路器|继电器|配电|输电|变电|电缆|电线|绝缘|光伏|储能|电池|锂电池|钠电池|燃料电池|氢能|逆变器|太阳能', '电力设备'),
        (r'商贸|零售|百货|超市|便利店|购物|免税|电商|贸易|外贸|进出口|批发|供销', '商贸零售'),
        (r'旅游|酒店|餐饮|景区|旅行社|教育|培训|体育|检测|认证|人力资源|会展|环保|环卫|污水|垃圾|环境|生态', '社会服务'),
        (r'美容|护理|化妆|护肤|医美|整形|口腔|牙科|眼科|植发', '美容护理'),
    ]

    for pattern, industry in rules:
        if re.search(pattern, name):
            return industry

    return None

# ============================================================
# 3. DeepSeek API 批量分类
# ============================================================

def classify_with_deepseek(batch, stock_info, client):
    """用DeepSeek批量分类股票行业"""
    # 构建提示
    items = []
    for code in batch:
        name = stock_info.get(code, '未知')
        items.append(f"{code} {name}")

    items_text = "\n".join(items)

    prompt = f"""你是一个A股行业分类专家。请对以下股票逐一判断其申万一级行业。

申万一级行业共31类:
{', '.join(SW_LIST)}

请严格按以下JSON格式输出,不要输出任何其他内容:
```json
[{{"code": "股票代码", "industry": "行业名"}}, ...]
```

股票列表:
{items_text}

注意:
1. 如果股票名称为"未知",根据代码段推断(如688xxx多为电子/医药/机械设备)
2. 行业必须是上述31类之一
3. ETF(代码15xxxx/16xxxx/51xxxx)根据名称中的行业关键词判断,如"军工ETF"→国防军工"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        result_text = response.choices[0].message.content.strip()

        # 提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
        if json_match:
            result_text = json_match.group(1)
        else:
            # 尝试直接找 []
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)

        results = json.loads(result_text)
        return {item['code']: item['industry'] for item in results}

    except Exception as e:
        print(f"  DeepSeek API 错误: {e}")
        return {}

# ============================================================
# 4. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("Step 2: 股票代码 → 申万一级行业映射")
    print("=" * 60)

    # 加载股票
    all_codes, stock_info = load_stock_list()

    # 第一轮: 规则匹配
    print("\n第一轮: 关键词规则匹配...")
    mapping = {}
    for code in all_codes:
        name = stock_info.get(code, '')
        ind = classify_by_rules(code, name)
        if ind:
            mapping[code] = {'industry': ind, 'source': 'rule'}

    print(f"  规则匹配: {len(mapping)} 只")

    # 补充: 从策略名推断
    print("\n第二轮: 策略名推断...")
    strat = pd.read_csv('clean_strategies.csv')
    strategy_stocks = strat[['stock_code', 'strategy_name']].drop_duplicates()

    strategy_ind_map = {
        '煤炭': '煤炭', '半导体': '电子', '医药': '医药生物', '医疗': '医药生物',
        '军工': '国防军工', '计算机': '计算机', '化工': '基础化工',
        '食品': '食品饮料', '酒': '食品饮料', '游戏': '传媒', '通信': '通信',
        '旅游': '社会服务', '房地产': '房地产', '养殖': '农林牧渔',
        '机器人': '机械设备', '创业板': '综合', '科创': '电子',
        '双创': '综合', '红利': '综合', '成长': '综合', '形态': '综合',
        '动量': '综合', '综合': '综合', '均衡': '综合', '杠铃': '综合',
        '300增强': '综合', '500增强': '综合', '800增强': '综合',
        '1000增强': '综合', '2000增强': '综合',
    }

    for _, row in strategy_stocks.iterrows():
        code = normalize_code(row['stock_code'])
        if code in mapping:
            continue
        sname = str(row['strategy_name'])
        for key, ind in strategy_ind_map.items():
            if key in sname:
                mapping[code] = {'industry': ind, 'source': 'strategy'}
                break

    print(f"  策略名补充: {len(mapping)} 只")

    # 第三轮: 代码段推断 (ETF)
    print("\n第三轮: ETF代码段推断...")
    etf_industry_map = {
        '159': '综合', '510': '综合', '511': '综合', '512': '综合',
        '513': '综合', '515': '综合', '516': '综合', '517': '综合',
        '518': '综合', '588': '综合', '560': '综合', '561': '综合',
        '562': '综合', '563': '综合',
    }
    for code in all_codes:
        if code in mapping:
            continue
        for prefix, ind in etf_industry_map.items():
            if code.startswith(prefix):
                mapping[code] = {'industry': ind, 'source': 'code_segment'}
                break

    print(f"  代码段补充: {len(mapping)} 只")

    # 统计
    unmapped = [c for c in all_codes if c not in mapping]
    zonghe = [c for c in mapping if mapping[c]['industry'] == '综合']
    print(f"\n当前: 匹配 {len(mapping)}, 其中综合={len(zonghe)}, 未匹配={len(unmapped)}")

    # 第四轮: DeepSeek API — 先处理未匹配
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    BATCH_SIZE = 30

    if unmapped:
        print(f"\n第四轮: DeepSeek API 处理未匹配 ({len(unmapped)} 只)...")
        for i in range(0, len(unmapped), BATCH_SIZE):
            batch = unmapped[i:i + BATCH_SIZE]
            print(f"  批次 {i//BATCH_SIZE + 1}/{(len(unmapped)-1)//BATCH_SIZE + 1}: {len(batch)} 只...")
            result = classify_with_deepseek(batch, stock_info, client)
            for code, ind in result.items():
                if ind in SW_LIST:
                    mapping[code] = {'industry': ind, 'source': 'deepseek'}
            time.sleep(1)

    # 第五轮: DeepSeek API — 处理"综合"类(宽基策略的个股)
    zonghe_left = [c for c in mapping if mapping[c]['industry'] == '综合']
    if zonghe_left:
        print(f"\n第五轮: DeepSeek API 处理综合类 ({len(zonghe_left)} 只)...")
        for i in range(0, len(zonghe_left), BATCH_SIZE):
            batch = zonghe_left[i:i + BATCH_SIZE]
            print(f"  {i//BATCH_SIZE+1}/{len(zonghe_left)//BATCH_SIZE+1}: {len(batch)}只 ...", end=' ')
            result = classify_with_deepseek(batch, stock_info, client)
            for code, ind in result.items():
                if ind in SW_LIST:
                    mapping[code] = {'industry': ind, 'source': 'deepseek'}
            print(f"本批匹配: {len(result)}")
            time.sleep(1)

    # 最终统计
    still_unmapped = [c for c in all_codes if c not in mapping]
    print(f"\n最终: 匹配 {len(mapping)}, 未匹配 {len(still_unmapped)}")

    # 行业分布
    ind_count = defaultdict(int)
    for code, info in mapping.items():
        ind_count[info['industry']] += 1

    print(f"\n行业分布:")
    for ind in sorted(ind_count.keys(), key=lambda x: -ind_count[x]):
        print(f"  {ind}: {ind_count[ind]} 只")

    if still_unmapped:
        print(f"\n仍无法匹配: {len(still_unmapped)} 只 (保存到 unmapped_stocks.txt)")
        with open('unmapped_stocks.txt', 'w', encoding='utf-8') as f:
            for c in still_unmapped:
                f.write(f"{c}\t{stock_info.get(c, '')}\n")

    # 保存映射表
    rows = []
    for code in all_codes:
        info = mapping.get(code, {})
        rows.append({
            'stock_code': code,
            'stock_name': stock_info.get(code, ''),
            'industry': info.get('industry', '综合'),
            'source': info.get('source', 'unmapped'),
        })
    df = pd.DataFrame(rows)
    df.to_csv('stock_industry_mapping.csv', index=False, encoding='utf-8-sig')
    print(f"\n已保存: stock_industry_mapping.csv ({len(df)} 条)")

    return mapping

if __name__ == '__main__':
    mapping = main()
