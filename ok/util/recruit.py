"""全境征才(전지역모집)标签抽卡概率计算器。

复刻自 https://github.com/inittt/tenkaassist 的 recruit/recruit.js 计算逻辑。

翻译来源:标签与角色名的简体中文译名均提取自上游 `js/common.js` 的
`translate` 对象(`sc` 字段),即网站官方简体中文语言包,非手动翻译。
- 含「领袖(리더)」标签:枚举单标签/双标签组合,返回匹配 SSR 角色及均匀概率,
  同一角色被多个组合命中时取最大概率。
- 不含「领袖」标签:枚举单/双/三标签组合,返回每个组合的 SR 出现概率,
  池内按稀有度加权(SR:R:N = 1:10:30)。

角色数据 name_ko 为权威来源,name_zh 为上游官方简体译名(2 个上游未翻译的
角色 KS-Ⅷ/3호 保留字面 fallback),可通过 NAME_ZH_OVERRIDES 覆盖。
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from ok.util.logger import Logger

logger = Logger.get_logger("recruit")


# --------------------------------------------------------------------------- #
# 标签翻译表:简体中文(官方 sc) -> 韩文标签
# 译名取自上游 common.js 的 translate 对象 sc 字段
# --------------------------------------------------------------------------- #

# 官方简体中文标签 -> 韩文标签
TAG_ZH_TO_KO: dict[str, str] = {
    # 属性
    "火属性": "화속성",
    "水属性": "수속성",
    "风属性": "풍속성",
    "光属性": "광속성",
    "暗属性": "암속성",
    # 职业
    "攻击者": "딜러",
    "治疗者": "힐러",
    "守护者": "탱커",
    "辅助者": "서포터",
    "妨碍者": "디스럽터",
    # 种族
    "人类": "인간",
    "魔族": "마족",
    "亚人": "야인",
    # 体型
    "小体型": "작은체형",
    "中体型": "표준체형",
    # 胸型
    "贫乳": "빈유",
    "美乳": "미유",
    "巨乳": "거유",
    # 阶级
    "士兵": "병사",
    "菁英": "정예",
    "领袖": "리더",
    # 战斗标签
    "防御": "방어",
    "干扰": "방해",
    "输出": "데미지",
    "保护": "보호",
    "回复": "회복",
    "支援": "지원",
    "削弱": "쇠약",
    # 特殊
    "爆发力": "폭발력",
    "生存力": "생존력",
    "越战越强": "전투",
    "群体攻击": "범위공격",
    "回击": "반격",
}

# 简写/别名 -> 官方简体中文标签。
# 含常见中文习惯说法的兼容映射(均不与官方标签名冲突)。
# 注:中文「回」歧义(回复/回击),不在此表,需传「回复」或「回击」(或用「复」「击」)。
TAG_ZH_ALIAS: dict[str, str] = {
    # 属性首字
    "火": "火属性",
    "水": "水属性",
    "风": "风属性",
    "光": "光属性",
    "暗": "暗属性",
    # 职业首字
    "攻": "攻击者",
    "治": "治疗者",
    "守": "守护者",
    "辅": "辅助者",
    "妨": "妨碍者",
    # 种族首字
    "人": "人类",
    "魔": "魔族",
    "亚": "亚人",
    # 体型首字
    "小": "小体型",
    "中": "中体型",
    # 胸型首字
    "贫": "贫乳",
    "美": "美乳",
    "巨": "巨乳",
    # 阶级首字
    "士": "士兵",
    "菁": "菁英",
    "领": "领袖",
    # 战斗标签首字
    "防": "防御",
    "干": "干扰",
    "输": "输出",
    "保": "保护",
    "支": "支援",
    "削": "削弱",
    # 特殊首字
    "爆": "爆发力",
    "存": "生存力",
    "越": "越战越强",
    "群": "群体攻击",
    # 「回」歧义消歧快捷
    "复": "回复",
    "击": "回击",
    # 常见习惯说法兼容(不与官方名冲突)
    "群攻": "群体攻击",
    "队长": "领袖",
    "坦克": "守护者",
    "治疗": "治疗者",
    "奶": "治疗者",
    "辅助": "辅助者",
    "野人": "亚人",
    "精英": "菁英",
}

# 韩文标签全集,用于直接透传韩文输入
_TAG_KO_SET: set[str] = set(TAG_ZH_TO_KO.values())

# 首字「回」歧义集合,用于报错提示
_AMBIGUOUS_ZH: dict[str, list[str]] = {
    "回": ["回复", "回击"],
}

# 稀有度权重:模拟游戏内抽卡池相对出率(SR 最少,N 最多)
RARITY_WEIGHT: dict[str, int] = {"SR": 1, "R": 10, "N": 30}


# --------------------------------------------------------------------------- #
# 角色数据:复刻自 recruit.js 的 recruitJson.data
# 字段: (id, name_ko, name_zh, rarity, tags_ko_string)
# name_zh 为上游 common.js translate 对象的官方简体译名(sc 字段)。
# --------------------------------------------------------------------------- #

# 用户级角色中文名覆盖(优先级最高)
NAME_ZH_OVERRIDES: dict[str, str] = {}

_RAW_DATA: list[tuple[int, str, str, str, str]] = [
    # SSR
    (10001, "바알", "巴尔", "SSR", "화속성 딜러 마족 표준체형 리더 데미지"),
    (10002, "사탄", "撒旦", "SSR", "암속성 탱커 마족 표준체형 거유 리더 방어 생존력 반격"),
    (10003, "이블리스", "伊布力斯", "SSR", "광속성 딜러 마족 표준체형 리더 데미지 생존력 범위공격"),
    (10004, "살루시아", "赛露西亚", "SSR", "풍속성 서포터 야인 표준체형 거유 리더 지원 폭발력"),
    (10005, "란", "兰儿", "SSR", "수속성 딜러 야인 작은체형 빈유 리더 데미지 폭발력 전투"),
    (10006, "루루", "露露", "SSR", "풍속성 힐러 인간 리더 회복"),
    (10007, "밀레", "圣米勒", "SSR", "광속성 딜러 표준체형 리더 지원"),
    (10008, "KS-Ⅷ", "KS-Ⅷ", "SSR", "암속성 딜러 표준체형 리더 데미지 폭발력 전투"),
    (10018, "울타", "古勇", "SSR", "풍속성 탱커 인간 표준체형 리더 보호 방어 생존력"),
    (10019, "아야네", "现勇", "SSR", "광속성 딜러 인간 표준체형 리더 폭발력 데미지"),
    (10020, "무엘라", "未勇", "SSR", "풍속성 디스럽터 인간 표준체형 리더 지원 쇠약"),
    (10021, "하쿠", "贤者", "SSR", "풍속성 힐러 야인 리더 회복 지원"),
    (10028, "치즈루", "千鹤", "SSR", "풍속성 딜러 마족 표준체형 리더 데미지 폭발력"),
    (10033, "아르티아", "睡萝", "SSR", "암속성 디스럽터 야인 빈유 리더 쇠약"),
    (10037, "메스미나", "蛇后", "SSR", "화속성 디스럽터 마족 빈유 리더 쇠약"),
    (10039, "라티아", "血族", "SSR", "암속성 딜러 마족 표준체형 거유 리더 데미지 폭발력"),
    (10045, "슈텐", "伊吹", "SSR", "화속성 딜러 야인 표준체형 빈유 리더 데미지"),
    (10047, "테키", "狄", "SSR", "풍속성 딜러 인간 표준체형 리더 데미지 쇠약"),
    (10048, "모모", "莫默", "SSR", "수속성 딜러 마족 빈유 리더 데미지 폭발력"),
    (10049, "파야", "法雅", "SSR", "화속성 힐러 마족 리더 회복 지원"),
    (10056, "카시피나", "堕龙", "SSR", "수속성 탱커 야인 표준체형 거유 리더 보호 방어 반격"),
    (10057, "에피나", "煌星", "SSR", "암속성 서포터 인간 표준체형 빈유 리더 지원"),
    (10059, "이노리", "马娘", "SSR", "풍속성 딜러 야인 표준체형 리더 데미지"),
    (10062, "세라프", "商狐", "SSR", "수속성 힐러 야인 표준체형 리더 보호 회복 지원"),
    (10063, "에밀리", "大女仆", "SSR", "광속성 서포터 인간 표준체형 거유 리더 회복 지원"),
    (10066, "안젤리카", "千咒", "SSR", "암속성 딜러 리더 데미지 폭발력 전투"),
    (10068, "렌", "莲", "SSR", "화속성 힐러 인간 표준체형 리더 회복 보호 지원"),
    (10084, "미루", "咪噜", "SSR", "화속성 딜러 리더 폭발력 생존력"),
    # SR
    (10009, "아이카", "艾可", "SR", "암속성 서포터 마족 표준체형 미유 정예 지원"),
    (10010, "레오나", "雷欧娜", "SR", "수속성 탱커 인간 표준체형 미유 정예 보호 방어 생존력"),
    (10011, "피오라", "菲欧菈", "SR", "광속성 힐러 인간 표준체형 미유 정예 회복"),
    (10012, "리츠키", "凛月", "SR", "풍속성 딜러 인간 표준체형 미유 정예 데미지 폭발력 범위공격"),
    (10013, "미나요미", "神无雪", "SR", "화속성 딜러 야인 표준체형 미유 정예 쇠약 전투"),
    (10014, "시즈카", "静", "SR", "수속성 디스럽터 야인 작은체형 미유 정예 방해 쇠약"),
    (10015, "쥬노안", "朱诺安", "SR", "암속성 딜러 인간 표준체형 거유 정예 데미지 지원"),
    (10016, "브리트니", "布兰妮", "SR", "광속성 디스럽터 인간 미유 정예 지원 쇠약 폭발력 범위공격"),
    (10036, "나프라라", "娜芙菈菈", "SR", "풍속성 탱커 마족 표준체형 거유 정예 보호 방어 회복 생존력"),
    (10038, "토타라", "托特拉", "SR", "광속성 딜러 인간 표준체형 미유 정예 데미지 쇠약 폭발력"),
    (10041, "호타루", "小萤", "SR", "수속성 힐러 인간 표준체형 빈유 정예 회복 지원"),
    (10046, "가벨", "刺针", "SR", "풍속성 딜러 인간 표준체형 미유 정예 데미지"),
    (10051, "프리실라", "银龙", "SR", "암속성 디스럽터 야인 미유 정예 쇠약"),
    (10055, "타노시아", "塔诺西雅", "SR", "광속성 서포터 야인 미유 정예 회복"),
    # R
    (10801, "아이린", "艾琳", "R", "광속성 힐러 인간 표준체형 거유 회복"),
    (10802, "나나", "娜娜", "R", "풍속성 딜러 마족 작은체형 빈유 데미지"),
    (10803, "아이리스", "伊维丝", "R", "화속성 딜러 야인 작은체형 빈유 데미지 전투 범위공격"),
    (10804, "도라", "朵拉", "R", "풍속성 탱커 야인 표준체형 미유 보호 방어 생존력"),
    (10805, "세바스", "撒芭丝", "R", "암속성 디스럽터 마족 표준체형 미유 방해"),
    (10806, "마를렌", "玛莲", "R", "수속성 힐러 야인 표준체형 미유 회복"),
    (10807, "유이", "尤依", "R", "화속성 딜러 인간 작은체형 거유 데미지 전투"),
    (10808, "소라카", "索拉卡", "R", "암속성 디스럽터 야인 표준체형 미유 쇠약"),
    (10813, "이아", "伊艾", "R", "광속성 힐러 인간 작은체형 빈유 회복"),
    # N
    (10901, "사이렌", "赛莲", "N", "암속성 탱커 인간 표준체형 미유 병사 보호 방어"),
    (10902, "페트라", "佩托拉", "N", "광속성 딜러 인간 표준체형 빈유 병사 데미지 범위공격"),
    (10903, "프레이", "芙蕾", "N", "광속성 탱커 마족 표준체형 미유 병사 보호 방어"),
    (10904, "마누엘라", "玛努艾拉", "N", "암속성 딜러 마족 표준체형 미유 병사 데미지"),
    (10905, "키쿄", "桔梗", "N", "화속성 디스럽터 인간 표준체형 미유 병사 쇠약"),
    (10906, "카에데", "枫", "N", "풍속성 힐러 인간 표준체형 미유 병사 회복"),
    (10907, "올라", "奧菈", "N", "풍속성 딜러 야인 표준체형 미유 병사 데미지"),
    (10908, "콜레트", "可儿", "N", "수속성 딜러 야인 작은체형 빈유 병사 데미지 폭발력"),
    (10909, "샤린", "夏琳", "N", "화속성 탱커 인간 표준체형 미유 병사 보호 방어 범위공격"),
    (10910, "마티나", "玛蒂娜", "N", "광속성 탱커 인간 표준체형 미유 병사 보호 방어 생존력"),
    (10911, "클레어", "克蕾雅", "N", "광속성 힐러 인간 표준체형 미유 병사 회복"),
    (10912, "로라", "萝尔", "N", "수속성 디스럽터 마족 작은체형 미유 병사 회복 쇠약 생존력"),
    (10913, "미르노", "米诺", "N", "풍속성 탱커 야인 표준체형 거유 병사 보호 방어 방해"),
    (10914, "라미아", "拉米亚", "N", "화속성 디스럽터 마족 표준체형 미유 병사 방해 쇠약"),
    (10915, "하피", "哈比", "N", "풍속성 디스럽터 마족 표준체형 미유 병사 방해 쇠약"),
    (10916, "안나", "安娜", "N", "화속성 탱커 인간 표준체형 미유 병사 보호 방어"),
    (10917, "브란", "布兰", "N", "풍속성 딜러 인간 표준체형 미유 병사 데미지 방어"),
    (10918, "노노카", "诺诺可", "N", "수속성 딜러 인간 표준체형 미유 병사 데미지 폭발력"),
    (10919, "징벌천사", "惩戒天使", "N", "수속성 탱커 병사 생존력"),
    (10920, "복음천사", "福音天使", "N", "수속성 힐러 병사"),
    (10921, "몰리", "茉莉", "N", "인간 수속성 빈유 딜러 작은체형 병사 데미지"),
    (10922, "3호", "3号", "N", "광속성 딜러 작은체형 미유 병사 데미지 생존력"),
    (10923, "세실", "赛希", "N", "풍속성 딜러 야인 표준체형 거유 병사 데미지 폭발력"),
    (10924, "무무", "穆穆", "N", "암속성 디스럽터 표준체형 미유 병사 보호 방해 생존력"),
    (10933, "안야", "安雅", "N", "인간 풍속성 디스럽터 병사"),
]


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #

@dataclass
class Character:
    id: int
    name_ko: str  # 韩文原名(权威)
    name_zh: str  # 上游官方简体译名
    rarity: str  # "SSR" | "SR" | "R" | "N"
    tags_ko: list[str]  # 韩文标签列表


@dataclass
class SSRResult:
    """Leader 模式下的单个 SSR 角色结果。"""

    character: Character
    percent: float  # 0-1
    matched_tags_zh: list[str]  # 命中的中文标签(产生该概率的组合)


@dataclass
class SRChance:
    """非 Leader 模式下,某个标签组合的 SR 出现概率。"""

    percent: float  # 0-1
    matched_tags_zh: list[str]  # 该组合对应的中文标签


@dataclass
class RecruitResult:
    leader_mode: bool
    ssr_results: list[SSRResult] = field(default_factory=list)  # leader_mode=True 时填充
    sr_chances: list[SRChance] = field(default_factory=list)  # leader_mode=False 时填充


# --------------------------------------------------------------------------- #
# 初始化角色表
# --------------------------------------------------------------------------- #

def _build_characters() -> list[Character]:
    out: list[Character] = []
    for cid, name_ko, name_zh, rarity, tags_str in _RAW_DATA:
        zh = NAME_ZH_OVERRIDES.get(name_ko, name_zh)
        out.append(Character(
            id=cid,
            name_ko=name_ko,
            name_zh=zh,
            rarity=rarity,
            tags_ko=tags_str.split(),
        ))
    return out


CHARACTERS: list[Character] = _build_characters()


# --------------------------------------------------------------------------- #
# 计算器
# --------------------------------------------------------------------------- #

class RecruitCalculator:
    """全境征才概率计算器。

    传入简体中文标签(官方译名、习惯别名或首字简写,如「火属性」「火」「攻击者」
    「攻」「领袖」「队长」),内部翻译为韩文标签后复刻上游 JS 的计算逻辑。
    """

    def __init__(self, characters: Optional[list[Character]] = None):
        self._characters = characters if characters is not None else CHARACTERS

    # ---- 标签解析 -------------------------------------------------------- #

    def resolve_tag(self, tag: str) -> Optional[str]:
        """把中文标签/别名/韩文标签解析为韩文标签。

        无法解析返回 None。「回」因歧义(回复/回击)也返回 None 并告警。
        """
        tag = tag.strip()
        if not tag:
            return None
        # 直接是韩文标签 -> 透传
        if tag in _TAG_KO_SET:
            return tag
        # 官方简体中文标签
        if tag in TAG_ZH_TO_KO:
            return TAG_ZH_TO_KO[tag]
        # 简写/别名
        if tag in TAG_ZH_ALIAS:
            return TAG_ZH_TO_KO[TAG_ZH_ALIAS[tag]]
        # 歧义首字
        if tag in _AMBIGUOUS_ZH:
            logger.warning(
                f"标签 '{tag}' 存在歧义,请使用完整名: {_AMBIGUOUS_ZH[tag]}"
            )
            return None
        logger.warning(f"未知标签: '{tag}',可用标签见 list_tags()")
        return None

    def resolve_tags(self, tags: list[str]) -> list[str]:
        """批量解析并去重(保持首次出现顺序)。"""
        seen: set[str] = set()
        out: list[str] = []
        for t in tags:
            ko = self.resolve_tag(t)
            if ko and ko not in seen:
                seen.add(ko)
                out.append(ko)
        return out

    @staticmethod
    def list_tags() -> list[str]:
        """返回所有可用的官方简体中文标签。"""
        return list(TAG_ZH_TO_KO.keys())

    # ---- 主入口 ---------------------------------------------------------- #

    def calculate(self, tags_zh: list[str]) -> RecruitResult:
        """根据中文标签列表计算结果。

        含「领袖」进入 Leader 模式(返回 SSR 列表),否则进入 SR 模式
        (返回每个标签组合的 SR 出现概率)。最多取前 5 个有效标签(对齐上游上限)。
        """
        resolved = self.resolve_tags(tags_zh)
        if len(resolved) > 5:
            logger.info(f"标签数 {len(resolved)} 超过 5,仅取前 5 个")
            resolved = resolved[:5]

        if "리더" in resolved:
            cur = [t for t in resolved if t != "리더"]
            ssr = self._calc_leader(cur)
            return RecruitResult(leader_mode=True, ssr_results=ssr, sr_chances=[])
        else:
            chances = self._calc_sr(resolved)
            return RecruitResult(leader_mode=False, ssr_results=[], sr_chances=chances)

    # ---- Leader 模式:SSR 均匀分布,取最大概率组合 ------------------------ #

    def _calc_leader(self, cur_tags_ko: list[str]) -> list[SSRResult]:
        # 中文回译表:ko -> zh
        ko_to_zh = {v: k for k, v in TAG_ZH_TO_KO.items()}
        best: dict[int, SSRResult] = {}
        # 仅使用单标签 + 双标签组合(对齐上游 tag1/tag2)
        combos: list[list[str]] = []
        combos += [list(c) for c in combinations(cur_tags_ko, 1)]
        combos += [list(c) for c in combinations(cur_tags_ko, 2)]

        for combo in combos:
            matched = [
                ch for ch in self._characters
                if ch.rarity == "SSR" and all(t in ch.tags_ko for t in combo)
            ]
            if not matched:
                continue
            per = 1.0 / len(matched)
            combo_zh = [ko_to_zh.get(t, t) for t in combo]
            for ch in matched:
                prev = best.get(ch.id)
                if prev is None or per > prev.percent:
                    best[ch.id] = SSRResult(
                        character=ch,
                        percent=per,
                        matched_tags_zh=combo_zh,
                    )
        return sorted(best.values(), key=lambda r: r.percent, reverse=True)

    # ---- 非 Leader 模式:SR 加权占比 ------------------------------------- #

    def _calc_sr(self, cur_tags_ko: list[str]) -> list[SRChance]:
        ko_to_zh = {v: k for k, v in TAG_ZH_TO_KO.items()}
        combos: list[list[str]] = []
        combos += [list(c) for c in combinations(cur_tags_ko, 1)]
        combos += [list(c) for c in combinations(cur_tags_ko, 2)]
        combos += [list(c) for c in combinations(cur_tags_ko, 3)]

        chances: list[SRChance] = []
        for combo in combos:
            per = self._find_sr_percent(combo)
            if per <= 0:
                continue
            combo_zh = [ko_to_zh.get(t, t) for t in combo]
            chances.append(SRChance(percent=per, matched_tags_zh=combo_zh))
        chances.sort(key=lambda c: c.percent, reverse=True)
        return chances

    def _find_sr_percent(self, combo: list[str]) -> float:
        """SR / (SR*1 + R*10 + N*30)。"""
        matched = [
            ch for ch in self._characters
            if ch.rarity != "SSR" and all(t in ch.tags_ko for t in combo)
        ]
        if not matched:
            return 0.0
        sr_cnt = 0
        total = 0
        for ch in matched:
            w = RARITY_WEIGHT.get(ch.rarity, 0)
            if ch.rarity == "SR":
                sr_cnt += 1
            total += w
        if total == 0:
            return 0.0
        return sr_cnt / total


# --------------------------------------------------------------------------- #
# 模块级便捷函数
# --------------------------------------------------------------------------- #

_default_calc = RecruitCalculator()


def resolve_tag(tag: str) -> Optional[str]:
    return _default_calc.resolve_tag(tag)


def calculate(tags_zh: list[str]) -> RecruitResult:
    """便捷入口:用默认角色表计算。"""
    return _default_calc.calculate(tags_zh)


def format_result(result: RecruitResult, top: int = 20) -> str:
    """把结果格式化成易读文本(用于 CLI / 日志)。"""
    lines: list[str] = []
    if result.leader_mode:
        lines.append(f"[Leader 模式] 命中 SSR {len(result.ssr_results)} 个:")
        for r in result.ssr_results[:top]:
            ch = r.character
            lines.append(
                f"  {ch.name_zh}({ch.name_ko}) [{ch.rarity}] "
                f"{r.percent * 100:.2f}%  <- {' '.join(r.matched_tags_zh)}"
            )
    else:
        lines.append(f"[SR 模式] 命中组合 {len(result.sr_chances)} 个:")
        for c in result.sr_chances[:top]:
            lines.append(
                f"  {c.percent * 100:.2f}%  <- {' '.join(c.matched_tags_zh)}"
            )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="全境征才标签概率计算")
    parser.add_argument("tags", nargs="+", help="中文标签(官方译名/别名/首字),最多 5 个")
    args = parser.parse_args()

    res = calculate(args.tags)
    print(format_result(res))
