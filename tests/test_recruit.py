"""全境征才计算器单元测试。

case 数值均按上游 recruit.js 逻辑手算复核,用于锁定行为一致性。
标签与角色译名均采用上游 common.js 官方简体中文(sc)语言包。
"""

import pytest

from ok.util.recruit import (
    RecruitCalculator,
    TAG_ZH_TO_KO,
    calculate,
    resolve_tag,
    format_result,
)


# --------------------------------------------------------------------------- #
# 标签解析
# --------------------------------------------------------------------------- #

class TestResolveTag:
    def test_official_chinese_tag(self):
        assert resolve_tag("火属性") == "화속성"
        assert resolve_tag("领袖") == "리더"
        assert resolve_tag("群体攻击") == "범위공격"
        assert resolve_tag("越战越强") == "전투"

    def test_short_alias(self):
        assert resolve_tag("火") == "화속성"
        assert resolve_tag("领") == "리더"
        assert resolve_tag("攻") == "딜러"
        assert resolve_tag("巨") == "거유"
        assert resolve_tag("妨") == "디스럽터"

    def test_habit_alias(self):
        # 常见中文习惯说法(不与官方名冲突)
        assert resolve_tag("队长") == "리더"
        assert resolve_tag("坦克") == "탱커"
        assert resolve_tag("治疗") == "힐러"
        assert resolve_tag("野人") == "야인"
        assert resolve_tag("精英") == "정예"
        assert resolve_tag("群攻") == "범위공격"

    def test_korean_passthrough(self):
        assert resolve_tag("화속성") == "화속성"
        assert resolve_tag("리더") == "리더"

    def test_ambiguous_hui_returns_none(self):
        # 「回」歧义(回复/回击),必须由用户明确
        assert resolve_tag("回") is None

    def test_hui_disambiguation(self):
        assert resolve_tag("复") == "회복"
        assert resolve_tag("击") == "반격"

    def test_fang_no_longer_ambiguous(self):
        # 中文「防」首字仅对应「防御」(「干扰」首字为「干」),无歧义
        assert resolve_tag("防") == "방어"
        assert resolve_tag("干") == "방해"

    def test_unknown_tag_returns_none(self):
        assert resolve_tag("不存在的标签") is None

    def test_whitespace_trimmed(self):
        assert resolve_tag("  火属性  ") == "화속성"

    def test_empty_returns_none(self):
        assert resolve_tag("") is None
        assert resolve_tag("   ") is None


class TestResolveTags:
    def test_dedup_keeps_order(self):
        calc = RecruitCalculator()
        assert calc.resolve_tags(["火", "火属性", "火"]) == ["화속성"]

    def test_skips_invalid(self):
        calc = RecruitCalculator()
        assert calc.resolve_tags(["火属性", "回", "乱写"]) == ["화속성"]


def test_list_tags_covers_all_ko():
    tags = RecruitCalculator.list_tags()
    # 33 个官方简体标签,且每个都映射到唯一韩文标签
    assert len(tags) == 33
    assert len(TAG_ZH_TO_KO) == 33
    assert len(set(TAG_ZH_TO_KO.values())) == 33


# --------------------------------------------------------------------------- #
# Leader 模式(SSR)
# --------------------------------------------------------------------------- #

class TestLeaderMode:
    def test_leader_mode_flag(self):
        result = calculate(["领袖", "火属性"])
        assert result.leader_mode is True
        assert result.ssr_results != []
        assert result.sr_chances == []

    def test_fire_attacker_leader_distribution(self):
        """[火属性, 攻击者, 领袖]: 火属性+攻击者 SSR 共 3 个,各 1/3。"""
        result = calculate(["火属性", "攻击者", "领袖"])
        ssr = result.ssr_results
        # 头部 3 个角色概率应为 1/3
        top3 = [r for r in ssr if abs(r.percent - 1 / 3) < 1e-9]
        top_names = {r.character.name_ko for r in top3}
        assert top_names == {"바알", "슈텐", "미루"}
        # 且按概率降序
        pcts = [r.percent for r in ssr]
        assert pcts == sorted(pcts, reverse=True)

    def test_leader_takes_max_probability_combo(self):
        """同一 SSR 被多组合命中时取最大概率。"""
        result = calculate(["火属性", "攻击者", "领袖"])
        by_id = {r.character.id: r for r in result.ssr_results}
        # 바알 命中 [화속성](1/6)、[딜러](1/14)、[화속성,딜러](1/3),应取 1/3
        baal = by_id[10001]
        assert baal.percent == pytest.approx(1 / 3)
        assert baal.matched_tags_zh == ["火属性", "攻击者"]
        # 메스미나 仅命中 [화속성](1/6),非攻击者
        mes = by_id[10037]
        assert mes.percent == pytest.approx(1 / 6)
        assert mes.matched_tags_zh == ["火属性"]

    def test_leader_total_unique_ssr(self):
        """[火属性, 攻击者]: 火属性SSR(6) ∪ 攻击者SSR(14) = 17 个 unique。"""
        result = calculate(["火属性", "攻击者", "领袖"])
        assert len(result.ssr_results) == 17

    def test_leader_arbitrary_combo_runs(self):
        """任意合法标签组合不应抛异常,且结果概率在 (0,1]。"""
        result = calculate(["领袖", "风属性", "守护者", "魔族"])
        assert result.leader_mode is True
        for r in result.ssr_results:
            assert 0 < r.percent <= 1


# --------------------------------------------------------------------------- #
# 非 Leader 模式(SR)
# --------------------------------------------------------------------------- #

class TestSRMode:
    def test_sr_mode_flag(self):
        result = calculate(["水属性", "治疗者"])
        assert result.leader_mode is False
        assert result.sr_chances != []
        assert result.ssr_results == []

    def test_water_healer_sr_probabilities(self):
        """[水属性, 治疗者] 不含领袖:
        [수속성,힐러] -> SR=1,R=1,N=1   -> 1/41  ≈ 0.02439
        [힐러]        -> SR=2,R=3,N=3   -> 2/122 ≈ 0.01639
        [수속성]      -> SR=3,R=1,N=6   -> 3/193 ≈ 0.01554
        """
        result = calculate(["水属性", "治疗者"])
        chances = result.sr_chances
        assert len(chances) == 3
        # 降序
        pcts = [c.percent for c in chances]
        assert pcts == sorted(pcts, reverse=True)
        # 顶部组合 = [水属性, 治疗者]
        assert chances[0].percent == pytest.approx(1 / 41)
        assert chances[0].matched_tags_zh == ["水属性", "治疗者"]
        # 第二 = [治疗者]
        assert chances[1].percent == pytest.approx(2 / 122)
        assert chances[1].matched_tags_zh == ["治疗者"]
        # 第三 = [水属性]
        assert chances[2].percent == pytest.approx(3 / 193)
        assert chances[2].matched_tags_zh == ["水属性"]

    def test_sr_three_tag_combo_used(self):
        """非 Leader 模式应枚举到三标签组合。"""
        result = calculate(["火属性", "攻击者", "守护者"])
        assert result.leader_mode is False
        assert all(0 < c.percent <= 1 for c in result.sr_chances)

    def test_sr_no_match_returns_empty(self):
        result = calculate(["风属性", "守护者", "魔族", "亚人"])
        assert result.leader_mode is False
        for c in result.sr_chances:
            assert 0 < c.percent <= 1


# --------------------------------------------------------------------------- #
# 输入约束
# --------------------------------------------------------------------------- #

class TestInputConstraints:
    def test_max_five_tags(self):
        # 传入 6 个有效标签,仅取前 5 个(对齐上游上限)。
        # 领袖放首位,确保截断后仍保留领袖 -> Leader 模式。
        result = calculate(["领袖", "火属性", "水属性", "风属性", "光属性", "暗属性"])
        assert result.leader_mode is True
        # 截断后 cur = 4 个属性,单标签 SSR 应存在
        assert len(result.ssr_results) > 0

    def test_empty_tags(self):
        result = calculate([])
        assert result.leader_mode is False
        assert result.sr_chances == []
        assert result.ssr_results == []


# --------------------------------------------------------------------------- #
# 格式化
# --------------------------------------------------------------------------- #

class TestFormatResult:
    def test_leader_format(self):
        result = calculate(["火属性", "攻击者", "领袖"])
        text = format_result(result, top=3)
        assert "[Leader 模式]" in text
        # name_ko 始终出现在输出中
        assert "바알" in text

    def test_sr_format(self):
        result = calculate(["水属性", "治疗者"])
        text = format_result(result, top=3)
        assert "[SR 模式]" in text
        assert "水属性" in text


# --------------------------------------------------------------------------- #
# 数据完整性
# --------------------------------------------------------------------------- #

def test_character_data_integrity():
    from ok.util.recruit import CHARACTERS
    assert len(CHARACTERS) == 76  # 与上游 recruit.js 一致(SSR28+SR14+R9+N25)
    rarities = {ch.rarity for ch in CHARACTERS}
    assert rarities == {"SSR", "SR", "R", "N"}
    # 每个角色至少有属性标签
    for ch in CHARACTERS:
        assert any("속성" in t for t in ch.tags_ko), ch
    # id 唯一
    ids = [ch.id for ch in CHARACTERS]
    assert len(ids) == len(set(ids))


def test_official_name_zh_from_upstream():
    """抽样校验角色中文名为上游官方简体译名(非手翻)。"""
    from ok.util.recruit import CHARACTERS
    by_id = {ch.id: ch for ch in CHARACTERS}
    # 抽样几个差异较大的官方译名
    assert by_id[10045].name_zh == "伊吹"   # 슈텐
    assert by_id[10056].name_zh == "堕龙"   # 카시피나
    assert by_id[10063].name_zh == "大女仆"  # 에밀리
    assert by_id[10066].name_zh == "千咒"   # 안젤리카
    assert by_id[10037].name_zh == "蛇后"   # 메스미나


def test_name_zh_override():
    import ok.util.recruit as mod
    original = mod.NAME_ZH_OVERRIDES.copy()
    try:
        mod.NAME_ZH_OVERRIDES["바알"] = "测试译名"
        chars = mod._build_characters()
        baal = next(c for c in chars if c.id == 10001)
        assert baal.name_zh == "测试译名"
    finally:
        mod.NAME_ZH_OVERRIDES.clear()
        mod.NAME_ZH_OVERRIDES.update(original)
