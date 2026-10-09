"""提示词组装、合格标准与 Hermes 提交工具的定向测试：python3 -m unittest test_autotune"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import metrics
import promptlib

HERE = Path(__file__).resolve().parent
OLD = ('逐一核查候选框内目标是否符合初审类别，不把框线、文字当作目标证据。编号图只用于定位；若有第四张原图，只有能唯一对应同一目标时辅助判断。'
       '看不清、遮挡、不能唯一定位则输出uncertain，明确符合类别才输出true，明显不符合输出false。'
       '车辆细分类必须看见所需结构，三轮车须有三轮证据；人员须有可见人体结构。光伏板须有规则组件结构，不推断热故障。'
       '违建没有审批或区域业务依据，一律uncertain；游商等违法属性不能仅凭外观判断。车牌不读号码。'
       '不运行任何工具，只返回schema中的JSON；target_index必须使用下面原编号，解释使用中文。候选：')


class Tests(unittest.TestCase):
    def test_p000_identical_to_production(self):
        c = [{'target_index': 0, 'class_name': '车', 'bbox': [0.1, 0.1, 0.2, 0.2]}]
        self.assertEqual(promptlib.build(json.loads((HERE / 'prompts/p000.json').read_text()), c), OLD + json.dumps(c, ensure_ascii=False))

    def test_only_present_cards_and_aliases(self):
        doc = json.loads((HERE / 'prompts/p001.json').read_text())
        text = promptlib.build(doc, [{'target_index': 0, 'class_name': '路面裂纹', 'bbox': [0, 0, 1, 1]}])
        self.assertIn('[道路裂纹]', text); self.assertNotIn('[人员]', text); self.assertTrue(text.split(promptlib.SUFFIX)[0])
        self.assertEqual(promptlib.validate(doc), [])

    def test_validate_rejects_forcing_and_unknown(self):
        errs = promptlib.validate({'common': 'x', 'cards': {'人员': '看到头顶必须输出true', '飞机': 'y'}})
        self.assertTrue(any('必须输出true' in e for e in errs)); self.assertTrue(any('飞机' in e for e in errs))

    def test_gates(self):
        labels = {f'k{i}': {'label': 'true' if i < 6 else 'false', 'class_name': '人员', 'split': 'dev'} for i in range(12)}
        base_res = {f'k{i}': ('uncertain' if i < 6 else 'false') for i in range(12)}
        cand_res = {f'k{i}': 'true' for i in range(12)}
        items = [{'split': 'unlabeled', 'candidates': [{'key': 'u1', 'class_name': '道路裂纹'}]}]
        base = {'dev': metrics.score(base_res, labels, 'dev'), 'unlabeled': metrics.unlabeled({'u1': 'uncertain'}, items)}
        cand = {'dev': metrics.score(cand_res, labels, 'dev'), 'unlabeled': metrics.unlabeled({'u1': 'true'}, items)}
        g = metrics.gates(cand, base)
        self.assertFalse(g['gates']['G3_人员非人放过不比基线多3个以上']); self.assertFalse(g['passed'])
        self.assertEqual(base['dev']['by_class']['人员']['true_rejected'], 6)

    def test_submit_prompt_tool(self):
        try:
            import importlib
            os.environ['AUTOTUNE_STATE'] = tempfile.mkdtemp()
            import mcp_server; importlib.reload(mcp_server)
        except ImportError:
            self.skipTest('mcp 未安装')
        st = Path(os.environ['AUTOTUNE_STATE'])
        for v in ('p000', 'p001'):
            (st / 'prompts').mkdir(exist_ok=True); (st / 'rounds').mkdir(exist_ok=True)
            (st / 'prompts' / f'{v}.json').write_text((HERE / f'prompts/{v}.json').read_text())
            (st / 'rounds' / f'{v}.json').write_text(json.dumps({'status': 'done', 'submitted_at': '2026-10-09T00:00:00'}))
        r = mcp_server.submit_prompt('p001', '改人员卡', {'人员': '新的人员卡'})
        self.assertTrue(r['accepted']); self.assertEqual(r['version'], 'p002'); self.assertEqual(r['changed'], ['人员'])
        self.assertFalse(mcp_server.submit_prompt('p001', '再来', {'车': '新车卡'})['accepted'])  # 已有排队
        (st / 'STOP').write_text('test')
        self.assertTrue(mcp_server.get_state()['stop_requested'])


if __name__ == '__main__':
    unittest.main()
