"""验证所有视图来自干净原帧、使用投影坐标，不进行模型调用。"""
import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw
from luna_clean_only_eval import request_payload, sha
from luna_prompt_autotune import luna_experiment_worker as worker
from score_luna_clean_only import paired, score, CONDITIONS


class CleanOnlyTests(unittest.TestCase):
    def test_six_way_join_and_incomplete_pair_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';new=Path(tmp)/'new';base.mkdir();new.mkdir()
            def put(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            put(base/'included.json',[{'unit':'one','event_id':'event','split':'holdout','clean_sha256':'same',
                'source_local_path':'video','source_kind':'video','candidates':[{'key':'box','class_name':'人员'}]}])
            put(base/'labels.json',{'box':{'label':'false','split':'holdout','class_name':'人员'}})
            for condition in CONDITIONS:
                is_new=condition.startswith('clean_only')
                put((new if is_new else base)/'results'/condition/'one.json',{
                    'success':True,'execution_id':condition,'prompt_sha256':'prompt','image_sha256':'same',
                    'requested_model':'test','attempts':[{'status':'ok','seconds':1}],
                    'results':[{'key':'box','decision':'false' if is_new else 'true'}]})
            score(new,base);report=json.loads((new/'report.json').read_text())
            self.assertEqual(report['scored_candidates'],1)
            self.assertEqual(report['splits']['holdout']['人员']['clean_only-1']['false_accepted'],0)
            (new/'results/clean_only-2/one.json').unlink()
            score(new,base);report=json.loads((new/'report.json').read_text())
            self.assertEqual(report['scored_candidates'],0);self.assertEqual(report['incomplete_images'],['one'])

    def test_false_acceptance_delta_has_correct_sign(self):
        rows=[{'label':'false','event_id':'one','source_local_path':'video','decisions':{
            'screenshot-1':'true','screenshot-2':'true','plus_clean-1':'true','plus_clean-2':'true',
            'clean_only-1':'false','clean_only-2':'uncertain'}}]
        result=paired(rows,'plus_clean')
        self.assertEqual(result['mean_false_acceptance_rate_delta_pp'],-100)
        self.assertEqual(result['false_acceptance_rate_delta_pp_cluster_bootstrap95'],[-100,-100])

    def fixture(self, root):
        (root/'clean').mkdir()
        image = Image.new('RGB', (100, 100), 'blue')
        ImageDraw.Draw(image).rectangle((60, 60, 80, 80), fill='lime')
        path = root/'clean/one.jpg'; image.save(path, quality=95)
        unit = {'unit': 'one', 'clean_sha256': sha(path), 'candidates': [
            {'key': 'private-id', 'class_name': '车', 'bbox': [.1,.1,.2,.2], 'clean_bbox': [.6,.6,.8,.8], 'label': 'true'}]}
        return unit, {'common': '按可见证据判断', 'cards': {'车': '识别车辆'}}

    def test_request_uses_projection_and_no_fourth_image_or_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);unit,doc=self.fixture(root);p=request_payload(root,unit,doc)
            self.assertEqual(p['candidates'][0]['bbox'], [.6,.6,.8,.8])
            self.assertNotIn('clean_image',p)
            self.assertNotIn('private-id',p['prompt']);self.assertNotIn('label',p['prompt'])
            self.assertEqual(base64.b64decode(p['image']), (root/'clean/one.jpg').read_bytes())

    def test_modified_native_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);unit,doc=self.fixture(root)
            (root/'clean/one.jpg').write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'clean_frame_hash_changed'):request_payload(root,unit,doc)

    def test_actual_worker_generates_three_views_from_projected_clean_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);unit,doc=self.fixture(root);p=request_payload(root,unit,doc)
            (root/'auth.json').write_text(json.dumps({'OPENAI_API_KEY':'test-placeholder-not-a-credential'}))
            captured=[]
            def fake(cmd, **kwargs):
                paths=[cmd[i+1] for i,x in enumerate(cmd) if x=='-i']
                captured.extend(Image.open(x).copy() for x in paths)
                process=Mock(returncode=0)
                event={'type':'item.completed','item':{'type':'agent_message','text':json.dumps({'results':[{'target_index':0,'decision':'true','reasoning':'测试'}]})}}
                process.communicate.return_value=(json.dumps(event),'')
                return process
            with patch.dict(os.environ,{'CODEX_REVIEW_BINARY':'unused-test-binary','CODEX_REVIEW_HOME':str(root),'CODEX_REVIEW_TMP_DIR':str(root)}),patch.object(worker.subprocess,'Popen',side_effect=fake):
                worker.execute(p)
            self.assertEqual(len(captured),3)
            original,locator,crops=captured
            self.assertGreater(original.getpixel((70,70))[1],200)
            r,g,b=locator.getpixel((60,70));self.assertGreater(r,150);self.assertGreater(g,150);self.assertLess(b,min(r,g))
            self.assertGreater(crops.getpixel((30,54))[1],200)


if __name__=='__main__':unittest.main()
