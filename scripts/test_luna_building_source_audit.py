import importlib.util,pathlib,unittest
p=pathlib.Path(__file__).with_name('prepare_luna_building_release_validation.py');s=importlib.util.spec_from_file_location('prep',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class SourceAudit(unittest.TestCase):
 def test_task_id_misnamed_as_event_resolves_by_object(self):
  row={'source_event_id':'task-not-event','image_object_key':'/bucket/image.jpg'}
  self.assertEqual(m.gold_source_event(row,{'bucket/image.jpg':{'event_id':'actual-event'}}),'actual-event')
 def test_url_and_object_path_are_same_source(self):
  self.assertEqual(m.norm('https://host/minio-src/bucket/raw.mp4'),m.norm('/bucket/raw.mp4'))
 def test_unresolved_is_not_an_invented_event(self):
  self.assertIsNone(m.gold_source_event({'source_event_id':'task','image_object_key':'unknown'},{}))
if __name__=='__main__':unittest.main()
