import csv,importlib.util,json,pathlib,shutil,subprocess,sys,tempfile,time,unittest,urllib.request,urllib.error
from PIL import Image
P=pathlib.Path(__file__).parent
s=importlib.util.spec_from_file_location('labels',P/'label_server.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.p=pathlib.Path(self.tmp.name);self.old=M.HERE;M.HERE=self.p
  self.row={'key':'fixture_000','event_id':'fixture','task_id':'task','task_name':'测试任务','event_time':'2026-10-10','raw_point':[100,100],'raw_size':[400,400],'screenshot_size':[400,400],'homography':[[1,0,0],[0,1,0],[0,0,1]],'raw_image':'images/raw.jpg','screenshot_image':'images/shot.jpg'}
  (self.p/'targets.json').write_text(json.dumps({'items':[self.row]}));(self.p/'images').mkdir();im=Image.new('RGB',(400,400),'gray');im.save(self.p/'images/raw.jpg');im.save(self.p/'images/shot.jpg');shutil.copy(P/'index.html',self.p/'index.html')
  with (self.p/'annotations.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=M.FIELDS);w.writeheader();w.writerow({'key':'fixture_000'})
 def tearDown(self):M.HERE=self.old;self.tmp.cleanup()
 def test_validation_and_persistence(self):
  for box in [None,[0,0,1,0],[0,0,float('nan'),1],[-.1,0,.1,.1]]:
   with self.assertRaises(ValueError):M.validate({'key':'fixture_000','label':'是光斑','bbox':box})
  r=M.validate({'key':'fixture_000','label':'是光斑','bbox':[.2,.2,.3,.3]});M.save(r);self.assertEqual(json.loads(M.load()['fixture_000']['screenshot_bbox']),[.2,.2,.3,.3]);self.assertEqual(M.load()['fixture_000']['label'],'是光斑')
  r=M.validate({'key':'fixture_000','label':'看不清','bbox':None});self.assertEqual(r['bbox'],'')
 def test_browser_and_request_boundaries(self):
  import threading
  server=M.ThreadingHTTPServer(('127.0.0.1',0),M.Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}'
  try:
   from playwright.sync_api import sync_playwright
   with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True);page=browser.new_page(viewport={'width':1280,'height':1000});page.goto(base);page.wait_for_function('crop !== null');page.get_by_role('button',name='1 是光斑',exact=True).click();self.assertIn('先在左侧',page.locator('#error').inner_text());b=page.locator('#canvas').bounding_box();page.mouse.move(b['x']+130,b['y']+130);page.mouse.down();page.mouse.move(b['x']+190,b['y']+190);page.mouse.up();page.get_by_role('button',name='1 是光斑',exact=True).click();page.wait_for_function("rows[0].label === '是光斑'");page.reload();page.wait_for_function('crop !== null');self.assertIn('已标 1',page.locator('#progress').inner_text());browser.close()
   self.assertEqual(M.load()['fixture_000']['label'],'是光斑')
   req=urllib.request.Request(base+'/api/label',data=b'{}',headers={'Content-Type':'application/json','Origin':'https://untrusted.invalid'})
   with self.assertRaises(urllib.error.HTTPError) as e:urllib.request.urlopen(req)
   self.assertEqual(e.exception.code,403)
   with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(base+'/images/../annotations.csv')
  finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':unittest.main()
