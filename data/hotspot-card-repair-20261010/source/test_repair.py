import ast,io,json,pathlib,unittest
from PIL import Image,ImageDraw
from input_views import make_sheet,project_box
ROOT=pathlib.Path(__file__).resolve().parents[1]
class RepairTests(unittest.TestCase):
 def test_legacy_sheet_byte_compatible(self):
  im=Image.new('RGB',(960,720),'green');ImageDraw.Draw(im).rectangle((300,200,310,215),fill='red');cs=[{'target_index':0,'class_name':'光斑','bbox':[.3,.2,.35,.3]}]
  ref=Image.new('RGB',(1280,280),'white');d=ImageDraw.Draw(ref)
  for i,c in enumerate(cs):
   b=c['bbox'];x1,y1,x2,y2=b[0]*im.width,b[1]*im.height,b[2]*im.width,b[3]*im.height;mx,my=max(20,(x2-x1)*.25),max(20,(y2-y1)*.25)
   crop=im.crop((max(0,x1-mx),max(0,y1-my),min(im.width,x2+mx),min(im.height,y2+my)));crop.thumbnail((256,256));ref.paste(crop,((i%5)*256,(i//5)*280+24));d.text(((i%5)*256,(i//5)*280),str(i),fill='black')
  got=make_sheet(im,cs,'legacy');self.assertEqual(got.tobytes(),ref.tobytes())
  ext=make_sheet(im,cs,'context');self.assertEqual(ext.crop((0,0,1280,280)).tobytes(),ref.tobytes())
 def test_clipped_and_invalid_projection(self):
  H=[[1,0,0],[0,1,-10],[0,0,1]];r=project_box([.2,0,.3,.2],H,(100,100),(100,100));self.assertFalse(r['complete']);self.assertAlmostEqual(r['retained_fraction'],.5)
  good=project_box([.2,.2,.3,.3],H,(100,100),(100,100));self.assertTrue(good['complete']);self.assertEqual(good['bbox'],[.2,.1,.3,.2])
  with self.assertRaises(ValueError):project_box([0,0,.1,.1],[[1,0,0],[0,1,0],[0,0,0]],(100,100),(100,100))
 def test_prompt_builder_unchanged(self):
  def fn(p):return ast.dump(next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='build_prompt'))
  self.assertEqual(fn(ROOT/'source/production_worker.py'),fn(ROOT/'source/experiment_worker.py'))
 def test_no_holdout_and_no_label_mutation(self):
  units=json.loads((ROOT/'split.json').read_text());labels=json.loads((ROOT/'labels.json').read_text());self.assertEqual(len(units),7);self.assertTrue(all(u['split']=='development' for u in units));self.assertEqual({b['key'] for u in units for b in u['boxes']},{r['key'] for r in labels});self.assertEqual(len(labels),20)
if __name__=='__main__':unittest.main()
