import unittest
import base64
import io
from unittest.mock import patch
import numpy as np
from PIL import Image
from luna_person_ablation import project_box, zoom_sheet, zoom_payload
from luna_person_score import score


class ViewTests(unittest.TestCase):
    def test_zoom_consumes_exact_input_pixels_without_jpeg_generation(self):
        pixels=np.random.default_rng(41).integers(0,256,(100,100,3),dtype=np.uint8)
        image=Image.fromarray(pixels); stream=io.BytesIO(); image.save(stream,format='PNG')
        payload={'image':base64.b64encode(stream.getvalue()).decode(),'candidates':[{'target_index':0,'bbox':[.4,.4,.6,.6]}]}
        with patch('luna_person_ablation.zoom_sheet',return_value=image) as renderer:
            zoom_payload(payload)
            np.testing.assert_array_equal(np.asarray(renderer.call_args.args[0]),pixels)

    def test_tiny_target_is_larger_without_changing_source(self):
        image = Image.new('RGB', (100,100), 'gray')
        image.paste((255,0,0), (47,46,53,54))
        original = image.tobytes()
        candidates = [{'target_index':0,'bbox':[.47,.46,.53,.54]}]
        # Existing worker calls thumbnail: it preserves this target at 6x8 pixels.
        old = image.crop((27,26,73,74)); old.thumbnail((256,256))
        self.assertEqual(old.size,(46,48))
        result = zoom_sheet(image,candidates)
        pixels = np.array(result)
        ys,xs = np.where((pixels[:,:,0]>220)&(pixels[:,:,1]<35)&(pixels[:,:,2]<35))
        self.assertGreaterEqual(xs.max()-xs.min()+1,20)
        self.assertGreaterEqual(ys.max()-ys.min()+1,28)
        self.assertEqual(image.tobytes(),original)
        self.assertEqual(result.size,(256,280))

    def test_multi_target_sheet_and_edges(self):
        image=Image.new('RGB',(100,100),'gray')
        cs=[{'target_index':i,'bbox':[0,0,.1,.1]} for i in range(7)]
        self.assertEqual(zoom_sheet(image,cs).size,(768,840))

    def test_projection_preserves_frame_conversion(self):
        actual=project_box([.1,.2,.4,.5],[[2,0,0],[0,1.5,0],[0,0,1]],[960,720],[1920,1080])
        np.testing.assert_allclose(actual,[.1,.2,.4,.5])
        with self.assertRaises(ValueError):
            project_box([0,0,.1,.1],[[2,0,-2],[0,1.5,0],[0,0,1]],[960,720],[1920,1080])

    def test_scoring_maps_reversed_ids_and_preserves_missing(self):
        units=[{'unit':'synthetic','candidates':[{'gold':'true'},{'gold':'false'}]}]
        records={'synthetic':{'success':True,'result':{'results':[
            {'target_index':1,'decision':'false'},{'target_index':0,'decision':'true'}]}}}
        self.assertEqual(score(units,records)['tp'],1)
        self.assertEqual(score(units,records)['fp'],0)
        failed=score(units,{'synthetic':{'success':False}})
        self.assertEqual(failed['missing'],2)
        self.assertEqual(failed['true:false'],0)

    def test_invalid_index_is_rejected_instead_of_scored(self):
        units=[{'unit':'synthetic','candidates':[{'gold':'true'}]}]
        for index in (True,3):
            with self.assertRaises(AssertionError):
                score(units,{'synthetic':{'success':True,'result':{'results':[{'target_index':index,'decision':'false'}]}}})


if __name__=='__main__':
    unittest.main()
