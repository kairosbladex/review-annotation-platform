import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from luna_person_model_compare import image_contract, plan


class ContractTests(unittest.TestCase):
    def test_plan_has_two_complete_rounds_and_reversed_view_order(self):
        units = [{'unit': str(i)} for i in range(10)]
        jobs = plan(units)
        self.assertEqual(len(jobs), 40)
        self.assertEqual([j[0] for j in jobs[:2]], ['SOL-A1', 'SOL-H1'])
        self.assertEqual([j[0] for j in jobs[20:22]], ['SOL-H2', 'SOL-A2'])
        self.assertEqual(len({(j[0],j[1]['unit']) for j in jobs}), 40)

    def test_input_swap_is_rejected_before_model_call(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for i,color in enumerate(('red','green','blue')):
                p=Path(d)/f'{i}.png';Image.new('RGB',(32,24),color).save(p);paths.append(p)
            expected=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            command=['codex','-m','gpt-5.6-sol']+sum((['-i',str(p)] for p in paths),[])
            self.assertEqual(image_contract(command,expected,'gpt-5.6-sol')['image_count'],3)
            Image.new('RGB',(32,24),'white').save(paths[2])
            with self.assertRaises(ValueError):image_contract(command,expected,'gpt-5.6-sol')


if __name__=='__main__':unittest.main()
