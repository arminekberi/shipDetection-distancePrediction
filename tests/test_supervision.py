import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from boatdet.supervision import (StateLog, load_manifest, prepare, read_samples,
                                 reference_position, sequences, truth, write_json)
from rig_calibration import fit, evaluate
from scripts.demo_rig_calibration import generate, write_csv


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = generate(self.root/'source', duration=12)

    def prepared(self):
        return prepare(self.manifest,self.root/'prepared')

    def test_affine_clocks_match_same_video_instant(self):
        manifest=load_manifest(self.manifest)
        session=manifest['sessions'][0]
        spec=session['observer']
        log=StateLog(spec['path'],spec['clock'],.6)
        self.assertAlmostEqual(log.at(1.125)['x_m'],.225)
        self.assertIsNone(log.at(-.01))
        self.assertIsNone(log.at(12.01))

    def test_heading_interpolates_across_wrap_and_not_through_bad_fix(self):
        spec=load_manifest(self.manifest)['sessions'][0]['observer']
        path=Path(spec['path'])
        with path.open() as stream: rows=list(csv.DictReader(stream))[:3]
        rows[0]['heading_rad']=math.radians(179)
        rows[1]['heading_rad']=math.radians(-179)
        write_csv(path,rows)
        log=StateLog(path,spec['clock'],.6)
        self.assertAlmostEqual(abs(log.at(.125)['heading_rad']),math.pi)
        rows[1]['valid']=0
        write_csv(path,rows)
        log=StateLog(path,spec['clock'],.6)
        self.assertIsNone(log.at(.125))
        self.assertIsNone(log.at(.25))
        self.assertIsNone(log.at(.375))

    def test_gap_and_duplicate_timestamps_rejected(self):
        spec=load_manifest(self.manifest)['sessions'][0]['observer']
        self.assertIsNone(StateLog(spec['path'],spec['clock'],.1).at(.125))
        path=Path(spec['path'])
        with path.open() as stream: rows=list(csv.DictReader(stream))[:2]
        rows[1]['timestamp']=rows[0]['timestamp']
        write_csv(path,rows)
        with self.assertRaisesRegex(ValueError,'strictly increasing'):
            StateLog(path,spec['clock'],.6)

    def test_lever_arm_and_horizontal_slant_range_are_distinct(self):
        state={'x_m':0,'y_m':0,'z_m':0,'roll_rad':0,'pitch_rad':0,'heading_rad':math.pi/2}
        np.testing.assert_allclose(reference_position(state,[2,1,-3]),[1,2,3],atol=1e-10)
        spec=load_manifest(self.manifest)['sessions'][0]['observer']
        own=StateLog(spec['path'],spec['clock'],.6).at(0)
        target={**own,'x_m':3,'y_m':4}
        gt=truth(own,target,[0,0,-12],[0,0,0])
        self.assertAlmostEqual(gt['horizontal_range_m'],5)
        self.assertAlmostEqual(gt['slant_range_m'],13)
        self.assertAlmostEqual(gt['relative_bearing_deg'],-53.1301023542)

    def test_split_leakage_and_ambiguous_associations_fail(self):
        original=json.loads(self.manifest.read_text())
        modified=copy.deepcopy(original)
        modified['sessions'][1]['group']=modified['sessions'][0]['group']
        write_json(self.manifest,modified)
        with self.assertRaisesRegex(ValueError,'cannot cross'):
            load_manifest(self.manifest)
        modified=copy.deepcopy(original)
        modified['sessions'][0]['associations']*=2
        write_json(self.manifest,modified)
        with self.assertRaisesRegex(ValueError,'unambiguous'):
            load_manifest(self.manifest)

    def test_predicted_rows_excluded_and_changed_dataset_detected(self):
        rows,report=self.prepared()
        self.assertEqual(report['rejected']['predicted'],6)
        self.assertEqual(len(rows),141)
        self.assertTrue(report['synthetic'])
        path=self.root/'prepared/samples.jsonl'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'changed'):
            read_samples(path.parent)

    def test_fit_recovers_known_scale_bias_without_validation_truth(self):
        rows,provenance=self.prepared()
        model=fit(rows,provenance)
        params=model['models']['synthetic-camera']
        self.assertAlmostEqual(params['bearing_bias_deg'],-1.7,delta=.1)
        self.assertAlmostEqual(params['depth']['scale'],1.3,delta=.1)
        changed=copy.deepcopy(rows)
        for row in changed:
            if row['split'] != 'train':
                row['truth']['horizontal_range_m']=1e9
                row['truth']['relative_bearing_deg']=177
        self.assertEqual(fit(changed,provenance),model)
        result=evaluate(rows,model)
        self.assertEqual(result['split'],'val')
        self.assertLess(result['sessions'][0]['bearing_deg']['mae'],.3)

    def test_sequences_are_causal_and_target_state_is_never_an_input(self):
        rows,_=self.prepared()
        sequences(rows,self.root/'sequences',window=4,max_gap_s=.5)
        changed=copy.deepcopy(rows)
        for row in changed:
            row['truth']['target_x_m']+=100
            row['truth']['target_vx_mps']+=7
            row['inputs']['visual_depth_m']=99999
        sequences(changed,self.root/'changed',window=4,max_gap_s=.5)
        with np.load(self.root/'sequences/train.npz') as a, np.load(self.root/'changed/train.npz') as b:
            np.testing.assert_array_equal(a['X'],b['X'])
            np.testing.assert_allclose(b['y'][:,0]-a['y'][:,0],100,atol=1e-5)
            np.testing.assert_allclose(b['y'][:,2]-a['y'][:,2],7)
            self.assertEqual(a['X'].shape[1:],(4,9))
        self.assertEqual(json.loads((self.root/'sequences/schema.json').read_text())['uses_target_state_as_input'],False)

    def test_pixel_bearing_uses_scaled_intrinsics_and_synchronized_attitude(self):
        manifest=json.loads(self.manifest.read_text())
        session=manifest['sessions'][0]
        session.update(bearing_source='pixels_with_state_attitude',camera_calibration='camera.json',working_size=[640,360])
        write_json(self.root/'source/camera.json',{'camera_matrix':[[1200,0,640],[0,1200,360],[0,0,1]],
                                                'camera_height_m':1,'image_size':[1280,720]})
        path=self.root/'source'/session['observations']
        with path.open() as stream: observations=list(csv.DictReader(stream))
        for row in observations:
            row['visual_center_u']=380
            row['visual_center_v']=180
            row['visual_bearing_deg']=''  # Must derive from pixels, not the existing log bearing.
        write_csv(path,observations)
        write_json(self.manifest,manifest)
        rows,_=self.prepared()
        self.assertAlmostEqual(rows[0]['inputs']['visual_bearing_deg'],math.degrees(math.atan(.1)))


if __name__ == '__main__':
    unittest.main()
