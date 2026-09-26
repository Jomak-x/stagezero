"""Archive retention tests independent of GPU and HTTP timing."""
import io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from swing_server import LiveRun
class ArchiveTests(unittest.TestCase):
    def test_recorded_interval_remains_immutable_after_playback(self):
        with tempfile.TemporaryDirectory() as d:
            run=LiveRun.__new__(LiveRun)
            run.log=io.StringIO();run.run_id='test';run.run_dir=Path(d)
            run.time=12.;run.frame=720;run.commands=[];run.metrics={};run.events=[]
            run.scene={'provenance':{'sha256':'source'}}
            run.controller=SimpleNamespace(snapshot=lambda:{'phase':'kiss'})
            run.recording_sealed=False
            run.save(recording=True)
            self.assertTrue(run.recording_sealed)
            before=(run.run_dir/'report.json').read_bytes()
            run.time=30;run.frame=1800;run.save()
            self.assertEqual(before,(run.run_dir/'report.json').read_bytes())
            self.assertEqual(before,(run.run_dir/'recorded-report.json').read_bytes())
            self.assertEqual(json.loads((run.run_dir/'latest-state-report.json').read_text())['frames'],1800)
if __name__=='__main__':unittest.main()
