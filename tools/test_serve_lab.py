import http.client
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from serve_lab import RangeHandler


class RangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        Path(cls.folder.name, 'video.mp4').write_bytes(b'0123456789')
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(RangeHandler, directory=cls.folder.name))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.folder.cleanup()

    def request(self, value=None, method='GET'):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        client.request(method, '/video.mp4', headers={'Range': value} if value else {})
        response = client.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        client.close()
        return result

    def test_partial(self):
        status, headers, body = self.request('bytes=2-5')
        self.assertEqual((status, body), (206, b'2345'))
        self.assertEqual(headers['Content-Range'], 'bytes 2-5/10')
        self.assertEqual(headers['Content-Length'], '4')

    def test_suffix_and_open_end(self):
        self.assertEqual(self.request('bytes=-3')[2], b'789')
        self.assertEqual(self.request('bytes=8-')[2], b'89')

    def test_unsatisfiable(self):
        self.assertEqual(self.request('bytes=10-')[0], 416)
        self.assertEqual(self.request('bytes=8-2')[0], 416)

    def test_full_and_head(self):
        self.assertEqual(self.request()[2], b'0123456789')
        status, headers, body = self.request('bytes=2-5', method='HEAD')
        self.assertEqual((status, body), (206, b''))
        self.assertEqual(headers['Content-Length'], '4')
