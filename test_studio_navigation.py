import unittest
from types import SimpleNamespace
from studio_navigation import navigate_tab

class NavigationTests(unittest.TestCase):
    def test_navigation_is_sent_only_to_requesting_client_and_repeated_actions_are_distinct(self):
        sent=[]
        client=SimpleNamespace(_websock_connection=SimpleNamespace(queue_message=sent.append))
        tabs=SimpleNamespace(_impl=SimpleNamespace(uuid='tabs'))
        navigate_tab(tabs, 0, client)
        navigate_tab(tabs, 0, client)
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0].uuid, 'tabs')
        self.assertEqual(sent[0].updates['_studio_tab_request']['index'], 0)
        self.assertNotEqual(sent[0].updates, sent[1].updates)
