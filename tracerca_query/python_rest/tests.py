import unittest
from pprint import pprint
from pathlib import Path

from query.models import TracesQueryParams
from query._utils import recurse_dict_print, load_pickle_file
from query.jaeger_query import JaegerQuery

class QueryTests(unittest.TestCase):
    host = "localhost"
    port = "16686"
    
    params = {
        "start": "1673578635204000",
        "end": "1673582235204000", # 1_570_438_141_766_835 1_673_582_235_204_000
        "loop": "1h",
        "limit": "1",
        "service": "gateway" 
    }
    tracesQueryParams = TracesQueryParams(**params)

        
    def test_services_query(self):
        jq = JaegerQuery(self.host, self.port)

        res = jq.query_services()

        self.assertTrue(isinstance(res, list))


    def test_find_traces_query(self):
        jq = JaegerQuery(self.host, self.port)

        res = jq.query_traces(self.tracesQueryParams)

        self.assertTrue(isinstance(res[0], dict))

    def test_find_traces_query_res_structure(self):
        jq = JaegerQuery(self.host, self.port)

        res = jq.query_traces(self.tracesQueryParams)

        self.assertEqual(recurse_dict_print(res), None)

    def test_find_traces_query_res_destructure(self):
        jq = JaegerQuery(self.host, self.port)

        res = jq.parse_jaeger_traces(self.tracesQueryParams)
        # spans = res[0]
        # spans = spans["spans"]

        # spans = sorted(spans, key=lambda x: x["startTime"]+x["duration"])
        # pprint(spans[1]["duration"])
        # pprint(spans[1]["startTime"])
        # pprint(spans[1]["tags"])
        # pprint(spans[0]["processID"])
        # pprint(spans[1]["processID"])
        # pprint(spans[2]["processID"])
        # pprint(spans[1]["logs"])

        self.assertTrue(True)

class UtilTests(unittest.TestCase):
    def test_recursive_dict(self):
        test_obj = {
            "depth1": "literal",
            "depth2": {"depth2.1": "literal"},
            "depth3": {"depth3.1": {"depth3.1.1": "literal"}},
        }

        recurse_dict_print(test_obj)
        self.assertTrue(True)

    def test_recursive_list(self):
        test_obj = {
            "to_depth1": "literal",
            "to_depth2": {"to_depth2.1": "literal"},
            "to_depth3": {"to_depth3.1": {"to_depth3.1.1": "literal"}},
        }
        test_obj_with_list = {
            "depth1": [test_obj, test_obj],
            "depth2": {"to_depth2.1": [test_obj]},
        }

        recurse_dict_print(test_obj_with_list)
        self.assertTrue(True)

    def test_target_trace_structure(self):
        input = Path("./example_data/uninjected_trace.pkl")
        pickle_content = load_pickle_file(input)
        for content in pickle_content:
            pprint(content["label"])

        input = Path("./example_data/uninjected_trace.pkl")
        pickle_content = load_pickle_file(input)
        for content in pickle_content:
            pprint(content["label"])

if __name__ == "__main__":
    unittest.main()
