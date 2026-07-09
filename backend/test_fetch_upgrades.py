import unittest
from unittest.mock import patch

import requests

from scraper.fetch import fetch_source


def make_response(url: str, status_code: int, body: str, content_type: str = "text/html"):
    response = requests.Response()
    response.status_code = status_code
    response.reason = "OK" if status_code < 400 else "Error"
    response.url = url
    response._content = body.encode("utf-8")
    response.headers["Content-Type"] = content_type
    return response


class FetchUpgradeTests(unittest.TestCase):
    @patch("scraper.fetch.time.sleep")
    @patch("scraper.fetch.requests.get")
    def test_retries_transient_errors(self, mock_get, _mock_sleep):
        source = {"url": "https://example.com/doc"}
        mock_get.side_effect = [
            make_response(source["url"], 503, "temporary outage"),
            make_response(source["url"], 200, "<main><h1>Recovered</h1></main>"),
        ]

        result = fetch_source(source, max_retries=2, respect_robots=False)

        self.assertTrue(result.success)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(mock_get.call_count, 2)

    @patch("scraper.fetch.time.sleep")
    @patch("scraper.fetch.requests.get")
    def test_github_repo_root_falls_back_from_main_to_master(self, mock_get, _mock_sleep):
        source = {"url": "https://github.com/example/project"}
        mock_get.side_effect = [
            make_response(
                "https://raw.githubusercontent.com/example/project/main/README.md",
                404,
                "not found",
            ),
            make_response(
                "https://raw.githubusercontent.com/example/project/master/README.md",
                200,
                "# Project\n\nREADME body",
                "text/plain",
            ),
        ]

        result = fetch_source(source, max_retries=1, respect_robots=False)

        self.assertTrue(result.success)
        self.assertEqual(result.content_type, "markdown")
        self.assertEqual(result.url, "https://raw.githubusercontent.com/example/project/master/README.md")

    @patch("scraper.fetch.requests.get")
    def test_detects_anti_bot_page(self, mock_get):
        source = {"url": "https://example.com/doc"}
        mock_get.return_value = make_response(
            source["url"],
            200,
            "<html><title>Verify you are human</title></html>",
        )

        result = fetch_source(source, max_retries=1, respect_robots=False)

        self.assertFalse(result.success)
        self.assertIn("anti-bot", result.error)


if __name__ == "__main__":
    unittest.main()
